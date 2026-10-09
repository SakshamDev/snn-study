import hashlib
import json
import queue
import threading
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .config import ROOT

class Images(Dataset):
    def __init__(self, images, labels, indices):
        self.images, self.labels = images, labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self._gathered = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = int(self.indices[index])
        x = self.images[original].reshape(-1).to(torch.float32) / 255.0
        return x, int(self.labels[original]), original

    def _arrays(self):
        """Subset gathered once as flat uint8 pixels, int64 labels and original ids."""
        if self._gathered is None:
            index = torch.from_numpy(self.indices)
            self._gathered = (self.images[index].reshape(len(self), -1).contiguous(),
                              self.labels[index].to(torch.int64),
                              index.clone())
        return self._gathered

    def batches(self, batch_size, order=None):
        """Yield (float32 pixels / 255, labels, ids) exactly as a DataLoader would collate them."""
        pixels, labels, ids = self._arrays()
        order = torch.arange(len(self)) if order is None else order
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            yield pixels[index].to(torch.float32) / 255.0, labels[index], ids[index]

def stratified_indices(labels, train_size, val_size, seed):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) != 10 or train_size % 10 or val_size % 10:
        raise ValueError("Expected ten classes and subset sizes divisible by ten")
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in classes:
        ids = rng.permutation(np.flatnonzero(labels == label))
        nt, nv = train_size // 10, val_size // 10
        if len(ids) < nt + nv:
            raise ValueError("Not enough examples in a class")
        train.extend(ids[:nt])
        val.extend(ids[nt:nt + nv])
    return rng.permutation(train), rng.permutation(val)

def synthetic_partition(size, seed):
    """Ten easy spatial patterns, solely for offline software verification."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(size) % 10
    images = torch.randint(0, 25, (size, 784), generator=generator, dtype=torch.int64)
    for label in range(10):
        images[labels == label, label * 70:(label + 1) * 70] += 210
    return images.to(torch.uint8).reshape(size, 28, 28), labels

def load_data(cfg, download=False):
    root = ROOT / cfg.data_dir
    if cfg.dataset == "synthetic":
        train_images, train_labels = synthetic_partition(500, 7301)
        test_images, test_labels = synthetic_partition(100, 7302)
    else:
        from torchvision import datasets
        dataset = getattr(datasets, cfg.dataset)
        training = dataset(root=str(root), train=True, download=download)
        testing = dataset(root=str(root), train=False, download=download)
        train_images, train_labels = training.data, training.targets
        test_images, test_labels = testing.data, testing.targets
    data_hash = hashlib.sha256()
    for tensor in (train_images, train_labels, test_images, test_labels):
        data_hash.update(tensor.numpy().tobytes())
    dataset_digest = data_hash.hexdigest()
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"{cfg.dataset}_{cfg.split_seed}_{cfg.train_size}_{cfg.val_size}.npz"
    expected_train, expected_val = stratified_indices(train_labels.numpy(), cfg.train_size, cfg.val_size, cfg.split_seed)
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved:
            train_ids, val_ids = saved["train"], saved["validation"]
            if str(saved["dataset_sha256"].item()) != dataset_digest:
                raise RuntimeError("Dataset contents changed since the split was saved")
            if not np.array_equal(train_ids, expected_train) or not np.array_equal(val_ids, expected_val):
                raise RuntimeError("Saved split differs from the declared stratification procedure")
    else:
        train_ids, val_ids = expected_train, expected_val
        np.savez_compressed(split_path, train=train_ids, validation=val_ids, dataset_sha256=dataset_digest)
    split_digest = hashlib.sha256(train_ids.tobytes() + val_ids.tobytes()).hexdigest()
    manifest = {"dataset": cfg.dataset, "synthetic": cfg.dataset == "synthetic",
                "dataset_sha256": dataset_digest, "split_sha256": split_digest,
                "split_seed": cfg.split_seed, "train_size": len(train_ids), "val_size": len(val_ids),
                "test_size": len(test_labels), "preprocessing": "uint8 / 255; flattened 784; no augmentation"}
    return {
        "train": Images(train_images, train_labels, train_ids),
        "validation": Images(train_images, train_labels, val_ids),
        "test": Images(test_images, test_labels, np.arange(len(test_labels))),
    }, manifest

def loader(dataset, batch_size, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)

def shuffled_order(size, generator):
    """One epoch of the order DataLoader(shuffle=True, generator=generator) would visit.

    Per epoch, DataLoader draws one base seed, RandomSampler draws the permutation, and then
    RandomSampler draws a second, discarded zero-length remainder permutation. Mirroring all
    three draws keeps the shuffle stream identical across epochs.
    """
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    order = torch.randperm(size, generator=generator)
    torch.randperm(size, generator=generator)
    return order

def encode_bool(x, steps, max_steps, generator):
    """CPU Bernoulli draws in example-major order; common prefixes across T/batch size.

    Returns a boolean (steps, batch, features) tensor (4x cheaper to transfer than float).
    """
    if x.device.type != "cpu" or not 0 < steps <= max_steps:
        raise ValueError("Encoding expects CPU images and 0 < steps <= max_steps")
    uniform = torch.rand((x.shape[0], max_steps, x.shape[1]), generator=generator)
    return (uniform[:, :steps] < x[:, None]).permute(1, 0, 2).contiguous()

def encode(x, steps, max_steps, generator):
    """CPU Bernoulli draws in example-major order; common prefixes across T/batch size."""
    return encode_bool(x, steps, max_steps, generator).to(x.dtype)

def prepared_batches(dataset, batch_size, order, spiking, steps, max_steps, generator, pin=False):
    """Batches with CPU-side preparation (spike encoding for SNNs), in deterministic order."""
    for x, y, ids in dataset.batches(batch_size, order):
        payload = encode_bool(x, steps, max_steps, generator) if spiking else x
        yield (payload.pin_memory() if pin else payload), y, ids

def prefetch(iterable, depth=3):
    """Produce items on one background thread so CPU encoding overlaps with device compute.

    A single producer thread consumes the iterable strictly in order, so every random
    stream is advanced exactly as in a serial loop. Producer errors are re-raised.
    """
    buffer, stop, finished = queue.Queue(maxsize=depth), threading.Event(), object()

    def put(item):
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            for item in iterable:
                if not put((None, item)):
                    return
            put((None, finished))
        except BaseException as error:
            put((error, None))

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    try:
        while True:
            error, item = buffer.get()
            if error is not None:
                raise error
            if item is finished:
                return
            yield item
    finally:
        stop.set()
        worker.join()

