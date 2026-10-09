import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .config import ROOT

class Images(Dataset):
    def __init__(self, images, labels, indices):
        self.images, self.labels = images, labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = int(self.indices[index])
        x = self.images[original].reshape(-1).to(torch.float32) / 255.0
        return x, int(self.labels[original]), original

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

def encode(x, steps, max_steps, generator):
    """CPU Bernoulli draws in example-major order; common prefixes across T/batch size."""
    if x.device.type != "cpu" or not 0 < steps <= max_steps:
        raise ValueError("Encoding expects CPU images and 0 < steps <= max_steps")
    uniform = torch.rand((x.shape[0], max_steps, x.shape[1]), generator=generator)
    return (uniform[:, :steps] < x[:, None]).to(x.dtype).permute(1, 0, 2).contiguous()

