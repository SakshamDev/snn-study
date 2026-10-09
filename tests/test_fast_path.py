import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from surj_snn.data import (Images, encode, encode_bool, loader, prefetch, prepared_batches,
                           shuffled_order, synthetic_partition)
from surj_snn.experiment import parse_shard, shard_items
from surj_snn.models import SNN


def make_images(count=130):
    images, labels = synthetic_partition(count, 5)
    return Images(images, labels, np.random.default_rng(1).permutation(count)[:count - 7])


def test_fast_batches_match_dataloader_values_and_dtypes():
    data = make_images()
    reference = list(loader(data, 32))
    fast = list(data.batches(32))
    assert len(reference) == len(fast)
    for (rx, ry, ri), (fx, fy, fi) in zip(reference, fast):
        assert torch.equal(rx, fx) and torch.equal(ry, fy) and torch.equal(ri, fi)
        assert (rx.dtype, ry.dtype, ri.dtype) == (fx.dtype, fy.dtype, fi.dtype)


def test_shuffled_order_reproduces_dataloader_shuffle_across_epochs():
    data = make_images()
    reference_loader = loader(data, 16, shuffle=True, seed=20011)
    generator = torch.Generator().manual_seed(20011)
    for _ in range(3):
        expected = torch.cat([ids for _, _, ids in reference_loader])
        order = shuffled_order(len(data), generator)
        got = torch.cat([ids for _, _, ids in data.batches(16, order)])
        assert torch.equal(expected, got)


def test_encode_bool_matches_float_encode_and_stream():
    x = torch.rand(5, 9)
    a, b = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(encode_bool(x, 4, 8, a).float(), encode(x, 4, 8, b))
    assert torch.equal(torch.rand(3, generator=a), torch.rand(3, generator=b))


def test_prefetched_encoding_equals_serial_encoding():
    data = make_images()
    serial_generator = torch.Generator().manual_seed(8)
    serial = [encode_bool(x, 4, 8, serial_generator) for x, _, _ in data.batches(32)]
    threaded_generator = torch.Generator().manual_seed(8)
    threaded = [p for p, _, _ in prefetch(prepared_batches(data, 32, None, True, 4, 8, threaded_generator))]
    assert len(serial) == len(threaded)
    assert all(torch.equal(a, b) for a, b in zip(serial, threaded))


def test_prefetch_preserves_order_propagates_errors_and_stops_early():
    assert list(prefetch(iter(range(50)), depth=2)) == list(range(50))

    def broken():
        yield 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        list(prefetch(broken()))
    stream = prefetch(iter(range(10 ** 6)), depth=2)
    assert next(stream) == 0
    stream.close()


def reference_time_major_forward(model, spikes):
    """The original time-major implementation, kept as an oracle for the layer-major forward."""
    steps, batch, _ = spikes.shape
    membranes = [spikes.new_zeros((batch, width)) for width in model.widths[1:]]
    records = [[] for _ in model.layers]
    for time in range(steps):
        activation = spikes[time]
        for index, (layer, cell) in enumerate(zip(model.layers, model.cells)):
            activation, membranes[index] = cell(layer(activation), membranes[index])
            records[index].append(activation)
    return [torch.stack(record) for record in records]


@pytest.mark.parametrize("multiplier", [1.0, 1.5])
def test_layer_major_forward_matches_time_major_reference(multiplier):
    torch.manual_seed(0)
    model = SNN(hidden_sizes=(24, 16), input_size=40)
    model.set_threshold_multiplier(multiplier)
    spikes = (torch.rand(12, 20, 40) < 0.4).float()
    _, _, records = model(spikes)
    expected = reference_time_major_forward(model, spikes)
    for got, want in zip(records[1:], expected):
        assert (got != want).float().mean().item() < 1e-3


def test_layer_major_gradients_match_time_major_reference():
    torch.manual_seed(1)
    model = SNN(hidden_sizes=(24, 16), input_size=40)
    spikes = (torch.rand(10, 16, 40) < 0.4).float()
    labels = torch.randint(0, 10, (16,))
    logits, penalty, _ = model(spikes)
    (nn.functional.cross_entropy(logits, labels) + 0.1 * penalty).backward()
    new = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    outputs = reference_time_major_forward(model, spikes)
    ref_logits = model.logit_scale * outputs[-1].mean(dim=0)
    ref_penalty = (outputs[0].mean() + outputs[1].mean()) / 2
    (nn.functional.cross_entropy(ref_logits, labels) + 0.1 * ref_penalty).backward()
    for a, b in zip(new, [p.grad for p in model.parameters()]):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_shards_are_disjoint_and_complete():
    items = list(range(23))
    parts = [shard_items(items, (i, 4)) for i in range(4)]
    assert sorted(sum(parts, [])) == items
    assert len({x for part in parts for x in part}) == len(items)
    assert shard_items(items, None) == items
    assert parse_shard("1/3") == (1, 3)
    with pytest.raises(ValueError):
        parse_shard("3/3")
