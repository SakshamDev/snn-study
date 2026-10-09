import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from surj_snn.analysis import choose_candidate, interval, paired_effects, seed_means
from surj_snn.config import Config
from surj_snn.data import encode, stratified_indices
from surj_snn.metrics import classification_metrics, effective_operations, footprint
from surj_snn.models import ANN, LIFCell, SNN

def test_lif_leaks_integrates_and_resets_immediately():
    cell = LIFCell(beta=.5, threshold=1)
    membrane = torch.zeros(1)
    spikes, states = [], []
    for value in [.8, .8, 0., 1.]:
        spike, membrane = cell(torch.tensor([value]), membrane)
        spikes.append(spike.item())
        states.append(membrane.item())
    assert spikes == [0, 1, 0, 1]
    assert states == pytest.approx([.8, .2, .1, .05])

def test_exact_threshold_does_not_fire():
    spike, membrane = LIFCell()(torch.ones(1), torch.zeros(1))
    assert spike.item() == 0
    assert membrane.item() == 1

def test_large_overshoot_emits_at_most_one_spike_per_step():
    cell = LIFCell(beta=1)
    spike, membrane = cell(torch.tensor([3.2]), torch.zeros(1))
    assert spike.item() == 1
    assert membrane.item() == pytest.approx(2.2)

def test_hidden_penalty_has_nonzero_gradient_to_input_weights():
    model = SNN(hidden_sizes=(2, 2), beta=.9, input_size=2, classes=2)
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.fill_(.3)
            layer.bias.zero_()
    _, penalty, _ = model(torch.ones(4, 3, 2))
    penalty.backward()
    assert model.layers[0].weight.grad.abs().sum().item() > 0

def test_independent_forward_calls_reset_all_membranes():
    torch.manual_seed(4)
    model = SNN(hidden_sizes=(3, 2), input_size=2, classes=2)
    x = torch.ones(6, 2, 2)
    first = model(x)
    model(torch.zeros_like(x))
    repeated = model(x)
    for a, b in zip(first[2], repeated[2]):
        assert torch.equal(a, b)

def test_threshold_adjustment_preserves_weights_and_output_threshold():
    model = SNN()
    weights = [layer.weight.clone() for layer in model.layers]
    model.set_threshold_multiplier(1.5)
    assert [cell.threshold.item() for cell in model.cells] == [1.5, 1.5, 1]
    assert all(torch.equal(a, layer.weight) for a, layer in zip(weights, model.layers))

def test_effective_operations_include_input_and_skip_zero_weight_connections():
    layer = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 0., 2.], [3., 4., 0.]]))
    spikes = torch.tensor([[[1., 0., 1.]], [[1., 1., 0.]]])
    # Source counts [2, 1, 1] times fanouts [2, 1, 1] = 6.
    assert effective_operations([spikes], [layer], temporal=True).item() == 6
    assert effective_operations([torch.tensor([[1., 0., 2.]])], [layer], temporal=False).item() == 3

def test_default_footprint_matches_hand_count():
    snn = SNN()
    info = footprint(snn, "snn")
    assert info["trainable_parameters"] == 109386
    assert info["parameter_bytes"] == 4 * 109386
    assert info["membrane_state_bytes_per_image"] == 4 * 202
    assert info["dense_macs_per_step"] == 109184
    assert footprint(ANN(), "ann")["trainable_parameters"] == 109386

def test_encoder_extremes_common_prefix_and_batch_invariance():
    x = torch.tensor([[0., 1., .5], [.2, .8, .6], [.4, .3, .7]])
    g = lambda: torch.Generator().manual_seed(34)
    full = encode(x, 8, 8, g())
    assert torch.equal(full[:3], encode(x, 3, 8, g()))
    stream = g()
    batches = torch.cat([encode(x[:1], 8, 8, stream), encode(x[1:], 8, 8, stream)], dim=1)
    assert torch.equal(full, batches)
    assert full[:, 0, 0].sum() == 0
    assert full[:, 0, 1].sum() == 8

def test_split_balanced_disjoint_and_reproducible():
    labels = np.arange(500) % 10
    train, validation = stratified_indices(labels, 200, 50, 2026)
    repeated = stratified_indices(labels, 200, 50, 2026)
    assert set(train).isdisjoint(validation)
    assert np.array_equal(train, repeated[0])
    assert np.array_equal(np.bincount(labels[train]), np.repeat(20, 10))
    assert np.array_equal(np.bincount(labels[validation]), np.repeat(5, 10))

def test_macro_f1_includes_missing_classes():
    scores = classification_metrics([0, 0, 1, 1], [0, 0, 0, 1], classes=2)
    assert scores["accuracy"] == .75
    assert scores["macro_f1"] == pytest.approx((.8 + 2/3)/2)

def test_encoding_repeats_do_not_inflate_seed_count():
    rows = []
    for seed, values in [(11, [.6, .8, .7]), (22, [.8, 1., .9])]:
        for repeat, value in enumerate(values):
            rows.append(dict(kind="snn", steps=16, penalty=0., threshold_multiplier=1., train_seed=seed,
                             encoding_seed=repeat, examples=100, accuracy=value))
    means = seed_means(pd.DataFrame(rows))
    assert len(means) == 2
    assert means.accuracy.tolist() == pytest.approx([.7, .9])
    assert interval(means.accuracy)["n"] == 2

def test_single_seed_has_no_fabricated_confidence_interval():
    result = interval([.7])
    assert result["ci95_low"] is None
    assert result["sd"] is None

def test_selection_uses_tolerance_and_total_activity():
    rows = [dict(penalty=0, threshold_multiplier=1, accuracy=.9, synops_total=100),
            dict(penalty=.1, threshold_multiplier=1, accuracy=.895, synops_total=80),
            dict(penalty=1, threshold_multiplier=1, accuracy=.88, synops_total=50)]
    assert choose_candidate(rows, .9, 1)["penalty"] == .1

def test_paired_effects_pair_seed_identity_and_sign():
    a = pd.DataFrame(dict(train_seed=[2, 1], accuracy=[.9, .8], synops_total=[60., 80.], synops_hidden_outputs=[30., 40.]))
    b = pd.DataFrame(dict(train_seed=[1, 2], accuracy=[.81, .91], synops_total=[100., 100.], synops_hidden_outputs=[50., 50.]))
    result = paired_effects(a, b)
    assert result[0]["mean"] == pytest.approx(-1)
    assert result[1]["mean"] == pytest.approx(30)
    with pytest.raises(ValueError, match="same unique"):
        paired_effects(a.iloc[:1], b)

def test_bad_config_rejects_overlapping_validation_test_randomness():
    with pytest.raises(ValueError, match="disjoint"):
        Config(validation_encoding_seeds=[1], test_encoding_seeds=[1]).validate()

