import torch
from torch import nn
from snntorch import surrogate

class LIFCell(nn.Module):
    """Immediate subtractive reset; strict > threshold; reset gradient detached.

    State is explicitly passed, so no state survives independent forward calls.
    snnTorch supplies the surrogate derivative, not an implicit stateful cell.
    """
    def __init__(self, beta=0.9, threshold=1.0, alpha=2.0):
        super().__init__()
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.spike_function = surrogate.atan(alpha=alpha)

    def forward(self, current, membrane):
        voltage = self.beta * membrane + current
        spike = self.spike_function(voltage - self.threshold)
        membrane = voltage - self.threshold * spike.detach()
        return spike, membrane

class SNN(nn.Module):
    def __init__(self, hidden_sizes=(128, 64), beta=0.9, threshold=1.0,
                 surrogate_alpha=2.0, logit_scale=10.0, input_size=784, classes=10):
        super().__init__()
        widths = [input_size, *hidden_sizes, classes]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.cells = nn.ModuleList(LIFCell(beta, threshold, surrogate_alpha) for _ in self.layers)
        self.logit_scale = logit_scale
        self.base_threshold = threshold
        self.widths = widths

    def set_threshold_multiplier(self, multiplier):
        if multiplier < 1:
            raise ValueError("This protocol only raises hidden thresholds")
        for index, cell in enumerate(self.cells):
            cell.threshold.fill_(self.base_threshold * (multiplier if index < 2 else 1.0))

    def forward(self, spikes):
        # Layers are strictly feed-forward, so layer k at time t depends only on layer k-1
        # at time t and its own membrane. Evaluating layer-major is therefore equivalent to
        # the time-major loop, but the synaptic matmul runs once over all timesteps.
        steps, batch, _ = spikes.shape
        outputs, activation = [], spikes
        for layer, cell, width in zip(self.layers, self.cells, self.widths[1:]):
            currents = layer(activation)
            membrane = spikes.new_zeros((batch, width))
            record = []
            for time in range(steps):
                spike, membrane = cell(currents[time], membrane)
                record.append(spike)
            activation = torch.stack(record)
            outputs.append(activation)
        logits = self.logit_scale * outputs[-1].mean(dim=0)
        penalty = (outputs[0].mean() + outputs[1].mean()) / 2
        return logits, penalty, [spikes, *outputs]

class ANN(nn.Module):
    def __init__(self, hidden_sizes=(128, 64), input_size=784, classes=10):
        super().__init__()
        widths = [input_size, *hidden_sizes, classes]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.widths = widths

    def forward(self, x):
        activations = [x]
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            activations.append(x)
        return self.layers[-1](x), activations

def make_model(cfg, kind):
    if kind == "ann":
        return ANN(cfg.hidden_sizes)
    if kind != "snn":
        raise ValueError("Unknown model kind")
    return SNN(cfg.hidden_sizes, cfg.beta, cfg.threshold, cfg.surrogate_alpha, cfg.logit_scale)

