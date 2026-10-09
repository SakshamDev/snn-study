import numpy as np
import torch

def classification_metrics(targets, predictions, classes=10):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(targets), np.asarray(predictions)), 1)
    tp = np.diag(matrix)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(classes, dtype=float), where=denominator != 0)
    return {"accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "confusion_matrix": matrix.tolist()}

def effective_operations(activations, layers, temporal):
    """Count active, nonzero-weight synapses. AC and MAC counts stay distinct."""
    operations = []
    for activation, layer in zip(activations, layers):
        active = activation.detach().sum(dim=0) if temporal else (activation.detach() != 0).to(torch.float32)
        fanout = (layer.weight.detach() != 0).sum(dim=0).to(active.dtype)
        operations.append(active @ fanout)
    return torch.stack(operations, dim=1)

def footprint(model, kind):
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "persistent_buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "membrane_state_bytes_per_image": sum(model.widths[1:]) * 4 if kind == "snn" else 0,
        "dense_macs_per_step": sum(layer.weight.numel() for layer in model.layers),
    }
