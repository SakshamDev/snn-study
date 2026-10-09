from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F

from .config import conditions, run_name, token
from .data import encode, load_data, loader
from .metrics import classification_metrics, effective_operations, footprint
from .models import make_model
from .utils import (code_digest, configure_runtime, environment, now, read_json,
                    sha256_file, synchronize, write_json)

def initialize_study(cfg, manifest):
    cfg.output.mkdir(parents=True, exist_ok=True)
    path = cfg.output / "study.json"
    if path.exists():
        old = read_json(path)
        if old["config_sha256"] != cfg.digest or old["data"] != manifest:
            raise RuntimeError("Study config/data changed. Use a new results_dir for a new protocol.")
        if old["code_sha256"] != code_digest():
            raise RuntimeError("Implementation changed. Use a new results_dir; do not mix code revisions.")
    else:
        write_json(path, {"created_utc": now(), "config_sha256": cfg.digest,
                          "code_sha256": code_digest(), "config": cfg.as_dict(), "data": manifest})
    frozen = cfg.output / "protocol.json"
    if frozen.exists():
        check_protocol(cfg, manifest)

def freeze_protocol(cfg):
    _, manifest = load_data(cfg)
    initialize_study(cfg, manifest)
    path = cfg.output / "protocol.json"
    if path.exists():
        check_protocol(cfg, manifest)
        return path
    if any(cfg.output.glob("*/evaluations/test_*/metrics.json")):
        raise RuntimeError("Cannot retroactively freeze after test evaluation")
    write_json(path, {"frozen_utc": now(), "config_sha256": cfg.digest,
                      "code_sha256": code_digest(), "data": manifest,
                      "primary_contrast": {"steps": cfg.primary_timestep, "penalty": cfg.primary_penalty,
                                           "baseline_penalty": 0.0, "threshold_multiplier": 1.0},
                      "accuracy_tolerance_pp": cfg.accuracy_tolerance_pp,
                      "test_plan": "Evaluate the entire predeclared grid; selected-policy contrasts are secondary.",
                      "config": cfg.as_dict()})
    return path

def check_protocol(cfg, manifest):
    path = cfg.output / "protocol.json"
    if not path.exists():
        raise RuntimeError("Freeze the protocol before final testing: scripts/freeze_protocol.py")
    frozen = read_json(path)
    if (frozen["config_sha256"] != cfg.digest or frozen["code_sha256"] != code_digest()
            or frozen["data"] != manifest):
        raise RuntimeError("Frozen protocol does not match current config, code, or data")

def assert_condition(cfg, kind, steps, penalty, seed):
    if (kind, steps, penalty, seed) not in list(conditions(cfg)):
        raise ValueError("Condition is outside the declared configuration grid")

def quick_accuracy(model, kind, dataset, cfg, steps, device):
    model.eval()
    accuracies = []
    seeds = cfg.validation_encoding_seeds if kind == "snn" else [0]
    with torch.inference_mode():
        for encoding_seed in seeds:
            generator = torch.Generator().manual_seed(encoding_seed)
            correct, count = 0, 0
            for x, y, _ in loader(dataset, cfg.eval_batch_size):
                if kind == "snn":
                    input_spikes = encode(x, steps, max(cfg.timesteps), generator).to(device)
                    logits, _, _ = model(input_spikes)
                else:
                    logits, _ = model(x.to(device))
                correct += int((logits.argmax(1).cpu() == y).sum())
                count += len(y)
            accuracies.append(correct / count)
    return float(np.mean(accuracies))

def train_condition(cfg, kind, steps, penalty, seed, data_bundle=None):
    assert_condition(cfg, kind, steps, penalty, seed)
    datasets, manifest = data_bundle or load_data(cfg)
    initialize_study(cfg, manifest)
    destination = cfg.output / run_name(kind, steps, penalty, seed)
    metadata_path = destination / "run.json"
    if metadata_path.exists():
        previous = read_json(metadata_path)
        if previous.get("status") == "complete":
            if sha256_file(destination / "best.pt") != previous["checkpoint_sha256"]:
                raise RuntimeError("Completed checkpoint was modified")
            print(f"SKIP complete: {destination.name}", flush=True)
            return destination
        raise RuntimeError(f"Incomplete run preserved at {destination}. Move it aside before retrying.")
    if (cfg.output / "selection.json").exists():
        raise RuntimeError("Validation selection is locked; use a new study for new training")
    destination.mkdir(parents=True, exist_ok=True)
    device = configure_runtime(cfg, seed)
    metadata = {"status": "running", "started_utc": now(), "kind": kind, "steps": steps,
                "penalty": penalty, "train_seed": seed, "config_sha256": cfg.digest,
                "data": manifest, "device": str(device), "environment": environment(),
                "initialization": "PyTorch nn.Linear default; identical initial weights for matching seeds",
                "shuffle_seed": seed + 20000, "training_encoding_seed": seed + 10000}
    write_json(metadata_path, metadata)
    try:
        model = make_model(cfg, kind).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        training_loader = loader(datasets["train"], cfg.batch_size, shuffle=True, seed=seed + 20000)
        generator = torch.Generator().manual_seed(seed + 10000)
        best_accuracy = -1.0
        history = []
        total_start = time.perf_counter()
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            sums = {"loss": 0.0, "cross_entropy": 0.0, "hidden_occupancy": 0.0}
            count, correct = 0, 0
            for x, y, _ in training_loader:
                optimizer.zero_grad(set_to_none=True)
                y_device = y.to(device)
                if kind == "snn":
                    input_spikes = encode(x, steps, max(cfg.timesteps), generator).to(device)
                    logits, regularizer, _ = model(input_spikes)
                else:
                    logits, _ = model(x.to(device))
                    regularizer = logits.new_zeros(())
                cross_entropy = F.cross_entropy(logits, y_device)
                loss = cross_entropy + penalty * regularizer
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; failed condition retained")
                loss.backward()
                optimizer.step()
                size = len(y)
                count += size
                correct += int((logits.detach().argmax(1).cpu() == y).sum())
                sums["loss"] += float(loss.detach()) * size
                sums["cross_entropy"] += float(cross_entropy.detach()) * size
                sums["hidden_occupancy"] += float(regularizer.detach()) * size
            validation_accuracy = quick_accuracy(model, kind, datasets["validation"], cfg, steps, device)
            row = {"epoch": epoch, **{key: value / count for key, value in sums.items()},
                   "train_accuracy": correct / count, "validation_accuracy": validation_accuracy,
                   "epoch_wall_seconds_including_validation": time.perf_counter() - epoch_start}
            history.append(row)
            write_json(destination / "history.json", history)
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                temporary = destination / "best.pt.tmp"
                torch.save({"state_dict": state, "kind": kind, "steps": steps, "penalty": penalty,
                            "train_seed": seed, "epoch": epoch, "config_sha256": cfg.digest}, temporary)
                temporary.replace(destination / "best.pt")
                metadata["best_epoch"] = epoch
            print(f"{destination.name} epoch {epoch}/{cfg.epochs}: loss={row['loss']:.4f}, val_acc={validation_accuracy:.4f}", flush=True)
        metadata.update(status="complete", completed_utc=now(), best_validation_accuracy=best_accuracy,
                        training_wall_seconds_including_validation=time.perf_counter() - total_start,
                        checkpoint_sha256=sha256_file(destination / "best.pt"), footprint=footprint(model, kind))
        write_json(metadata_path, metadata)
    except Exception:
        metadata.update(status="failed", failed_utc=now(), traceback=traceback.format_exc())
        write_json(metadata_path, metadata)
        raise
    return destination

def evaluation_paths(cfg, split):
    for kind, steps, penalty, seed in conditions(cfg):
        multipliers = cfg.threshold_multipliers if kind == "snn" and penalty == 0 else [1.0]
        for multiplier in multipliers:
            path = cfg.output / run_name(kind, steps, penalty, seed) / "evaluations" / f"{split}_a{token(multiplier)}" / "metrics.json"
            yield (kind, steps, penalty, seed, multiplier), path

def validation_digest(cfg):
    import hashlib
    digest = hashlib.sha256()
    for _, path in evaluation_paths(cfg, "validation"):
        if not path.exists():
            raise RuntimeError(f"Missing validation evaluation: {path}")
        digest.update(path.relative_to(cfg.output).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def check_test_ready(cfg, manifest):
    check_protocol(cfg, manifest)
    selection_path = cfg.output / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("Lock validation-only selection first: scripts/select.py")
    selection = read_json(selection_path)
    if (selection["config_sha256"] != cfg.digest or selection["code_sha256"] != code_digest()
            or selection["validation_sha256"] != validation_digest(cfg)):
        raise RuntimeError("Selection is stale; do not test with changed validation evidence")

def evaluate_condition(cfg, kind, steps, penalty, seed, split="validation", multiplier=1.0, data_bundle=None):
    assert_condition(cfg, kind, steps, penalty, seed)
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    allowed = cfg.threshold_multipliers if kind == "snn" and penalty == 0 else [1.0]
    if multiplier not in allowed:
        raise ValueError("Threshold condition is outside the predeclared grid")
    datasets, manifest = data_bundle or load_data(cfg)
    initialize_study(cfg, manifest)
    if split == "test":
        check_test_ready(cfg, manifest)
    run_dir = cfg.output / run_name(kind, steps, penalty, seed)
    metadata = read_json(run_dir / "run.json")
    if metadata["status"] != "complete":
        raise RuntimeError("Training has not completed")
    checkpoint_digest = sha256_file(run_dir / "best.pt")
    if checkpoint_digest != metadata["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint was modified")
    destination = run_dir / "evaluations" / f"{split}_a{token(multiplier)}"
    result_path = destination / "metrics.json"
    if result_path.exists():
        previous = read_json(result_path)
        if previous["checkpoint_sha256"] != checkpoint_digest:
            raise RuntimeError("Cached evaluation refers to different weights")
        print(f"SKIP evaluated: {run_dir.name} {split} threshold={multiplier}", flush=True)
        return result_path
    destination.mkdir(parents=True, exist_ok=True)
    device = configure_runtime(cfg, seed)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["config_sha256"] != cfg.digest:
        raise RuntimeError("Checkpoint configuration mismatch")
    model = make_model(cfg, kind).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if kind == "snn":
        model.set_threshold_multiplier(multiplier)
    model.eval()
    encoding_seeds = (cfg.validation_encoding_seeds if split == "validation" else cfg.test_encoding_seeds) if kind == "snn" else [None]
    rows = []
    for encoding_seed in encoding_seeds:
        generator = torch.Generator().manual_seed(encoding_seed or 0)
        saved = {"ids": [], "targets": [], "predictions": [], "operations_per_layer": []}
        spikes_saved, ties_saved, zero_saved = [], [], []
        active_neurons = [torch.zeros(width, dtype=torch.bool) for width in model.widths[1:]]
        forward_seconds = 0.0
        with torch.inference_mode():
            for x, y, ids in loader(datasets[split], cfg.eval_batch_size):
                inputs = encode(x, steps, max(cfg.timesteps), generator).to(device) if kind == "snn" else x.to(device)
                synchronize(device)
                start = time.perf_counter()
                output = model(inputs)
                synchronize(device)
                forward_seconds += time.perf_counter() - start
                logits = output[0]
                predictions = logits.argmax(1)
                maximum = logits.max(1, keepdim=True).values
                ties_saved.append(((logits == maximum).sum(1) > 1).cpu().numpy())
                if kind == "snn":
                    records = output[2]
                    counts = torch.stack([spikes.sum(dim=(0, 2)) for spikes in records], dim=1)
                    operations = effective_operations(records[:-1], model.layers, temporal=True)
                    spikes_saved.append(counts.cpu().numpy())
                    zero_saved.append((records[-1].sum(dim=(0, 2)) == 0).cpu().numpy())
                    for index, spikes in enumerate(records[1:]):
                        active_neurons[index] |= spikes.any(dim=0).any(dim=0).cpu()
                else:
                    operations = effective_operations(output[1], model.layers, temporal=False)
                saved["ids"].append(ids.numpy())
                saved["targets"].append(y.numpy())
                saved["predictions"].append(predictions.cpu().numpy())
                saved["operations_per_layer"].append(operations.cpu().numpy())
        arrays = {key: np.concatenate(value) for key, value in saved.items()}
        row = {"kind": kind, "steps": steps, "penalty": penalty, "train_seed": seed,
               "threshold_multiplier": multiplier, "encoding_seed": encoding_seed,
               "split": split, "examples": len(arrays["targets"]),
               **classification_metrics(arrays["targets"], arrays["predictions"]),
               "output_tie_fraction": float(np.concatenate(ties_saved).mean()),
               "software_forward_ms_per_image": 1000 * forward_seconds / len(arrays["targets"])}
        per_layer = arrays["operations_per_layer"].mean(axis=0)
        if kind == "snn":
            arrays["spikes_per_layer"] = np.concatenate(spikes_saved)
            means = arrays["spikes_per_layer"].mean(axis=0)
            for layer, (count, width) in enumerate(zip(means, model.widths)):
                row[f"spikes_layer{layer}"] = float(count)
                row[f"occupancy_layer{layer}"] = float(count / (steps * width))
            for layer, cost in enumerate(per_layer):
                row[f"synops_layer{layer + 1}"] = float(cost)
            row["synops_total"] = float(per_layer.sum())
            row["synops_hidden_outputs"] = float(per_layer[1:].sum())
            row["spikes_total_including_input"] = float(means.sum())
            row["neuron_updates"] = steps * sum(model.widths[1:])
            row["dense_simulator_macs"] = steps * sum(layer.weight.numel() for layer in model.layers)
            row["all_zero_output_fraction"] = float(np.concatenate(zero_saved).mean())
            for layer, active in enumerate(active_neurons, start=1):
                row[f"inactive_layer{layer}_fraction"] = float((~active).float().mean())
        else:
            row["effective_macs"] = float(per_layer.sum())
            row["dense_macs"] = sum(layer.weight.numel() for layer in model.layers)
        np.savez_compressed(destination / f"predictions_encoding{encoding_seed}.npz", **arrays)
        rows.append(row)
    write_json(result_path, {"created_utc": now(), "config_sha256": cfg.digest,
                            "checkpoint_sha256": checkpoint_digest, "environment": environment(),
                            "data": manifest, "footprint": footprint(model, kind), "rows": rows,
                            "timing_scope": "Instrumented model forward; excludes encoding, loading, and metrics; not neuromorphic latency."})
    print(f"EVALUATED {run_dir.name} {split} threshold={multiplier}: accuracy={np.mean([r['accuracy'] for r in rows]):.4f}", flush=True)
    return result_path

def run_grid(cfg, stage):
    bundle = load_data(cfg)
    if stage == "train":
        for condition in conditions(cfg):
            train_condition(cfg, *condition, data_bundle=bundle)
    elif stage in {"validate", "test"}:
        split = "validation" if stage == "validate" else "test"
        for condition, _ in evaluation_paths(cfg, split):
            kind, steps, penalty, seed, multiplier = condition
            evaluate_condition(cfg, kind, steps, penalty, seed, split, multiplier, data_bundle=bundle)
    else:
        raise ValueError("Unknown grid stage")
