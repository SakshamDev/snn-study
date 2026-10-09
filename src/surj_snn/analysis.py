import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .config import conditions, run_name
from .data import load_data
from .experiment import (check_protocol, check_test_ready, evaluation_paths,
                         initialize_study, validation_digest)
from .utils import code_digest, now, read_json, sha256_file, write_json

KEYS = ["kind", "steps", "penalty", "threshold_multiplier"]

def interval(values):
    """Two-sided 95% t interval over independent training-seed measurements."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite, nonempty seed measurements")
    mean = float(values.mean())
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    sd = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}

def read_evaluations(cfg, split):
    rows = []
    for condition, path in evaluation_paths(cfg, split):
        if not path.exists():
            raise RuntimeError(f"Incomplete grid: missing {path}. Finish evaluation before analysis.")
        result = read_json(path)
        if result["config_sha256"] != cfg.digest or result["environment"]["code_sha256"] != code_digest():
            raise RuntimeError("Evaluation config/code provenance mismatch")
        kind, steps, penalty, seed, multiplier = condition
        run_dir = cfg.output / run_name(kind, steps, penalty, seed)
        if sha256_file(run_dir / "best.pt") != result["checkpoint_sha256"]:
            raise RuntimeError("Evaluation checkpoint provenance mismatch")
        expected = cfg.validation_encoding_seeds if split == "validation" else cfg.test_encoding_seeds
        expected = expected if kind == "snn" else [None]
        if sorted(str(r["encoding_seed"]) for r in result["rows"]) != sorted(str(s) for s in expected):
            raise RuntimeError("Missing or duplicate encoding realizations")
        for row in result["rows"]:
            if (row["kind"], row["steps"], row["penalty"], row["train_seed"], row["threshold_multiplier"]) != condition:
                raise RuntimeError("Evaluation identity mismatch")
            rows.append({key: value for key, value in row.items() if key != "confusion_matrix"})
    return pd.DataFrame(rows)

def seed_means(frame):
    keys = KEYS + ["train_seed"]
    excluded = set(keys + ["encoding_seed", "examples"])
    metrics = [column for column in frame.select_dtypes(include="number").columns if column not in excluded]
    # Encoding repeats are nested within a trained model, never independent seeds.
    return frame.groupby(keys, dropna=False)[metrics].mean().reset_index()

def summarize(frame):
    rows = []
    for keys, group in frame.groupby(KEYS):
        identity = dict(zip(KEYS, keys))
        for metric in frame.select_dtypes(include="number").columns:
            if metric in KEYS + ["train_seed"] or group[metric].isna().all():
                continue
            if group[metric].isna().any():
                raise RuntimeError(f"Partial metric missingness: {metric}")
            rows.append({**identity, "metric": metric, **interval(group[metric].to_numpy())})
    return pd.DataFrame(rows)

def choose_candidate(candidates, baseline_accuracy, tolerance_pp):
    feasible = [row for row in candidates if row["accuracy"] >= baseline_accuracy - tolerance_pp / 100]
    if not feasible:
        raise ValueError("No feasible candidate, including baseline")
    return min(feasible, key=lambda row: (row["synops_total"], -row["accuracy"], row["penalty"], row["threshold_multiplier"]))

def select_validation(cfg):
    _, manifest = load_data(cfg)
    initialize_study(cfg, manifest)
    check_protocol(cfg, manifest)
    path = cfg.output / "selection.json"
    if path.exists():
        check_test_ready(cfg, manifest)
        return path
    if any(cfg.output.glob("*/evaluations/test_*/metrics.json")):
        raise RuntimeError("Cannot select after test inspection")
    frame = seed_means(read_evaluations(cfg, "validation"))
    candidates = frame[frame.kind == "snn"].groupby(KEYS)[["accuracy", "synops_total"]].mean().reset_index()
    policies = []
    for steps in cfg.timesteps:
        sub = candidates[candidates.steps == steps]
        baseline = sub[(sub.penalty == 0) & (sub.threshold_multiplier == 1)].iloc[0]
        for strategy, pool in (("regularization", sub[sub.threshold_multiplier == 1]),
                               ("threshold", sub[sub.penalty == 0])):
            candidates_list = pool.to_dict(orient="records")
            selected = choose_candidate(candidates_list, float(baseline.accuracy), cfg.accuracy_tolerance_pp)
            policies.append({"strategy": strategy, "steps": steps,
                             "penalty": float(selected["penalty"]),
                             "threshold_multiplier": float(selected["threshold_multiplier"]),
                             "validation_accuracy": float(selected["accuracy"]),
                             "validation_synops_total": float(selected["synops_total"]),
                             "baseline_validation_accuracy": float(baseline.accuracy),
                             "candidates": candidates_list})
    write_json(path, {"locked_utc": now(), "config_sha256": cfg.digest, "code_sha256": code_digest(),
                      "validation_sha256": validation_digest(cfg), "policies": policies,
                      "selection_rule": "Lowest mean total SynOps within tolerance of mean same-window baseline accuracy; common setting across training seeds.",
                      "scope": "Validation point-estimate eligibility is not a confidence/noninferiority claim."})
    return path

def paired_effects(a, b):
    """A minus B accuracy; positive activity reduction favors A. Pair exact seeds."""
    if set(a.train_seed) != set(b.train_seed) or a.train_seed.duplicated().any() or b.train_seed.duplicated().any():
        raise ValueError("Paired comparisons require the same unique training seeds")
    a = a.set_index("train_seed").sort_index()
    b = b.set_index("train_seed").sort_index()
    output = [{"metric": "accuracy_difference_pp", **interval(100 * (a.accuracy - b.accuracy))}]
    for column in ["synops_total", "synops_hidden_outputs"]:
        if (b[column] <= 0).any():
            output.append({"metric": f"{column}_reduction_percent", "n": len(a), "mean": None,
                           "sd": None, "ci95_low": None, "ci95_high": None})
        else:
            output.append({"metric": f"{column}_reduction_percent", **interval(100 * (1 - a[column] / b[column]))})
    return output

def analyze(cfg, split):
    _, manifest = load_data(cfg)
    initialize_study(cfg, manifest)
    if split == "test":
        check_test_ready(cfg, manifest)
    raw = read_evaluations(cfg, split)
    seeds = seed_means(raw)
    summary = summarize(seeds)
    destination = cfg.output / "analysis" / split
    destination.mkdir(parents=True, exist_ok=True)
    raw.to_csv(destination / "encoding_metrics.csv", index=False)
    seeds.to_csv(destination / "seed_metrics.csv", index=False)
    summary.to_csv(destination / "summary.csv", index=False)
    base = seeds[(seeds.kind == "snn") & (seeds.steps == cfg.primary_timestep) & (seeds.penalty == 0) & (seeds.threshold_multiplier == 1)]
    treatment = seeds[(seeds.kind == "snn") & (seeds.steps == cfg.primary_timestep) & (seeds.penalty == cfg.primary_penalty) & (seeds.threshold_multiplier == 1)]
    primary = paired_effects(treatment, base)
    pd.DataFrame(primary).to_csv(destination / "primary_paired_effects.csv", index=False)
    accuracy_effect = primary[0]
    low = accuracy_effect["ci95_low"]
    conclusion = ("No interval: fewer than two independent training seeds."
                  if low is None else
                  "Accuracy interval lies above the negative tolerance; assess activity and limitations separately."
                  if low > -cfg.accuracy_tolerance_pp else
                  "Accuracy interval does not establish the declared tolerance.")
    if (cfg.output / "selection.json").exists():
        check_test_ready(cfg, manifest)
        selected = read_json(cfg.output / "selection.json")["policies"]
        effects = []
        for steps in cfg.timesteps:
            choices = {row["strategy"]: row for row in selected if row["steps"] == steps}
            groups = {}
            for strategy, choice in choices.items():
                groups[strategy] = seeds[(seeds.kind == "snn") & (seeds.steps == steps)
                    & (seeds.penalty == choice["penalty"]) & (seeds.threshold_multiplier == choice["threshold_multiplier"])]
            effects.extend({"steps": steps, "comparison": "selected_regularization_minus_selected_threshold", **row}
                           for row in paired_effects(groups["regularization"], groups["threshold"]))
        pd.DataFrame(effects).to_csv(destination / "selected_strategy_effects.csv", index=False)
    write_json(destination / "analysis_metadata.json", {"created_utc": now(), "config_sha256": cfg.digest,
               "split": split, "synthetic": cfg.dataset == "synthetic", "independent_training_seeds": len(cfg.seeds),
               "primary_accuracy_interpretation": conclusion,
               "interval": "Two-sided 95% Student t interval over training-seed means; conditional on split and encoding bank.",
               "selection_effects": "Secondary/descriptive; intervals condition on validation-selected settings.",
               "cost_scope": "SynOps are idealized effective accumulations, never measured energy or device latency."})
    make_figures(cfg, seeds, destination)
    return destination

def make_figures(cfg, seeds, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 180})
    figure_dir = destination / "figures"
    figure_dir.mkdir(exist_ok=True)
    snn = seeds[seeds.kind == "snn"]
    title = f"{cfg.dataset}: {cfg.name}" + (" (software check only)" if cfg.dataset == "synthetic" else "")
    def save(fig, name):
        for extension in ["png", "pdf"]:
            fig.savefig(figure_dir / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("tab10")
    for index, steps in enumerate(cfg.timesteps):
        for method, marker, mask in [("trained penalty", "o", snn.threshold_multiplier == 1),
                                     ("raised threshold", "^", (snn.penalty == 0) & (snn.threshold_multiplier > 1))]:
            sub = snn[(snn.steps == steps) & mask]
            label_used = False
            for _, group in sub.groupby(["penalty", "threshold_multiplier"]):
                x = interval(group.synops_total)
                y = interval(100 * group.accuracy)
                ax.scatter(group.synops_total, 100 * group.accuracy, color=colors(index), marker=marker, alpha=.25, s=18)
                xerr = None if x["ci95_low"] is None else max(0, x["mean"] - x["ci95_low"])
                yerr = None if y["ci95_low"] is None else max(0, y["mean"] - y["ci95_low"])
                ax.errorbar(x["mean"], y["mean"], xerr=xerr, yerr=yerr, fmt=marker, color=colors(index),
                            capsize=3, label=None if label_used else f"T={steps}, {method}")
                label_used = True
    ax.set(xlabel="Effective synaptic accumulations / image", ylabel="Accuracy (%)", title=title)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save(fig, "accuracy_vs_activity")

    means = snn[snn.threshold_multiplier == 1].groupby(["penalty", "steps"])[["accuracy", "synops_total"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, label, factor in zip(axes, ["accuracy", "synops_total"], ["Accuracy (%)", "Effective accumulations / image"], [100, 1]):
        grid = means[metric].unstack("steps").reindex(index=cfg.penalties, columns=cfg.timesteps).to_numpy() * factor
        handle = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(cfg.timesteps)), cfg.timesteps)
        ax.set_yticks(range(len(cfg.penalties)), cfg.penalties)
        ax.set(xlabel="Simulation steps", ylabel="Spike penalty", title=label)
        fig.colorbar(handle, ax=ax, shrink=.8)
    fig.suptitle(title)
    save(fig, "window_penalty_grid")

    sub = snn[snn.steps == cfg.primary_timestep].groupby(["penalty", "threshold_multiplier"])[["synops_layer1", "synops_layer2", "synops_layer3"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(sub))
    labels = [f"lambda={p:g}\na={a:g}" for p, a in sub.index]
    for column, label in zip(sub.columns, ["Input -> hidden 1", "Hidden 1 -> hidden 2", "Hidden 2 -> output"]):
        values = sub[column].to_numpy()
        ax.bar(labels, values, bottom=bottom, label=label)
        bottom += values
    ax.set(ylabel="Effective accumulations / image", title=f"{title}; T={cfg.primary_timestep}")
    ax.legend(fontsize=8)
    save(fig, "layer_operations")

    fig, ax = plt.subplots(figsize=(8, 4))
    for kind, steps, penalty, seed in conditions(cfg):
        if seed != cfg.seeds[0] or (kind == "snn" and steps != cfg.primary_timestep):
            continue
        history = read_json(cfg.output / run_name(kind, steps, penalty, seed) / "history.json")
        ax.plot([row["epoch"] for row in history], [100 * row["validation_accuracy"] for row in history],
                marker=".", label="ANN" if kind == "ann" else f"SNN lambda={penalty:g}")
    ax.set(xlabel="Epoch", ylabel="Validation accuracy (%)", title=f"{title}; training seed {cfg.seeds[0]}")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "learning_curves_first_seed")
