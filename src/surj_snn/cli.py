import argparse
import json
from pathlib import Path
from .config import ROOT, load_config

def main(command):
    parser = argparse.ArgumentParser(description=f"SNN research study: {command}")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / ("smoke.json" if command == "smoke_test" else "pilot.json"))
    if command == "prepare_data":
        parser.add_argument("--download", action="store_true", help="Download official datasets")
    if command in {"train", "evaluate"}:
        parser.add_argument("--model", choices=["ann", "snn"], default="snn")
        parser.add_argument("--steps", type=int, default=16)
        parser.add_argument("--penalty", type=float, default=0)
        parser.add_argument("--seed", type=int, default=0)
    if command in {"evaluate", "analyze"}:
        parser.add_argument("--split", choices=["validation", "test"], default="validation")
    if command == "evaluate":
        parser.add_argument("--threshold-multiplier", type=float, default=1)
    if command == "run_grid":
        parser.add_argument("--stage", choices=["train", "validate", "test"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    from . import experiment
    if command == "prepare_data":
        _, manifest = experiment.load_data(cfg, download=args.download)
        print(json.dumps(manifest, indent=2))
    elif command == "freeze_protocol":
        print(experiment.freeze_protocol(cfg))
    elif command == "train":
        steps, penalty = (1, 0.0) if args.model == "ann" else (args.steps, args.penalty)
        print(experiment.train_condition(cfg, args.model, steps, penalty, args.seed))
    elif command == "evaluate":
        steps, penalty = (1, 0.0) if args.model == "ann" else (args.steps, args.penalty)
        print(experiment.evaluate_condition(cfg, args.model, steps, penalty, args.seed, args.split, args.threshold_multiplier))
    elif command == "run_grid":
        experiment.run_grid(cfg, args.stage)
    elif command in {"analyze", "select"}:
        from .analysis import analyze, select_validation
        print(analyze(cfg, args.split) if command == "analyze" else select_validation(cfg))
    elif command == "smoke_test":
        if cfg.dataset != "synthetic":
            parser.error("smoke_test requires the synthetic dataset")
        from .analysis import analyze, select_validation
        experiment.freeze_protocol(cfg)
        experiment.run_grid(cfg, "train")
        experiment.run_grid(cfg, "validate")
        select_validation(cfg)
        experiment.run_grid(cfg, "test")
        print(analyze(cfg, "test"))
        print("Synthetic software verification complete; these are not research results.")
    else:
        parser.error("Unknown command")
