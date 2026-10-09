"""Run one stage of the grid as several parallel worker processes (one shard each).

Why: this model is tiny, so a single process cannot keep a T4 busy (CPU spike encoding and
kernel launches dominate). Several workers share one GPU, or are spread across GPUs.

Freeze the protocol and prepare data first; then, for example:
    python scripts/run_parallel.py --config configs/main_cuda.json --stage train --workers 3
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=1, help="Visible GPUs to spread workers over (round-robin)")
    args = parser.parse_args()
    processes = []
    for index in range(args.workers):
        env = dict(os.environ)
        if args.gpus > 1:
            env["CUDA_VISIBLE_DEVICES"] = str(index % args.gpus)
        command = [sys.executable, str(ROOT / "scripts" / "run_grid.py"), "--config", str(args.config),
                   "--stage", args.stage, "--shard", f"{index}/{args.workers}"]
        print("launch:", " ".join(command), flush=True)
        processes.append(subprocess.Popen(command, env=env, cwd=ROOT))
    codes = [process.wait() for process in processes]
    failed = [index for index, code in enumerate(codes) if code]
    if failed:
        sys.exit(f"Workers failed: {failed}. Completed runs are preserved; move any incomplete run aside, then rerun.")
    print(f"All {args.workers} workers finished the {args.stage} stage.")


if __name__ == "__main__":
    main()
