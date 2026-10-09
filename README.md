# Accuracy–Activity Trade-offs in Small Spiking Neural Networks

A beginner-friendly, software-only research repository for a possible SURJ paper. It compares a hidden-spike training penalty with post-training threshold adjustment, with an ANN reference and explicit activity accounting.

**Start with the MNIST pilot. The main Fashion-MNIST experiment contains 45 SNN training runs plus five ANN runs.** Training the main grid is an intentional, separate action.

## Reading map

- [Saved research response](docs/research_response.md): the complete prior research-planning answer, preserved from the chat.
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md): milestones, commands, acceptance checks, and remaining research decisions.
- [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md): exact implemented equations, controls, analysis, and limitations.
- [Original research prompt](docs/original_research_prompt.md): your initial objectives.
- [Validation record](docs/VALIDATION.md): checks actually performed during repository creation.

No author identity, research outcomes, publication claim, or remote Git host is assumed. The local repository is ready for your own experiments and writing.

## Quick start on this Windows workspace

Open PowerShell in this directory:

```powershell
.\run.ps1 -Action check
.\run.ps1 -Action smoke
.\run.ps1 -Action pilot
```

The launcher first uses a local `.venv`. In this delivered workspace it can also use the isolated validation environment at `../../work/snn-venv`. That fallback is a local convenience; it is not included when transferring the source repository.

With the workspace fallback, `check` uses the external temporary-folder compatibility runner when available. On a normal local environment, it runs pytest directly. The launcher keeps Matplotlib's cache under `runs/` unless you set `MPLCONFIGDIR` yourself.

The smoke check uses generated patterns solely to verify the complete workflow. The pilot downloads MNIST if necessary, trains two SNN conditions and one ANN for three epochs, evaluates validation data, and creates figures. The pilot launcher never evaluates MNIST's official test partition.

Completed runs and evaluations are skipped on rerun. Interrupted runs are preserved and clearly reported; this version resumes at completed-condition boundaries, not within a partially completed training run. Move an interrupted condition directory aside before retrying the identical protocol.

## Install on another machine

Python 3.11–3.13 is supported by the project metadata. The delivered lock was tested on Windows with Python 3.12 and CPU PyTorch.

```powershell
.\scripts\setup.ps1
```

Alternatively, use the standard commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-cpu.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS, create a virtual environment, install a compatible PyTorch/torchvision pair using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/), then run `python -m pip install -e ".[dev]"`. Capture a new environment lock before research runs. The Windows CPU lock is not a CUDA or universal-platform lock.

All Python scripts also work without an editable install because their bootstrap locates the repository's `src/` directory.

## Main study workflow

Read the implementation plan, inspect the pilot, and finalize `configs/main.json` before starting. A frozen protocol records the resolved configuration, dataset/split hashes, source hash, primary contrast, and accuracy tolerance.

```powershell
# Downloads Fashion-MNIST, freezes the plan, and starts the 50-run grid.
.\run.ps1 -Action main-train

# Evaluates all predeclared validation conditions and produces analysis.
.\run.ps1 -Action main-validate

# Locks validation-only choices, evaluates the test grid, and analyzes it.
.\run.ps1 -Action main-test
```

To run one condition with an activated environment:

```powershell
python scripts/prepare_data.py --config configs/main.json --download
python scripts/freeze_protocol.py --config configs/main.json
python scripts/train.py --config configs/main.json --model snn --steps 16 --penalty 0.1 --seed 11
python scripts/evaluate.py --config configs/main.json --model snn --steps 16 --penalty 0.1 --seed 11 --split validation
```

`scripts/run_grid.py --config ... --stage train|validate|test` provides the batch workflow. `scripts/select.py` locks validation choices; `scripts/analyze.py --split validation|test` generates tables and plots from the complete grid. Every script provides `--help`.

## What is implemented

- A 784–128–64–10 LIF network, using PyTorch with snnTorch's arctangent surrogate derivative.
- An identically sized ReLU ANN reference.
- Explicit immediate subtractive reset, with reset gradients detached and state local to each forward call.
- Bernoulli rate input encoding with shared evaluation realizations and common prefixes across simulation windows.
- Cross-entropy on scaled output rates plus an equally weighted hidden-layer occupancy penalty.
- Independent initialization, data-order, training-encoding, and evaluation randomness.
- Stratified, saved splits and separate official test data.
- Validation checkpoint selection; earliest epoch wins exact ties.
- Post-training hidden-threshold adjustment with fixed weights and output threshold.
- Layer spikes, occupancy, inactive neurons, output ties, effective synaptic accumulations, ANN MACs, footprint, and instrumented software timing.
- Seed-level aggregation, paired effects, and 95% t intervals when at least two independent training seeds exist.
- Validation-only constrained selection and explicit checks before test evaluation.
- PNG/PDF figures and CSV/JSON/NPZ raw outputs.

## Outputs

Generated files are placed in `runs/<study>/` and excluded from Git by default:

```text
runs/main/
  study.json                  resolved configuration and data provenance
  protocol.json               frozen research protocol
  selection.json              validation-only choices and evidence hash
  snn_T16_L0p1_seed11/
    run.json                  environment, status, footprint, checkpoint hash
    history.json              epoch-level training and validation records
    best.pt                   best validation checkpoint
    evaluations/
      validation_a1/
        metrics.json
        predictions_encoding101.npz
  analysis/test/
    encoding_metrics.csv      repeat-level measurements
    seed_metrics.csv          one row per independently trained model/condition
    summary.csv               means, SDs, and confidence intervals
    primary_paired_effects.csv
    selected_strategy_effects.csv
    analysis_metadata.json
    figures/                  four figures, each PNG and PDF
```

Back up `runs/` and `data/splits/` before cleaning anything. They are required research records even though they are not source-controlled. Publishing or uploading data/checkpoints is a separate choice.

## Interpretation

Effective synaptic accumulations assume events can skip inactive connections. Standard dense PyTorch execution does not automatically realize that saving. ANN MACs and SNN accumulations are different operation types. Neither counts nor software execution time establishes neuromorphic energy or latency.

The included one-seed pilot and synthetic check cannot establish a research hypothesis. Five main-study seeds are a starting point, and uncertainty may still be wide. Existing literature already studies sparsity penalties, threshold effects, and accuracy–activity trade-offs; the planned contribution is the controlled comparison and its explanation.

The code intentionally leaves optional noise studies, extra neuron models, conversion, and publication drafting to later milestones. The complete original planning response remains in `docs/research_response.md`.

