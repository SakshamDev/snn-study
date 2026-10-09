# Accuracy–Activity Trade-offs in Spiking Neural Networks: Regularization vs. Threshold Adaptation

[![Tests](https://github.com/SakshamDev/snn-study/actions/workflows/ci.yml/badge.svg)](https://github.com/SakshamDev/snn-study/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible research codebase investigating the trade-offs between classification accuracy and internal spiking activity in Spiking Neural Networks (SNNs).

This project compares **training-time spike regularization** against **post-training threshold scaling** across varying temporal horizons ($T \in \{8, 16, 32\}$), benchmarked against an equivalent Artificial Neural Network (ANN) reference baseline with explicit synaptic operation (SynOps) accounting.

---

## 🔬 Research Overview

### Key Questions
1. **Regularization vs. Threshold Scaling:** How effectively does an internal hidden-spike occupancy penalty ($\mathcal{L}_{\text{occupancy}}$) during surrogate-gradient backpropagation reduce spiking activity compared to post-training threshold scaling ($\theta \leftarrow \theta \cdot \alpha$)?
2. **Constrained Efficiency:** Can either strategy achieve meaningful reductions in total Synaptic Operations ($\text{SynOps}$) while respecting a pre-declared accuracy budget ($\le 1.0\text{ pp}$ drop relative to the unconstrained baseline)?
3. **Temporal Dynamics:** How do simulation temporal horizons ($T=8, 16, 32$ timesteps) and layer-wise fanouts govern the efficiency Pareto frontier relative to standard ReLU ANNs?

```
                       [ Input: 784 (Bernoulli Rate Encoding) ]
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       LIF Hidden Layer 1 (128 Neurons)           │
             │   v[t] = β·u[t-1] + W₁s₀[t] + b₁; s₁[t] = v > θ  │
             └──────────────────────────────────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       LIF Hidden Layer 2 (64 Neurons)            │
             │   v[t] = β·u[t-1] + W₂s₁[t] + b₂; s₂[t] = v > θ  │
             └──────────────────────────────────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       LIF Output Layer (10 Class Neurons)        │
             │         z = γ · (1/T) ∑ s_out[t],  γ = 10.0      │
             └──────────────────────────────────────────────────┘
                                      │
                        Loss = CrossEntropy(z, y)
                     + λ · 0.5 · (Occupancy₁ + Occupancy₂)
```

---

## 📐 Experimental Methodology & Architecture

- **SNN Dynamics:** 784–128–64–10 Leaky Integrate-and-Fire (LIF) network ($\beta = 0.9, \theta = 1.0$) with immediate subtractive reset ($\mathbf{u}[t] = \mathbf{v}[t] - \theta \cdot \mathbf{s}[t]$), detached reset gradient, and arctangent surrogate gradient ($\alpha = 2.0$).
- **ANN Reference:** Identical 784–128–64–10 topology and parameter count (109,386 parameters) with standard ReLU activations.
- **Activity Accounting:**
  - **SynOps (SNN):** Event-driven accumulation count ($\text{SynOps} = \sum_{l} \sum_{i \in \text{active}_{l-1}} \|\mathbf{w}_{l, :, i}\|_0$).
  - **Effective MACs (ANN):** Non-zero activation multiply-accumulate operations.
  - **Occupancy & Inactive Neurons:** Layer-wise spike rates and silent neuron proportions.
- **Randomness Controls:**
  - Shared Bernoulli rate-encoding streams across temporal windows ($T_{\text{max}} = 32$) to ensure prefix invariance.
  - 5 independent training seeds ($S \in \{11, 22, 33, 44, 55\}$).
  - Disjoint validation ($N=3$) and test ($N=3$) encoding realizations.
- **Selection & Statistical Rigor:**
  - Optimal intervention selected strictly on the validation partition within the tolerance budget ($\delta = 1.0\text{ pp}$).
  - Seed-level paired difference estimation ($\Delta \text{Acc}$, $\% \Delta \text{SynOps}$) with two-sided 95% Student's $t$ confidence intervals across independent seeds.

---

## 📁 Repository Structure

```
├── configs/
│   ├── main.json             # 50-run main study configuration (Fashion-MNIST, CPU)
│   ├── main_cuda.json        # Main study configuration optimized for CUDA / Kaggle T4
│   ├── pilot.json            # 3-epoch MNIST feasibility pilot
│   ├── smoke.json            # Fast synthetic workflow check (CPU)
│   └── smoke_cuda.json       # Fast synthetic workflow check (CUDA preflight)
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md # Mathematical formulas, network dynamics & statistical protocol
│   ├── IMPLEMENTATION_PLAN.md # Execution milestones, pipeline design & command reference
│   ├── KAGGLE.md             # Guide for running the main study on Kaggle T4 GPUs
│   └── VALIDATION.md         # Unit test & benchmark verification record (28 tests)
├── scripts/
│   ├── prepare_data.py       # Deterministic dataset downloading & stratification
│   ├── freeze_protocol.py    # Protocol hashing & integrity locking
│   ├── train.py              # Single-condition model training
│   ├── evaluate.py           # Condition evaluation on validation/test splits
│   ├── run_grid.py           # Batch grid execution with sharding support
│   ├── run_parallel.py       # Multi-process parallel worker dispatcher
│   ├── select.py             # Validation-gated strategy selection
│   ├── analyze.py            # Statistical synthesis & publication figure generation
│   └── smoke_test.py         # End-to-end synthetic verification suite
├── src/
│   └── surj_snn/             # Core Python package (models, data, metrics, experiment, analysis)
├── tests/
│   ├── test_science.py       # 19 tests: LIF dynamics, surrogate gradients, metrics, protocol
│   └── test_fast_path.py     # 9 tests: batching, exact shuffle replication, sharding
├── pyproject.toml            # Project packaging & dependencies
└── run.ps1                   # Windows PowerShell task runner
```

---

## 🚀 Quick Start

### 1. Installation

**Requirements:** Python 3.11 or 3.12.

```bash
# Clone the repository
git clone https://github.com/SakshamDev/snn-study.git
cd snn-study

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install CPU PyTorch and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### 2. Verify the Installation (28 Tests)

```bash
# Run automated tests
pytest -q

# Run synthetic end-to-end pipeline check
python scripts/smoke_test.py --config configs/smoke.json
```

On Windows with PowerShell:
```powershell
.\run.ps1 -Action check
.\run.ps1 -Action smoke
```

---

## 🔬 Running the Research Study

### Option A: Cloud GPU (Kaggle T4 / CUDA) — Recommended

Running the full 50-condition grid across 5 seeds is fast on a GPU using parallel workers (~1–2 hours total). See [docs/KAGGLE.md](docs/KAGGLE.md) for full details.

```python
# In a Kaggle Notebook (GPU T4, Internet: On):
CFG = "configs/main_cuda.json"

!python scripts/prepare_data.py --config {CFG} --download
!python scripts/freeze_protocol.py --config {CFG}

# Train 50 conditions with 3 parallel worker processes
!python scripts/run_parallel.py --config {CFG} --stage train --workers 3

# Evaluate validation partition & lock candidate selection
!python scripts/run_parallel.py --config {CFG} --stage validate --workers 3
!python scripts/analyze.py --config {CFG} --split validation
!python scripts/select.py --config {CFG}

# Evaluate final held-out test set & generate publication figures
!python scripts/run_parallel.py --config {CFG} --stage test --workers 3
!python scripts/analyze.py --config {CFG} --split test
```

### Option B: Local Multi-Core CPU

```powershell
# 1. Download Fashion-MNIST and train the 50-condition grid
.\run.ps1 -Action main-train

# 2. Evaluate validation split and produce interim analysis
.\run.ps1 -Action main-validate

# 3. Lock validation choices, evaluate test split, and generate final figures
.\run.ps1 -Action main-test
```

---

## 📊 Outputs & Artifacts

All experiment runs and analysis tables are written to `runs/<study_name>/`:

```text
runs/main/
├── study.json                   # Resolved configuration & data provenance
├── protocol.json                # Cryptographically frozen protocol metadata
├── selection.json               # Validation-locked strategy decisions
├── snn_T16_L0p1_seed11/
│   ├── run.json                 # Execution environment, timestamp & checkpoint SHA256
│   ├── history.json             # Epoch-by-epoch loss, occupancy, accuracy & wall time
│   ├── best.pt                  # Checkpoint with highest validation accuracy
│   └── evaluations/
│       ├── validation_a1/       # Validation metrics (3 encoding realizations)
│       └── test_a1/             # Test metrics & raw predictions
└── analysis/test/
    ├── encoding_metrics.csv     # Raw repeat-level metrics
    ├── seed_metrics.csv         # Aggregated seed-level metrics
    ├── summary.csv              # Means, SDs, and 95% confidence intervals
    ├── primary_paired_effects.csv # Primary hypothesis paired differences
    ├── selected_strategy_effects.csv # Validation-selected strategy comparisons
    └── figures/                 # Publication-ready figures (PNG & PDF):
        ├── fig1_accuracy_activity_tradeoff.pdf
        ├── fig2_penalty_timestep_heatmap.pdf
        ├── fig3_layer_operation_breakdown.pdf
        └── fig4_training_convergence_curves.pdf
```

---

## 📖 Detailed Documentation

- **[Experimental Protocol](docs/EXPERIMENT_PROTOCOL.md):** Formal definitions of SNN/ANN dynamics, SynOps calculations, loss functions, and statistical pairing.
- **[Implementation Plan & Architecture](docs/IMPLEMENTATION_PLAN.md):** Module breakdown, execution milestones, acceptance criteria, and command reference.
- **[Kaggle GPU Execution Guide](docs/KAGGLE.md):** Step-by-step notebook instructions, performance optimizations, and sharding details.
- **[Validation & Verification Record](docs/VALIDATION.md):** Summary of 28 passing unit/science tests and MNIST pilot results.

---

## 📄 License & Attribution

This project is licensed under the [MIT License](LICENSE).
