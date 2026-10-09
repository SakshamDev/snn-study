# Software & Scientific Validation Record

This document records the automated verification checks, numerical invariants, baseline pilot benchmarks, and performance testing completed for the codebase.

---

## 1. Automated Test Suite (28 Tests Passing)

The test suite in `tests/` verifies core mathematical and computational invariants:

### Biological & LIF Neuron Invariants (`tests/test_science.py`)
- **Discrete Recurrence & Subtractive Reset:** Hand-calculated membrane potential decay, exact threshold firing, and subtractive post-spike reset ($\mathbf{u}[t] = \mathbf{v}[t] - \theta \cdot \mathbf{s}[t]$).
- **Strict Threshold Inequality:** Potential exactly equal to threshold ($\mathbf{v}[t] = \theta$) does not trigger a spike.
- **Overshoot Spiking Bounds:** Arbitrarily large suprathreshold inputs emit strictly at most one discrete spike per timestep.
- **Surrogate Gradient Flow:** Non-zero gradient backpropagation through the arctangent surrogate derivative to preceding synaptic weights.
- **Stateless Forward Calls:** Complete membrane potential reset across independent forward passes.
- **Post-Training Threshold Scaling:** Parameter conservation and selective hidden-layer threshold adjustment without modifying output threshold or synaptic weights.

### Metrics & Protocol Guards
- **SynOps & MAC Accounting:** Exact match against hand-counted synaptic event fanouts and sparsity-skipping multiplication counts.
- **Bernoulli Rate Encoding:** Shared prefix conservation across temporal horizons and batch-size partitioning invariance.
- **Stratified Partitioning:** Class balance, reproducible split generation, and mutual disjointness of training/validation splits.
- **Statistical Estimators:** Nested repeat-level aggregation, seed-level pairing, missing variance handling for $N=1$, and constrained selection within tolerance.
- **Protocol Integrity:** Guard against schema drift, frozen protocol violations, and unvalidated test-set access.

### Performance & Pipeline Equivalence (`tests/test_fast_path.py`)
- **Fast Batch Gathering:** Exact batch tensor values and dtypes matching standard `DataLoader`.
- **Deterministic Shuffling:** Multi-epoch exact reproduction of `DataLoader` random sampling sequences.
- **Prefetched Spike Encoding:** Multi-threaded Bernoulli generator stream exact match against serial execution.
- **Layer-Major Forward Equivalence:** Mathematical and gradient equivalence between layer-major batched operations and time-major recurrent loops.
- **Grid Sharding:** Disjoint partition completeness across arbitrary worker splits.

---

## 2. Synthetic Smoke Verification

Running `scripts/smoke_test.py --config configs/smoke.json` validates the complete pipeline end-to-end:
- Generates synthetic spatial patterns.
- Freezes protocol metadata.
- Trains SNN ($\lambda \in \{0.0, 1.0\}$) and ANN models.
- Performs validation-gated candidate selection.
- Evaluates the held-out test split.
- Generates publication tables and multi-panel figures (`runs/smoke/analysis/test/figures/`).

---

## 3. Real-Data MNIST Pilot Benchmark

A 3-epoch pilot run (`configs/pilot.json`, 5,000 train / 1,000 val, $T=16$, seed 0) establishes baseline feasibility:

| Condition | Val Accuracy | Mean Effective SynOps / Image |
|---|---:|---:|
| Baseline SNN ($\lambda=0, \alpha=1.0$) | 90.00% | 264,285.39 |
| Baseline SNN + Threshold Scaling ($\alpha=1.25$) | 90.60% | 254,384.82 |
| Baseline SNN + Threshold Scaling ($\alpha=1.50$) | 88.40% | 246,620.99 |
| Regularized SNN ($\lambda=1.0, \alpha=1.0$) | 91.00% | 252,987.89 |
| ANN Reference (ReLU) | 88.20% | *Effective MACs (see raw metrics)* |

*Descriptive pilot metrics for pipeline verification. The full research study evaluates Fashion-MNIST across 5 seeds and 3 temporal horizons.*
