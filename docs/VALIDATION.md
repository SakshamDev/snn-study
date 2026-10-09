# Repository validation record

Verified October 9, 2026, on this Windows host using Python 3.12.4 and CPU execution. These checks establish the delivered software's tested scope; they do not establish the research hypothesis.

## Environment and packaging

- Installed PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, snnTorch 1.0.0, NumPy 2.1.3, pandas 2.2.3, SciPy 1.14.1, Matplotlib 3.9.2, and pytest 8.3.4 in an isolated workspace environment.
- `pip check`: no broken requirements.
- Built and installed the project in editable mode successfully, using the declared setuptools build configuration.
- Recorded the complete dependency versions in `requirements-lock-windows-cpu.txt`. A fresh installation from that lock on another computer has not been tested.
- Standard temporary-folder creation in this restricted desktop host encountered Windows ACL errors during environment creation, pytest fixtures, and editable installation. External workspace-only helpers made scratch directories inherit workspace permissions. The research code and test assertions were unchanged by those helpers. The PowerShell launcher uses the test helper when it uses the delivered workspace environment.

## Automated checks

**19 tests passed.** Coverage includes:

- Hand-calculated LIF recurrence, strict threshold equality, and one-spike-per-step overshoot behavior.
- Nonzero surrogate gradient for the hidden-spike penalty and membrane reset between forward calls.
- Threshold intervention preserves weights and the output threshold.
- Hand-counted effective SNN/ANN operations and model footprint.
- Bernoulli encoding extremes, shared timestep prefixes, and invariance to evaluation batch partitioning.
- Balanced, disjoint, reproducible train/validation splits and macro-F1 arithmetic.
- Encoding-repeat aggregation inside training seeds, paired seed alignment, reduction signs, single-seed missing confidence intervals, and constrained selection.
- Rejection of changed study settings and final testing without a frozen protocol and locked selection.

The delivered launcher command `./run.ps1 -Action check` also completed with 19 passing tests.

## Complete synthetic workflow

Ran `./run.ps1 -Action smoke` successfully: three trained models, five validation conditions, validation-only strategy selection, five test conditions, tables, and four figures in PNG and PDF. Model reload, protocol gating, metadata, and prediction export were exercised.

The generated patterns are deliberately tiny software fixtures. Their accuracy is not a benchmark result. No confidence interval is emitted for this single training seed. The initial plotting run reported a sandbox font-cache warning after writing its figures; subsequent launcher calls use a writable workspace cache.

Artifacts: `runs/smoke/`. The saved frozen protocol and selection remain available for inspection.

## Real-data MNIST pilot

Ran `./run.ps1 -Action pilot` successfully with the exact saved `configs/pilot.json`: 5,000 training examples, 1,000 validation examples, three epochs, training seed 0, T=16, and three validation encoding realizations. Trained the baseline SNN, lambda=1 SNN, and ANN, then evaluated all five validation conditions.

| Condition | Validation accuracy | Mean effective SNN accumulations per image |
|---|---:|---:|
| Baseline SNN, threshold multiplier 1 | 90.00% | 264,285.39 |
| Baseline SNN, multiplier 1.25 | 90.60% | 254,384.82 |
| Baseline SNN, multiplier 1.5 | 88.40% | 246,620.99 |
| Regularized SNN, lambda=1, multiplier 1 | 91.00% | 252,987.89 |
| ANN reference | 88.20% | Different operation type; see raw MAC metrics |

The regularized-versus-baseline pilot difference is +1.00 percentage point in accuracy, 4.27% fewer total effective accumulations, and 20.71% fewer accumulations driven by hidden outputs. These are descriptive values from one training seed. They have no seed-level confidence interval and must not be presented as confirmation of the main Fashion-MNIST hypothesis. The pilot uses lambda=1; the proposed main primary contrast uses lambda=0.1.

All four pilot PNG figures were visually inspected: accuracy versus activity, timestep/penalty heatmap, per-layer operation counts, and learning curves. Their PDF counterparts were generated. Raw metrics and checkpoints are preserved under `runs/pilot/`.

MNIST's official test partition was downloaded and hashed for provenance but was not used for model evaluation. No pilot test results or test-driven tuning were produced.

## Boundaries of this verification

The 50-run Fashion-MNIST study has not been started. CUDA, other operating systems, larger architectures, corruption tests, other neuron models, hardware energy, and paper submission have not been tested or performed. Main-study convergence and ANN tuning remain explicit development milestones in the implementation plan.

The research source hash used in the completed smoke and pilot runs is `679c0eb7fd8f551a60b208496be64f84743200c90f0925d9e52d11fb269a5a84`. Runs preceded the initial Git commit, so their `git_commit` values are null; the source hash supplies their implementation provenance. Source, configs, plans, and this report are committed; generated datasets and run artifacts remain local and Git-ignored.

## GPU-oriented optimisation pass (October 9, 2026)

- Added a layer-major SNN forward, device-resident loss accumulation, index-based batching with a DataLoader-identical shuffle, a bit-identical threaded spike encoder, and grid sharding. See docs/KAGGLE.md.
- 28 tests pass (9 new in tests/test_fast_path.py: shuffle/batch equivalence with DataLoader, encoder stream identity, forward and gradient equivalence with the original time-major loop, prefetch behaviour, sharding).
- Synthetic end-to-end workflow and a 2-worker sharded train stage completed on CPU. A 3-epoch MNIST run tracked the previous implementation closely (epoch-1 loss 1.4611 vs 1.4604) but is not bit-identical.
- CUDA execution, deterministic-mode behaviour on a T4, and GPU speedups have NOT been tested; run the preflight in docs/KAGGLE.md first. Runs under runs/pilot and runs/smoke predate this change and will be refused by the code-revision guard.

