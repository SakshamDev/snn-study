# Implementation and execution plan

Created October 9, 2026. This document translates the saved research response into the implemented workflow. It does not report the main experiment's outcomes.

## 1. Read the implementation before expanding it

| File | Responsibility | What to understand |
|---|---|---|
| `src/surj_snn/config.py` | Validated configuration and grid | All scientific choices come from one resolved config. |
| `data.py` | Images, saved splits, encoding | Train/validation disjointness and shared evaluation randomness. |
| `models.py` | LIF dynamics, SNN, ANN | Explicit state, strict threshold, detached reset gradient. |
| `metrics.py` | Classification, SynOps/MACs, footprint | Counts are hardware-independent proxies. |
| `experiment.py` | Training, checkpointing, evaluation, protocol checks | Validation is used for selection; test evaluation requires locked choices. |
| `analysis.py` | Seed aggregation, comparisons, selection, plots | Encoding repeats are nested within training seeds. |
| `scripts/` | Command-line entry points | Small scripts use the common implementation. |
| `tests/` | Hand-checkable scientific invariants | Validate the equations and counting conventions. |

## 2. Setup and software verification

1. Run `./run.ps1 -Action check` in PowerShell, or `python -m pytest -q` in the configured environment.
2. Run `./run.ps1 -Action smoke` to exercise training, validation selection, checkpoint loading, test gating, analysis, and figure creation on generated data.
3. Confirm there are three completed smoke training runs and five evaluated conditions per split: two trained SNNs, two additional raised-threshold conditions, and one ANN.
4. Inspect the smoke analysis metadata. It must state that data are synthetic and provide no training-seed confidence interval for the one-seed configuration.

Acceptance checks: no persistent membrane state between examples; nonzero regularizer gradient; correct fanout counting; matched evaluation input spikes; clear missing-run failures; no test selection from test metrics.

The delivered source supports normal virtual environments. On this Codex Windows host, Python's temporary-directory permissions interrupted standard `ensurepip`; creation-time verification used an isolated environment under `work/` with inherited permissions for temporary installation folders. This workaround is external to the research implementation. If recreating the environment inside a similarly restricted host fails, use a normal user PowerShell session or ask for environment-specific troubleshooting. Do not disable package hash verification.

## 3. Run the minimum real-data pilot

```powershell
.\run.ps1 -Action pilot
```

Configuration: MNIST, 5,000 train / 1,000 validation, seed 0, T=16, penalties 0 and 1, three epochs; thresholds 1, 1.25, 1.5 on baseline hidden layers; three validation encoding realizations; ANN reference.

Read `runs/pilot/analysis/validation/seed_metrics.csv`, `primary_paired_effects.csv`, and all four figures. Compare baseline and regularized input counts: they should match exactly for each encoding realization. Examine hidden activity and whole-network cost separately.

The pilot checks that the pipeline functions. Three epochs on a subset are insufficient to establish convergence or choose a final research conclusion. Do not describe an absent effect here as a general failure of regularization.

## 4. Finalize the scientific decisions using training/validation data

Before the main study, complete these decisions in a research log:

- Is 15 epochs sufficient for reasonable baseline convergence? Examine validation curves using a development run with a distinct results directory.
- Are 0.1 and 1.0 useful penalty magnitudes? A revised grid must retain zero and name a nonzero primary penalty.
- Is the ANN baseline reasonably tuned? The provided ANN shares the starting learning rate and budget; there is no automatic ANN hyperparameter search. If needed, compare a small declared learning-rate/epoch grid on development validation data and finalize the baseline before testing. To implement different final ANN hyperparameters, record them explicitly in a new config/schema and rerun correctness checks.
- Can the laptop afford all three windows? If not, remove one window before freezing, preserve five seeds, and keep the primary T=16 comparison.
- Keep the one-percentage-point accuracy tolerance unless a different practical criterion is justified before the main study.
- Confirm that the test plan, primary contrast, candidate grids, and interval interpretation in `EXPERIMENT_PROTOCOL.md` match your final choices.

Use a new `results_dir` for every changed config or implementation. The code rejects mixed studies. Source hashes cover Python implementation/scripts and `pyproject.toml`; documentation changes alone do not invalidate runs.

## 5. Freeze and train

```powershell
.\run.ps1 -Action main-train
```

This downloads official Fashion-MNIST if needed, saves the deterministic 54,000/6,000 split, records the full dataset and split hashes, freezes the protocol, and trains 45 SNN conditions plus five ANN seeds sequentially.

At every epoch, validation accuracy is averaged over the three fixed validation encoding seeds. The highest value selects the checkpoint; strict improvement preserves the earliest epoch on ties. The same rule applies to the ANN, evaluated once because it has no stochastic encoder.

Each run records package versions, OS/Python, code hash, Git commit if available, actual device, time, full history, and checkpoint hash. Incomplete runs remain marked as failed or running. Preserve them, identify the cause, and move the individual directory aside before retrying. Do not silently discard a difficult seed.

Measure one full representative run, including its validation passes, to budget the grid. The rough `52.5 * R16` estimate in the original response is a planning model; use observed runtime and remember that encoding, validation, and data overhead do not scale perfectly with T.

## 6. Validate and lock model selection

```powershell
.\run.ps1 -Action main-validate
```

Verify that all conditions are present, then inspect activity, accuracy, ties, inactive-neuron fractions, and training curves. Selection uses one common configuration across training seeds at each T, not one independently optimized setting per seed.

The next stage locks two strategies per T:

1. Regularization: choose among all declared penalties at threshold multiplier 1.
2. Threshold adjustment: choose among all declared multipliers on the lambda=0 checkpoint.

Each strategy minimizes mean total SynOps subject to mean validation accuracy at least baseline accuracy minus the tolerance. Ties favor higher accuracy, then smaller penalty/multiplier. The unmodified baseline belongs to both candidate sets. Consequently, a strategy can correctly choose "no intervention."

Candidate-set sizes are three each in the main grid. These are equal candidate counts, not equal training compute: regularization requires retraining and threshold adjustment does not.

## 7. Evaluate the held-out test set once choices are fixed

```powershell
.\run.ps1 -Action main-test
```

This locks selection, evaluates the entire predeclared test grid, and writes final tables and plots. Testing all conditions supports transparent sensitivity reporting. The primary comparison stays T=16, lambda=0.1 versus lambda=0 unless deliberately amended before freezing.

Repeated invocation reuses existing evaluations. The code checks config, source, checkpoint, and selection provenance. It cannot prevent someone from manually opening the dataset or writing a separate test-driven tuning script; experimental discipline is still required.

## 8. Interpret and write

Required outputs:

- Main accuracy/activity trade-offs with all five training seeds.
- Primary paired accuracy difference and operation reduction, including uncertainty.
- Separate hidden-output and whole-network operation reductions.
- Selected regularization versus selected threshold comparison, labeled secondary.
- Training/failure diagnostics and exact model/data assumptions.

Student t intervals assume approximately normal independent seed-level effects. With five seeds, the interval is imprecise and conditional on the fixed split and encoding banks. It does not estimate all forms of population uncertainty. No confidence interval is fabricated for the one-seed pilot.

If the primary accuracy interval crosses the negative tolerance, report that the tolerance is not established. If hidden activity decreases while total cost barely changes, explain input-layer dominance. If threshold adjustment performs similarly, discuss the extra training cost of regularization. Report negative or inconclusive findings without changing the primary question after seeing test results.

## Six-week milestones

| Week | Deliverable |
|---|---|
| 1 | Explain LIF/encoding/surrogate gradients; read core papers; understand and run the tests. |
| 2 | Complete the pilot; audit metrics; choose development settings and freeze the main protocol. |
| 3 | Complete the main training grid with logs/checkpoints. |
| 4 | Finish validation, locked test evaluation, seed-level analysis, and optionally one extension. |
| 5 | Draft methods/results/discussion; produce final figures and verify claims against raw data. |
| 6 | Rerun a principal comparison in a clean environment; finish writing, references, and SURJ formatting. |

Optional extensions are deliberately not part of the baseline implementation: MNIST transfer, input-corruption evaluation, or leak sensitivity. Choose at most one after the core study is complete. A diagram and final manuscript layout also remain authoring tasks; automatic plots are research analysis drafts.

