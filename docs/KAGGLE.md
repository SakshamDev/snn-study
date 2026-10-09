# Running the main study on Kaggle (T4 GPU)

The CUDA configs run exactly the same experiment as `configs/main.json`; only the device,
results directory and CPU thread count differ. **Run the whole study on one device type**:
CPU and GPU results differ at floating-point level, and the frozen protocol treats a new config
as a new study.

## What was optimised (and what was deliberately not)

Changes that keep the science identical:

- **Layer-major SNN forward.** The network is strictly feed-forward, so each layer's synaptic
  matmul now runs once over all timesteps instead of once per timestep. Same equations.
- **No per-step host synchronisation.** Loss/accuracy accumulate on the device; the finite-loss
  check runs every 50 steps and at each epoch end.
- **Batches gathered by index** instead of ~54,000 per-item Python calls per epoch, with the exact
  same shuffle order as the previous `DataLoader` (verified by a test).
- **CPU spike encoding overlapped with GPU compute** (one ordered producer thread, pinned memory,
  spikes transferred as booleans). The Bernoulli random stream is bit-identical to before.
- **Grid sharding** (`--shard i/n`, `scripts/run_parallel.py`) so several processes keep the GPU busy.

Deliberately **not** done, because they would change results: mixed precision (FP16/BF16),
TF32, larger batches, GPU-side random encoding, or any change to epochs/learning rate/architecture.

Measured on CPU here: results track the previous implementation closely but are not bit-identical
(batched matmuls round differently, which can flip a rare spike). A short MNIST run gave
epoch-1 loss 1.4611 vs 1.4604 before. Do not mix runs made with the old and new code; the study
guard refuses to resume across code revisions anyway.

## Notebook setup

1. New Kaggle Notebook → Settings: **Accelerator = GPU T4 x2** (or T4), **Internet = On**,
   **Persistence = Files**.
2. Cells:

```python
!git clone https://github.com/SakshamDev/snn-study.git
%cd snn-study
!pip install -q snntorch
```

Kaggle already ships PyTorch/torchvision; do not install the Windows lock file there.

3. **GPU preflight** (about a minute; proves CUDA + deterministic mode work):

```python
!python scripts/smoke_test.py --config configs/smoke_cuda.json
```

If this raises "does not have a deterministic implementation", copy the config, set
`"deterministic": false` and use a new `results_dir`; report that in the paper's methods.

4. **Main study**:

```python
CFG = "configs/main_cuda.json"
!python scripts/prepare_data.py --config {CFG} --download
!python scripts/freeze_protocol.py --config {CFG}
# 3 workers on one GPU; with T4 x2 use --workers 4 --gpus 2
!python scripts/run_parallel.py --config {CFG} --stage train --workers 3
!python scripts/run_parallel.py --config {CFG} --stage validate --workers 3
!python scripts/analyze.py --config {CFG} --split validation
!python scripts/select.py --config {CFG}
!python scripts/run_parallel.py --config {CFG} --stage test --workers 3
!python scripts/analyze.py --config {CFG} --split test
```

Always run `prepare_data` and `freeze_protocol` *before* starting workers (they create files the
workers would otherwise race to write).

5. **Save results** before the session ends:

```python
!zip -rq results.zip runs/main_cuda data/splits
```

Download `results.zip` from the Output tab (or save it as a Kaggle Dataset).

## If a session stops

Completed runs are skipped on rerun. A run that was in progress is preserved and reported;
move that run's directory aside, then launch the same command again. Restore a previous
`results.zip` into the new session first (unzip into the repository root).

## Tuning

Start with `--workers 3`. If GPU utilisation (`!nvidia-smi`) is low and CPU is not saturated, add a
worker; if the CPU is saturated (4 vCPUs on Kaggle), use fewer. Each worker needs ~1 GB of GPU memory.
