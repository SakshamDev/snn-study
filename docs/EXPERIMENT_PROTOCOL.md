# Implemented experimental protocol

Status: a proposed undergraduate comparative study. The main experiment has not been run during repository creation. Read `VALIDATION.md` for the actual software verification scope.

## Question and scope

At short simulation windows, how does a simple hidden-spike penalty compare with raising hidden firing thresholds after training, under a one-percentage-point accuracy budget? The experiment measures algorithmic activity in simulated SNNs. It does not measure chip energy, event-camera performance, or biological learning.

## Exact model

The default network is 784–128–64–10 with fully connected biased layers. An ANN uses ReLU in the two hidden layers. An SNN uses LIF units in all three layers, with independently reset state for each image.

For every layer and timestep:

```text
v[t] = beta * u[t-1] + W * s_previous_layer[t] + b
s[t] = 1 if v[t] > theta, else 0
u[t] = v[t] - theta * stop_gradient(s[t])
```

The forward reset immediately subtracts one threshold, emitting at most one spike per step even for large overshoots. Equality with threshold does not fire. The derivative of the spike operation uses `snntorch.surrogate.atan(alpha=2)`. The reset term is detached during backward propagation. Leak and threshold are fixed buffers, not learned parameters.

Implementation clarification relative to the saved planning response: the explicit LIF cell supplies this exact state update; snnTorch supplies the surrogate derivative. The code does not rely on implicit state or reset defaults from `snn.Leaky`. Sequential layers use spikes from the same simulation step; no additional axonal delay is modeled.

Logits are `10 * output_spikes.mean(time)`. Prediction uses `argmax`, which chooses the first class index on ties. Ties and all-zero outputs are included in accuracy and reported separately.

The loss is `cross_entropy(logits, labels) + lambda * (hidden1.mean() + hidden2.mean()) / 2`. Each hidden mean averages batch, time, and neurons. This equally weights the two layer occupancies, not individual neurons or fanout costs. Output firing is not penalized. A different regularizer would be a different experiment.

## Input encoding and random streams

Pixels are flattened and divided by 255 without centering or augmentation. For normalized pixel x, each input event is Bernoulli(x). CPU-generated random numbers are arranged example-major with the maximum declared T, then prefixes are used for shorter windows. At evaluation, order, images, maximum T, and encoding seeds are shared across methods. This also makes encoding independent of evaluation batch partitioning.

Initialization uses the training seed. Data ordering uses seed + 20000 and training encoding uses seed + 10000. Separate validation and test encoding banks are declared in the config. The main study has five independent training seeds and three encoding realizations per trained SNN. ANN evaluation has no redundant encoding repeats.

The synthetic dataset is ten generated spatial patterns for software testing only. It never stands in for MNIST or Fashion-MNIST research evidence.

## Main grid and controls

T = 8, 16, 32; lambda = 0, 0.1, 1; training seeds = 11, 22, 33, 44, 55. Train each T separately. Use Adam, initial learning rate 0.001, batch 128, and initially 15 epochs. Pilot convergence checks must determine whether these defaults are suitable.

The baseline checkpoint at each T/seed is evaluated with hidden-threshold multipliers 1, 1.25, 1.5. The output threshold stays 1 and weights are frozen. Regularized models are evaluated at multiplier 1 only. The ANN has the same number of trainable weights and biases, with its own checkpoint chosen by validation.

This is a controlled small-model comparison, not a claim that the ANN or SNN is state of the art. Equal epochs mean equal numbers of data passes, not equal training computation across T. The provided ANN training defaults require the same development-stage convergence assessment as the SNN.

## Measurement definitions

- `spikes_layer0`: encoded input spikes per image over the full window.
- `spikes_layer1/2/3`: emitted events in hidden 1, hidden 2, and output.
- `occupancy_layerK`: layer spike count divided by T times its neuron count.
- `synops_layer1/2/3`: input-to-hidden1, hidden1-to-hidden2, and hidden2-to-output effective accumulations.
- `synops_total`: sum of those three operation counts.
- `synops_hidden_outputs`: sum of layers 2 and 3, which can be affected directly by hidden firing reductions.
- `inactive_layerK_fraction`: fraction of neurons that never fire across that evaluation partition and encoding realization; repeat values are averaged within a training seed.
- `neuron_updates`: T times the 202 modeled hidden/output neurons for the default SNN.
- `dense_simulator_macs`: T times the number of weight connections; records the dense linear workload used by this implementation.
- ANN `effective_macs`: nonzero source activations times nonzero outgoing weights, measured per image.
- ANN `dense_macs`: number of weight connections per forward pass.
- `software_forward_ms_per_image`: time around the instrumented forward call, including synchronization on CUDA; excludes encoding, loading, metric computation, and file output. No warmup benchmark suite is implemented, so this is descriptive runtime, not a rigorous cross-device latency claim.

Each active source event contributes its nonzero-weight fanout. For dense default weights, SynOps = `128*S0 + 64*S1 + 10*S2`. Zero-weight connections are excluded if they occur. Negative weights still count. Bias additions, leakage multiplications, comparisons, resets, encoder work, memory accesses, routing, and output decoding are excluded from SynOps; neuron updates and software workload are reported separately, without converting them to joules.

Footprint includes trainable parameter bytes, registered fixed-buffer bytes, and an analytic float32 membrane-state size per image. It is not peak process memory, training memory, checkpoint file size, or full device memory usage.

## Selection and analysis

Checkpoint selection maximizes mean validation accuracy across its encoding bank, with earliest-epoch tie breaking. Candidate selection minimizes mean validation total SynOps within one percentage point of the same-T lambda=0, multiplier=1 baseline. Candidates are averaged across training seeds; a common setting is selected for all seeds at each T.

Primary comparison: T=16, lambda=0.1, multiplier=1 versus T=16, lambda=0, multiplier=1. The main grid is evaluated and reported in full; other comparisons are descriptive/secondary. If a development pilot changes the primary penalty, amend the config before freezing and use a fresh results folder.

For analysis, average encoding realizations inside each trained model first. Use paired differences across the same training-seed IDs:

```text
accuracy_difference_pp = 100 * (accuracy_A - accuracy_B)
activity_reduction_percent = 100 * (1 - operations_A / operations_B)
```

The summary is the mean of per-seed effects, not a ratio of pooled totals. A zero baseline operation count makes the relative reduction undefined; the output stays missing rather than inventing a percentage.

Two-sided 95% Student t intervals use training seeds as the independent unit. A single seed receives no SD/CI. The accuracy-tolerance interpretation requires the lower interval bound to exceed minus the declared tolerance, and should be interpreted with the interval assumptions. There is no automatic significance search across the grid or claim of universal noninferiority. Selected-strategy intervals are conditional on validation selection and are secondary.

## Reproducibility and integrity

Study, protocol, and selection files record resolved config, code/data/split hashes, and dates. Training records the environment and checkpoint hash. Final testing checks those records and requires complete validation evidence. Failed or incomplete training runs are preserved; incomplete grids cannot silently enter analysis.

These checks help prevent accidental leakage and mixed experiments. They do not constitute external preregistration or make it impossible to inspect test data manually. For a public preregistration, deposit the plan yourself before the study if desired.

## Relevant sources

- [Eshraghian et al. (2023), Training Spiking Neural Networks Using Lessons From Deep Learning](https://arxiv.org/abs/2109.12894)
- [Yan et al. (2022), Backpropagation With Sparsity Regularization for Spiking Neural Network Learning](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.760298/full)
- [Yik et al. (2025), NeuroBench](https://www.nature.com/articles/s41467-025-56739-4)
- [Yan et al., analytical energy perspectives, version 6 (2026)](https://arxiv.org/abs/2409.08290v6)
- [snnTorch Leaky source documentation](https://snntorch.readthedocs.io/en/stable/_modules/snntorch/_neurons/leaky.html)
- [SURJ author guidelines](https://ojs.stanford.edu/ojs/index.php/surj/author-guidelines)

The complete literature-informed topic comparison and journal assessment are preserved in `research_response.md`.
