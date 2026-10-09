# Experimental Protocol: Accuracy–Activity Trade-offs in Spiking Neural Networks

This document formally specifies the experimental protocol, mathematical formulations, network dynamics, control baselines, and statistical analysis procedures for investigating accuracy–activity trade-offs in small Spiking Neural Networks (SNNs).

---

## 1. Research Objectives and Scope

Spiking Neural Networks offer event-driven, energy-efficient computation on neuromorphic hardware when internal spike rates are sparse. However, reducing spike rates often incurs an accuracy penalty. This project systematically investigates:

1. **Regularization vs. Threshold Adaptation:** How does a hidden-spike occupancy penalty ($\mathcal{L}_{\text{occupancy}}$) incorporated during surrogate-gradient backpropagation compare to post-training hidden-neuron threshold scaling ($\theta \leftarrow \theta \cdot \alpha$) in reducing total synaptic operations?
2. **Constrained Efficiency:** Can either strategy achieve significant activity reduction while remaining within a predefined accuracy budget (e.g., $\le 1.0$ percentage point drop relative to the unconstrained baseline)?
3. **Simulation Temporal Horizon:** How do short simulation windows ($T \in \{8, 16, 32\}$ timesteps) modulate these trade-offs and interact with equivalent Artificial Neural Network (ANN) reference baselines?

> **Scope Note:** This study quantifies algorithmic workload and activity proxies (synaptic operations, spike counts, occupancy) in simulated neural models. It does not measure physical neuromorphic chip power dissipation (joules), event-camera asynchronous latency, or biological plasticity mechanisms.

---

## 2. Model Architecture and Dynamics

### 2.1 Spiking Neural Network (SNN)

The primary network is a fully connected 784–128–64–10 Leaky Integrate-and-Fire (LIF) architecture with learnable biases.

For each layer $l$ and discrete timestep $t \in \{1, \dots, T\}$:

$$\mathbf{v}_l[t] = \beta \cdot \mathbf{u}_l[t-1] + \mathbf{W}_l \mathbf{s}_{l-1}[t] + \mathbf{b}_l$$

$$\mathbf{s}_l[t] = \Theta(\mathbf{v}_l[t] - \theta_l) = \begin{cases} 1 & \text{if } \mathbf{v}_l[t] > \theta_l \\ 0 & \text{if } \mathbf{v}_l[t] \le \theta_l \end{cases}$$

$$\mathbf{u}_l[t] = \mathbf{v}_l[t] - \theta_l \cdot \text{sg}(\mathbf{s}_l[t])$$

Where:
- $\beta = 0.9$ is the membrane potential decay factor (fixed buffer).
- $\theta_l = 1.0$ is the base firing threshold (fixed buffer).
- $\mathbf{u}_l[t]$ is the post-reset membrane potential, implementing an **immediate subtractive reset**.
- $\text{sg}(\cdot)$ is the `stop_gradient` operator, ensuring reset dynamics do not propagate gradient artifacts.
- Surrogate derivative during backpropagation uses the arctangent formulation:
  $$\frac{\partial \mathbf{s}}{\partial \mathbf{v}} = \frac{1}{\pi (1 + (\alpha (\mathbf{v} - \theta))^2)}, \quad \alpha = 2.0$$
- State is strictly local to each forward call and initialized to zero ($\mathbf{u}[0] = \mathbf{0}$).

### 2.2 Output Decoding and Loss Function

Class predictions are derived from the mean firing rate of the 10 output neurons across the simulation window, scaled by a logit factor $\gamma = 10.0$:

$$\mathbf{z} = \gamma \cdot \frac{1}{T} \sum_{t=1}^T \mathbf{s}_{\text{out}}[t]$$

$$\hat{y} = \arg\max_c (\mathbf{z}_c)$$

The training loss combines standard cross-entropy with an equally weighted hidden-layer spike occupancy penalty:

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\mathbf{z}, y) + \lambda \cdot \frac{1}{2} \left( \bar{s}_{\text{hidden1}} + \bar{s}_{\text{hidden2}} \right)$$

where $\bar{s}_l = \frac{1}{B \cdot T \cdot N_l} \sum_{b=1}^B \sum_{t=1}^T \sum_{i=1}^{N_l} s_{l, i}[t, b]$ is the mean spike occupancy for layer $l$. Output layer spikes are unpenalized.

### 2.3 Artificial Neural Network (ANN) Reference

The reference ANN shares the identical topology (784–128–64–10) and parameter count, utilizing standard ReLU activations in hidden layers:

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{z}_{\text{ANN}} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3$$

---

## 3. Input Encoding and Randomness Controls

- **Bernoulli Rate Encoding:** Normalized grayscale inputs $x \in [0, 1]^{784}$ generate independent Poisson/Bernoulli spike trains where $P(s_{\text{in}}[t] = 1) = x$.
- **Shared Prefixes:** Random numbers are generated in example-major order across the maximum temporal window ($T_{\text{max}} = 32$), ensuring identical spike realization prefixes for shorter sub-windows ($T=8, 16$).
- **Seed Hierarchy:**
  - Training seeds: $S \in \{11, 22, 33, 44, 55\}$ (5 independent runs per condition).
  - Data shuffle seed: $\text{seed} + 20000$.
  - Training encoding generator: $\text{seed} + 10000$.
  - Evaluation encoding banks: 3 distinct validation seeds $\{101, 102, 103\}$ and 3 distinct test seeds $\{201, 202, 203\}$.

---

## 4. Experimental Grid

The primary benchmark dataset is **Fashion-MNIST** (54,000 train / 6,000 validation / 10,000 test).

| Parameter | Values |
|---|---|
| Timesteps ($T$) | $8, 16, 32$ |
| Regularization Penalty ($\lambda$) | $0.0, 0.1, 1.0$ |
| Threshold Multipliers ($\alpha$) | $1.0, 1.25, 1.5$ (evaluated on $\lambda=0.0$ baselines) |
| Optimizer | Adam ($\text{lr} = 10^{-3}$, batch size $= 128$, epochs $= 15$) |
| Replications | 5 independent seeds per configuration (45 SNN + 5 ANN runs = 50 training runs) |

### Interventions Under Comparison:
1. **Training-time Regularization:** Train SNN with $\lambda \in \{0.1, 1.0\}$, keep threshold multiplier $\alpha = 1.0$.
2. **Post-training Threshold Scaling:** Train unregularized baseline ($\lambda = 0.0$), evaluate at test time with modified hidden thresholds $\theta_1, \theta_2 \leftarrow \theta \cdot \alpha$ for $\alpha \in \{1.25, 1.5\}$, while keeping weights and output threshold $\theta_3 = 1.0$ fixed.

---

## 5. Quantitative Metrics and Activity Accounting

- **Layer Spikes ($S_l$):** Average total spikes emitted per image across the $T$-step window in layer $l$.
- **Layer Occupancy ($O_l$):** $\frac{S_l}{T \cdot N_l} \in [0, 1]$.
- **Effective Synaptic Operations ($\text{SynOps}$):** Event-driven accumulation count. An active spike in layer $l-1$ triggers additions across all non-zero outgoing synaptic connections to layer $l$:
  $$\text{SynOps}_{\text{total}} = \sum_{l=1}^3 \sum_{i \in \text{active}_{l-1}} \|\mathbf{w}_{l, :, i}\|_0$$
  For dense weights: $\text{SynOps}_{\text{total}} = 128 \cdot S_0 + 64 \cdot S_1 + 10 \cdot S_2$.
- **Hidden SynOps:** $\text{SynOps}_{\text{hidden}} = 64 \cdot S_1 + 10 \cdot S_2$ (activity directly manipulable by hidden-layer intervention).
- **ANN Effective MACs:** Multiply-accumulate operations accounted as non-zero input activations multiplied by non-zero outgoing weights per layer.
- **Model Footprint:** Analytic tracking of parameter storage bytes and per-image membrane state requirements.

---

## 6. Selection and Statistical Analysis

### 6.1 Validation-Gated Strategy Selection
Candidate strategies are selected purely on the validation partition:
- Define accuracy tolerance margin: $\delta = 1.0$ percentage point.
- For each $T$, identify candidates satisfying:
  $$\overline{\text{Acc}}_{\text{val}}(\text{candidate}) \ge \overline{\text{Acc}}_{\text{val}}(\text{baseline}_{T}) - \delta$$
- Among eligible candidates, select the one that minimizes mean validation $\text{SynOps}_{\text{total}}$.

### 6.2 Statistical Estimation
- Repeat measurements over encoding realizations are averaged within each seed before calculating across-seed statistics.
- Pairwise comparisons (e.g., Regularized vs. Baseline) compute per-seed paired differences:
  $$\Delta \text{Acc} = \text{Acc}_A - \text{Acc}_B$$
  $$\% \Delta \text{Activity} = 100 \times \left(1 - \frac{\text{SynOps}_A}{\text{SynOps}_B}\right)$$
- Report two-sided 95% Student's $t$ confidence intervals across the $N=5$ training seeds.

---

## 7. References

1. Eshraghian, J. K., et al. (2023). "Training Spiking Neural Networks Using Lessons From Deep Learning." *Proceedings of the IEEE*, 111(9), 1016–1054.
2. Yan, Z., et al. (2022). "Backpropagation With Sparsity Regularization for Spiking Neural Network Learning." *Frontiers in Neuroscience*, 16, 760298.
3. Yik, J., et al. (2025). "NeuroBench: Advancing Neuromorphic Computing through Collaborative, Fair and Representative Benchmarking." *Nature Communications*, 16, 56739.
