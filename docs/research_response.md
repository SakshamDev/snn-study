The most suitable project is a **small, controlled experimental study of an SNN design trade-off**, supported by established methods, repeated trials, and careful interpretation. You do not need to invent a neuron model or learning algorithm. You do need to answer a question that goes beyond reproducing a tutorial’s accuracy.

My primary recommendation is:

**“Accuracy–Activity Trade-offs in Spiking Neural Networks: Comparing Spike Regularization and Post-Training Threshold Adjustment.”**

The central question is whether encouraging fewer spikes during training offers a useful advantage over simply raising neurons’ firing thresholds afterward—and whether either approach meaningfully reduces whole-network computational activity.

This gives you an accessible introduction to neuromorphic computing, an implementable experiment, and a defensible connection to hardware. **Publication remains uncertain:** rigorous execution strengthens an incremental contribution, but does not automatically make it sufficiently original for SURJ.

This assessment uses sources available through **October 9, 2026**, rather than assuming coverage of the rest of October. No experiments have been run and no outcomes below are presented as results.

**1. What SURJ’s requirements and recent publications establish**

SURJ’s published requirements are reasonably compatible with your proposed project:

| Item | Published guidance |
|---|---|
| Eligibility | Current undergraduates from any institution; manuscripts may also be submitted within a year of graduation. |
| Article scope | SURJ considers “research-oriented papers,” including work beyond summaries of existing projects. |
| Length | Approximately 2,000–4,000 words; editors may accommodate and subsequently adjust other lengths. |
| Abstract | Approximately 200 words, within a 150–250-word range. |
| Review | Manuscripts must be anonymized for double-blind review. |
| Format | DOC, DOCX, or RTF preferred. |
| Figures and citations | Figures should be vector or at least 150 dpi; references use numbered, bracketed citations. |

These are journal requirements, drawn from its [Author Guidelines](https://ojs.stanford.edu/ojs/index.php/surj/author-guidelines). Its [submission checklist](https://ojs.stanford.edu/ojs/index.php/surj/about/submissions) additionally addresses prior publication, simultaneous consideration, separate submission of images/data, and authorship.

There is relevant publication precedent:

| Recent SURJ paper | What it establishes about subject fit |
|---|---|
| Merit Ghodrat, **“Efficient Deep Learning for Melanoma Diagnosis on CPU Hardware,” June 3, 2025** | A software-based engineering study using familiar neural-network components and reporting predictive performance and CPU execution measurements. [Article](https://ojs.stanford.edu/ojs/index.php/surj/article/view/3751) |
| Danielle Yang, **“Machine Learning for Real Time Classification of Transient Events: a Recurrent Neural Network Auto-Encoder and Gradient Boosting Classifier,” June 3, 2025; updated July 3, 2025** | A computational application combining established methods and evaluating them on an existing dataset. [Article](https://ojs.stanford.edu/ojs/index.php/surj/article/view/3729) |
| Goyal et al., **“NeurOM: Analyzing the Effects of OM Chants on Human Brain Activity Using EEG and MachineLearning,” March 8, 2026** | A neuroscience-related study comparing established machine-learning methods. [Article](https://ojs.stanford.edu/ojs/index.php/surj/article/view/3687) |

The 2026 issue also provides an example of presentation using an abstract, introductory explanation, related work, methodology, and quantitative discussion. That is useful stylistic precedent, rather than a mandatory section template. [Winter 2026 issue](https://ojs.stanford.edu/ojs/index.php/surj/issue/download/188/82)

**My interpretation of suitability:** a focused comparative study is a plausible fit for Natural Sciences and Engineering. The published examples show that computational research built from established techniques can appear in SURJ.

However, the guidelines I examined do **not explicitly define an acceptance policy for replication studies or benchmark comparisons**, or specify a minimum novelty threshold. “Research-oriented papers” leaves room for interpretation; it is not an assurance that a straightforward reproduction will be accepted. A research-oriented review might fit that broad wording, but hands-on experimentation better serves your objectives.

For your project, I would aim for:

- One clearly stated research question.
- A justified comparison with meaningful controls.
- Several independently trained models.
- Transparent uncertainty and failure reporting.
- A specific empirical finding whose scope is honestly bounded.

**2. The concepts behind the shortlist**

An ordinary neural-network activation is a number. A spiking neuron instead maintains an internal state and emits discrete events over time.

Think of a **leaky integrate-and-fire neuron** as a bucket:

- Incoming signals add water.
- Some water leaks away between steps.
- Crossing a marked level produces a spike.
- A reset reduces the water level afterward.

The bucket’s level represents **membrane potential**. The marked level is the **firing threshold**. The leak determines how long earlier inputs remain influential. This captures selected features of biological neurons while omitting much of their cellular complexity. [Neuronal Dynamics, introductory chapters](https://neuronaldynamics.epfl.ch/online/)

Three distinctions matter throughout this plan:

| Concept | Intuitive meaning | Experimental implication |
|---|---|---|
| Rate encoding | Stronger inputs produce more spikes over an observation window. | Longer windows can improve the estimate, while generating more events. |
| Temporal encoding | Information is represented by when spikes occur. | Timing resolution and the decoding rule become important. |
| Direct/current input | Numerical inputs directly drive spiking neurons. | Hidden layers can spike even though input processing uses numerical values rather than binary input events. |

Input encoding and output decoding are separate choices. A temporally encoded input does not automatically make the whole network a time-to-first-spike system. [snnTorch spike-encoding tutorial](https://snntorch.readthedocs.io/en/stable/tutorials/tutorial_1.html), [training tutorial](https://snntorch.readthedocs.io/en/stable/tutorials/tutorial_5.html)

Likewise, neuromorphic computing encompasses hardware and computational principles beyond SNNs. Training an SNN in PyTorch studies an algorithm relevant to that field; it does not reproduce the behavior or efficiency of a particular chip. [Schuman et al., 2022](https://doi.org/10.1038/s43588-021-00184-y)

**3. A verified literature foundation**

This is a selective reading list organized around your proposed experiments, rather than an exhaustive literature review. Dates distinguish published versions from preprints where relevant.

| Source and date | Contribution and relevance |
|---|---|
| **Neftci, Mostafa, and Zenke, “Surrogate Gradient Learning in Spiking Neural Networks,” 2019; initial preprint January 28** | Explains why discrete spikes complicate gradient-based training and introduces surrogate-gradient approaches. Essential methodological background. [Paper](https://arxiv.org/abs/1901.09948) |
| **Guo et al., “Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems,” March 2021** | Compares rate, time-to-first-spike, phase, and burst coding in an STDP-trained system, including hardware nonidealities. Establishes that coding comparisons are existing research, with conclusions tied to the studied system. [Paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full) |
| **Schuman et al., “Opportunities for neuromorphic computing algorithms and applications,” January 31, 2022** | Places algorithms within the broader neuromorphic hardware/software landscape and motivates evaluating applications and implementation constraints together. [Paper](https://doi.org/10.1038/s43588-021-00184-y) |
| **Eshraghian and Lu, “The fine line between dead neurons and sparsity in binarized spiking neural networks,” January 28, 2022, preprint** | Investigates the tension between sparse firing and neurons that stop firing, in binarized networks. Useful motivation for monitoring inactive neurons during threshold experiments. [Paper](https://arxiv.org/abs/2201.11915) |
| **Kim et al., “Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?”, ICASSP 2022; initial preprint January 31** | Compares coding choices across accuracy, adversarial robustness, and hardware-related cost. Supports treating encoding as an experimental variable rather than assuming one universally superior method. [Paper](https://arxiv.org/abs/2202.03133) |
| **Yan et al., “Backpropagation With Sparsity Regularization for Spiking Neural Network Learning,” April 14, 2022** | Studies training objectives that promote spike and synaptic sparsity. Direct evidence that sparsity regularization is established—not a novel algorithm you should claim to introduce. [Paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.760298/full) |
| **Bu et al., “Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks,” ICLR 2022; arXiv posting March 8, 2023** | Analyzes conversion error and introduces a quantization clip-floor-shift activation to improve short-window conversion. Important context for a conversion project; a basic converter is a limited baseline against this literature. [Conference manuscript](https://yuzhaofei.github.io/papers/22-ICLR-Optimal%20ANN-SNN%20Conversion%20for%20High-accuracy%20and%20Ultra-low-latency%20Spiking%20Neural%20Networks.pdf), [arXiv record](https://arxiv.org/abs/2303.04347) |
| **Eshraghian et al., “Training Spiking Neural Networks Using Lessons From Deep Learning,” September 6, 2023; initial preprint 2021** | Accessible tutorial covering neuron dynamics, encoding, training, and snnTorch. Your best main introductory reference. [Published paper](https://ieeexplore.ieee.org/document/10242251/), [open manuscript](https://arxiv.org/abs/2109.12894) |
| **Sakemi et al., “Sparse-firing regularization methods for spiking neural networks with time-to-first spike coding,” July 24, 2023, preprint** | Develops two regularization methods for TTFS networks and evaluates sparsity–accuracy trade-offs on image datasets. Shows that even sparsification of already sparse temporal networks has prior work. [Paper](https://arxiv.org/abs/2307.13007) |
| **Stanojevic et al., “High-performance deep spiking neural networks with 0.3 spikes per neuron,” August 9, 2024** | Studies specialized TTFS training and ANN–SNN relationships, including timing and quantization constraints. Demonstrates sophisticated sparse-network results, but is considerably more ambitious than your first project needs to be. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11315905/) |
| **Yik et al., “The neurobench framework for benchmarking neuromorphic computing algorithms and systems,” February 11, 2025** | Separates algorithm-level evaluation from system-level measurement and defines useful metrics for correctness, footprint, sparsity, and effective operations. This should guide your reporting. [Paper](https://www.nature.com/articles/s41467-025-56739-4) |
| **Ding et al., “Neuromorphic computing paradigms enhance robustness through spiking neural networks,” November 19, 2025** | Examines how encoding, temporal processing, decoding, and training interact under adversarial perturbations. Supports studying a precisely specified robustness setting rather than claiming inherent SNN robustness. [Paper](https://www.nature.com/articles/s41467-025-65197-x) |
| **Yan, Bai, Tang, and Wong, “Reconsidering the Energy Efficiency of Spiking Neural Networks Inference from Analytical Perspectives,” initial preprint August 29, 2024; version 6 August 4, 2026** | Reassesses efficiency using analytical models that include computation and data movement, with quantized ANN comparators. The current record reports TCAD acceptance. Useful for understanding why spike counts cannot establish energy savings. [Version 6](https://arxiv.org/abs/2409.08290v6) |
| **Du et al., “Benchmarking spiking neural networks across sensing modalities on edge devices,” August 27, 2026, preprint** | Reports a deployment-oriented comparison across sensing modalities and devices, finding task-dependent advantages and costs. Recent context, with preprint status kept explicit. [Paper](https://arxiv.org/abs/2609.00026) |

The established findings leave room for a modest, useful investigation:

| Already established | A feasible undergraduate investigation |
|---|---|
| SNN accuracy depends on coding, dynamics, training, and observation length. | Quantify a particular interaction under a fixed, reproducible protocol. |
| Spike penalties can reduce activity. | Compare them with an inexpensive post-training adjustment at a predefined accuracy budget. |
| Shorter windows change both decisions and activity. | Separate networks trained for short windows from long-window networks merely truncated at inference. |
| Robustness depends on the perturbation and model configuration. | Compare specific natural corruptions while controlling clean accuracy and encoding randomness. |
| Sparse activity can matter for hardware cost. | Measure layer-specific effective operations and identify costs omitted by that proxy. |

The contribution should be the **controlled comparison and resulting explanation**, not rediscovery of those broad facts.

**4. Six possible research topics**

For all six topics, use a shared minimum standard: a fixed training/validation/test split, five independent training seeds where feasible, validation-only tuning, reported effect sizes and uncertainty, and disclosure of unsuccessful configurations.

**Topic 1 — Accuracy–Activity Trade-offs in Spiking Neural Networks: Comparing Spike Regularization and Post-Training Threshold Adjustment**

- **A–B. Title and explanation:** The title above describes two ways to make a network quieter. One teaches it to use fewer spikes; the other makes its neurons harder to trigger after training.
- **C. Research question:** At short simulation windows, which approach reduces computational activity while keeping accuracy within a predefined tolerance?
- **D. Hypothesis:** A moderate spike penalty may preserve accuracy better than a large threshold increase because weights can adapt during training. Excessive penalties may impair learning. This is a prediction; prior sparsity research does not guarantee that regularization will win in your setup.
- **E. Methodology:** Train a small LIF multilayer perceptron on Fashion-MNIST across three window lengths and three penalty strengths. Compare with unregularized SNNs, threshold-adjusted versions of those SNNs, and a similarly sized ReLU ANN. Measure accuracy, per-layer spikes, effective synaptic operations, inactive neurons, and uncertainty across seeds.
- **F. Implementation:** PyTorch and snnTorch; laptop CPU feasible with a bounded grid, optional GPU helpful. Difficulty **2/5**. Main challenges: preserving gradients through the penalty, avoiding inactive networks, and counting operations correctly.
- **G. Results and figures:** Accuracy–activity curves, window/penalty heatmaps, layer-cost breakdowns, and paired regularization-versus-threshold comparisons.
- **H. Contribution:** Comparative study with a sensitivity analysis. The penalty itself is established; the contribution is its measured value relative to a simpler alternative.
- **I. SURJ suitability:** Strong explanatory potential and manageable scope. The main weakness is incremental novelty. A tutorial-level “fewer spikes with regularization” result would be too thin.

**Topic 2 — Rate and Latency Encoding in Small Spiking Neural Networks: A Controlled Comparison**

- **A–B. Title and explanation:** Compare representing brightness through repeated spikes with representing brightness through earlier spikes.
- **C. Research question:** How do input encodings change accuracy, variability, and activity under the same simulation budget?
- **D. Hypothesis:** Latency encoding should generate fewer input events by construction, but may be harder to train or more sensitive to timing resolution. Lower input activity does not guarantee lower whole-network activity.
- **E. Methodology:** Train separate, identical-sized networks for Bernoulli rate and latency input encoding at several window lengths. Include direct input as an explicitly numerical-input baseline. Use Fashion-MNIST; keep the output decoder fixed. Report both a shared-configuration comparison and results after equal-budget validation tuning.
- **F. Implementation:** PyTorch, snnTorch’s encoding utilities. Difficulty **3/5**; CPU feasible. Encoder scaling, silent pixels, loss selection, and fair tuning are the main challenges.
- **G. Results and figures:** Example spike rasters, accuracy versus window length, input/hidden event counts, and variability across input realizations.
- **H. Contribution:** Comparative reproduction of established coding questions in a clearly defined small-model setting.
- **I. SURJ suitability:** Excellent teaching value. Weakness: extensive prior comparisons, and an apparent encoding advantage can actually reflect an unsuitable training recipe.

**Topic 3 — Memory and Firing Threshold in Leaky Integrate-and-Fire Networks: A Sensitivity Analysis**

- **A–B. Title and explanation:** Investigate how long neurons remember input and how much evidence they require before firing.
- **C. Research question:** Which combinations of leak and threshold produce reliable classification without excessive activity or widespread inactivity?
- **D. Hypothesis:** Very high thresholds may suppress useful activity; strong leakage may prevent evidence accumulation. Interactions and weight adaptation may produce non-monotonic outcomes.
- **E. Methodology:** Use a compact grid of fixed leak and threshold values, retraining each configuration. Compare with a reference configuration and ANN. Separately evaluate post-training parameter changes on frozen reference weights. Measure accuracy, spikes, operation estimates, and the proportion of neurons inactive over the evaluation set.
- **F. Implementation:** PyTorch and snnTorch. Difficulty **2/5**; CPU-friendly if the grid stays small. The main challenge is distinguishing a training effect from sensitivity of an already trained model.
- **G. Results and figures:** Parameter heatmaps, activity distributions, example membrane traces, and seed variation.
- **H. Contribution:** Sensitivity analysis and reproducible documentation of failure regimes.
- **I. SURJ suitability:** Safest implementation choice. Its weakness is contribution strength: a parameter sweep needs an explanatory finding, rather than simply identifying the highest accuracy.

**Topic 4 — Classification Under Input Corruption: Comparing Spiking and ReLU Networks at Matched Model Size**

- **A–B. Title and explanation:** Test how ordinary and spiking networks behave when images become noisy.
- **C. Research question:** Does either model retain accuracy better under specified corruptions after accounting for its clean performance?
- **D. Hypothesis:** Integration and thresholding may suppress some small perturbations, while other perturbations may increase unwanted spikes. No universal SNN advantage is predicted.
- **E. Methodology:** Train small ANN and SNN models on clean Fashion-MNIST. Apply clipped Gaussian noise and salt-and-pepper corruption at predefined severities. Use identical corrupted images for both models and repeated corruption/encoding realizations. Report absolute accuracy and degradation from clean accuracy.
- **F. Implementation:** PyTorch, snnTorch, NumPy, and scikit-learn. Difficulty **2–3/5**; laptop feasible. Clean-accuracy differences, clipping, and stochastic encoding are important confounders.
- **G. Results and figures:** Corruption examples, accuracy–severity curves, class-specific degradation, and spike activity under noise.
- **H. Contribution:** Comparative robustness study for a restricted corruption setting.
- **I. SURJ suitability:** Accessible and interpretable. The main weakness is overgeneralization: natural-corruption results cannot establish adversarial robustness or general superiority.

**Topic 5 — Adaptive Stopping in Spiking Classification: Accuracy Versus Decision Time**

- **A–B. Title and explanation:** Let a network stop processing an image once its accumulated evidence is sufficiently decisive.
- **C. Research question:** Can an established stopping rule reduce average simulated steps and activity while preserving a specified accuracy level?
- **D. Hypothesis:** Easier examples may permit earlier decisions; difficult or corrupted examples may require longer observation.
- **E. Methodology:** Train a small SNN at a fixed maximum window. Compare fixed-duration inference with a spike-count margin rule and a minimum waiting period, all selected using validation data. Evaluate accuracy, average and tail stopping times, activity, and premature errors.
- **F. Implementation:** PyTorch and snnTorch. Difficulty **3/5**. Counting only activity before each example’s stopping time and avoiding confidence-selection leakage require care.
- **G. Results and figures:** Accuracy–steps curves, stopping-time distributions, activity savings, and early-error examples.
- **H. Contribution:** Evaluation of an established inference policy; avoid presenting adaptive stopping as a new concept.
- **I. SURJ suitability:** A clear engineering story with useful depth. Higher implementation risk than Topics 1 and 3, especially if simulated stopping is confused with actual device acceleration.

**Topic 6 — ANN-to-SNN Conversion in Small Networks: Calibration and Simulation-Window Trade-offs**

- **A–B. Title and explanation:** Train an ordinary network, then translate its numerical activations into spiking behavior.
- **C. Research question:** How do calibration choices and simulation length affect the converted network’s accuracy gap and activity?
- **D. Hypothesis:** Longer windows may improve rate approximation, while poor calibration may cause saturation or insufficient firing.
- **E. Methodology:** Train a small conversion-compatible ReLU network. Use an established converter with integrate-and-fire neurons, compare two documented calibration choices, and evaluate several windows. Include the original ANN and, if resources allow, a directly trained SNN. Calibration uses training or validation data only.
- **F. Implementation:** PyTorch with a documented SpikingJelly conversion workflow; avoid building a full converter from scratch. Difficulty **3–4/5**. Bias handling, reset behavior, unsupported layers, and calibration are common complications. [SpikingJelly paper](https://doi.org/10.1126/sciadv.adi1480)
- **G. Results and figures:** Accuracy gap versus window length, calibration comparisons, layer activation/rate mismatch, and operation estimates.
- **H. Contribution:** Reproduction and calibration sensitivity analysis.
- **I. SURJ suitability:** Relevant to your hardware transition. The weakness is that basic conversion limitations are thoroughly studied, while competitive modern conversion adds substantial complexity.

**5. Comparison and ranking**

These scores are mentoring judgments about the **specified projects**, not measured probabilities of success or acceptance.

A score of 9–10 indicates a strong fit; 7–8 indicates a good fit with manageable qualifications; 5–6 indicates a material difficulty or limitation.

For the overall score, I give beginner friendliness and implementation ease **20% each**, experimental depth **15%**, interpretability **10%**, educational value **5%**, hardware relevance **10%**, contribution **10%**, and manuscript fit **10%**.

| Rank | Topic | Beginner | Ease | Depth | Interpretable results | Education | Hardware relevance | Contribution | 2,000–4,000-word fit | Weighted score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Spike regularization versus threshold adjustment | 9 | 8 | 9 | 9 | 10 | 9 | 7 | 9 | **8.65** |
| 2 | LIF parameter sensitivity | 9 | 9 | 7 | 9 | 9 | 8 | 6 | 9 | **8.30** |
| 3 | Input-corruption robustness | 8 | 8 | 8 | 8 | 9 | 7 | 7 | 9 | **7.95** |
| 4 | Adaptive stopping | 7 | 6 | 9 | 8 | 9 | 9 | 8 | 8 | **7.70** |
| 5 | Encoding comparison | 7 | 7 | 8 | 7 | 9 | 9 | 6 | 8 | **7.45** |
| 6 | ANN-to-SNN conversion | 6 | 6 | 8 | 8 | 9 | 9 | 7 | 8 | **7.25** |

The differences have concrete reasons:

- **Topic 1** combines a familiar loss modification with a meaningful competing method. Layer-level accounting can explain why apparent improvements do—or do not—matter.
- **Topic 3** is easiest to implement and debug, but faces the greatest risk of becoming a descriptive sweep.
- **Topic 4** has straightforward experiments, although fair interpretation requires controlling clean performance and noise generation.
- **Topic 5** offers strong depth but introduces stopping-policy and accounting complications.
- **Topic 2** is deceptively difficult to compare fairly because encoding and training interact.
- **Topic 6** introduces another implementation stack and requires understanding conversion assumptions before interpreting failures.

My three recommendations are **Topics 1, 3, and 4**, in that order. Choose Topic 1 as the main project. Topic 3 is the fallback if training regularization proves difficult to stabilize.

**6. Detailed plan for the recommended topic**

**Research question and intended contribution**

Use this question:

> Under short simulation windows, can a simple hidden-layer spike penalty reduce computational activity within a predefined accuracy budget, and how does its trade-off compare with post-training threshold adjustment?

The paper would provide a **controlled comparative study**, not a new sparsification algorithm or a direct replication of the complete BPSR method.

Its strongest contribution would be explaining the relationship between:

1. The mechanism used to suppress firing.
2. The available simulation window.
3. Classification performance.
4. Hidden-layer activity.
5. Whole-network effective operations.

**Research foundation**

You already have most of the prerequisites. Review:

| Prerequisite | Why it matters |
|---|---|
| Matrix multiplication and tensor dimensions | Implementing layers and counting connections. |
| Cross-entropy and gradient descent | Understanding the training objective. |
| Chain rule and computational graphs | Understanding surrogate gradients. |
| Basic recurrent networks/backpropagation through time | An SNN’s internal state persists across steps. |
| Bernoulli random variables | Understanding stochastic input encoding. |
| Means, standard deviations, paired differences, confidence intervals | Reporting repeated experiments. |

You do not initially need Hodgkin–Huxley equations, detailed ion-channel models, STDP implementation, or circuit design.

Read in this order:

1. snnTorch [Tutorial 1: Spike Encoding](https://snntorch.readthedocs.io/en/stable/tutorials/tutorial_1.html).
2. [Tutorial 3: A Feedforward Spiking Neural Network](https://snntorch.readthedocs.io/en/stable/tutorials/tutorial_3.html).
3. [Tutorial 5: Training SNNs](https://snntorch.readthedocs.io/en/stable/tutorials/tutorial_5.html).
4. Selected sections of Eshraghian et al. on dynamics and training.
5. Yan et al. on sparsity regularization.
6. NeuroBench’s metric definitions.

Biologically, the useful concepts are **integration, leakage, thresholding, reset, and sparse communication**. Surrogate-gradient training is an engineering technique for optimizing these models; this experiment would not establish how biological brains learn.

**Primary hypothesis**

A moderate hidden-layer spike penalty will reduce hidden-layer activity with a small accuracy cost relative to an unregularized SNN at the same simulation length.

Secondary hypotheses:

- Stronger penalties will eventually degrade accuracy or produce inactive neurons.
- Post-training threshold increases may achieve some of the same benefit without retraining.
- Hidden-layer spike reductions will overstate whole-network savings when input processing dominates.
- The useful trade-off will depend on simulation length.

None of these should be assumed true before experimentation.

Choose an **accuracy budget of one percentage point** relative to the unregularized SNN at the same window length. This is a proposed engineering tolerance, not a literature result or journal requirement.

**Dataset and preprocessing**

Use **Fashion-MNIST as the primary dataset**. It contains 60,000 training and 10,000 test images, each 28 × 28 pixels, across ten classes. [Dataset repository](https://github.com/zalandoresearch/fashion-mnist)

Recommended protocol:

- Create one fixed, stratified **54,000/6,000 training/validation split**.
- Preserve the official 10,000-image test set.
- Scale pixels to \([0,1]\).
- Flatten images to 784 features.
- Use identical images and preprocessing for all methods.
- Begin without augmentation.
- Save the split indices.

Use MNIST for initial debugging. A limited MNIST replication can later test whether the central finding transfers, but should not displace completing the primary experiment.

For Bernoulli rate encoding:

\[
s_{0,i,t}\sim\operatorname{Bernoulli}(x_i).
\]

Here \(x_i\) is normalized pixel intensity. A brighter pixel has a higher probability of producing a spike at each step. This is a discrete Bernoulli rate encoder; calling it an exact continuous-time Poisson process would be inaccurate.

**Architecture and dynamics**

Use a deliberately small network:

```text
784 inputs
→ Linear(784, 128) → LIF
→ Linear(128, 64)  → LIF
→ Linear(64, 10)   → LIF
→ accumulated output spikes
```

Start with:

| Setting | Initial value |
|---|---|
| Hidden widths | 128 and 64 |
| Membrane retention \(\beta\) | 0.9 |
| Threshold \(\theta\) | 1.0 |
| Reset | Subtract threshold |
| Learned leak/threshold | Disabled |
| Simulation windows | 8, 16, 32 steps |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | \(10^{-3}\) |
| Training duration | Initially 15 epochs; finalize after pilot convergence checks |
| Main training seeds | 11, 22, 33, 44, 55 |

These are proposed starting settings, not claimed optimal values.

An immediate-reset formulation is:

\[
v_{\ell,t}
=\beta u_{\ell,t-1}+W_\ell s_{\ell-1,t}+b_\ell,
\]

\[
s_{\ell,t}=H(v_{\ell,t}-\theta),
\]

\[
u_{\ell,t}=v_{\ell,t}-\theta s_{\ell,t}.
\]

The neuron integrates input into \(v\), produces a binary spike \(s\), and resets its stored potential \(u\).

Document the implementation’s exact reset convention. In snnTorch, set reset behavior explicitly rather than relying on defaults; its `Leaky` interface exposes both reset mechanism and reset delay. [Leaky documentation](https://snntorch.readthedocs.io/en/stable/snn.neurons_leaky.html)

**Training objective**

Let \(r_1\) and \(r_2\) denote the average spike occupancy of the two hidden layers over the batch, neurons, and timesteps:

\[
r_\ell=
\frac{1}{BTN_\ell}
\sum_{b,t,i}s_{\ell,b,t,i}.
\]

Use:

\[
\mathcal L
=\operatorname{CE}(z,y)
+\lambda\frac{r_1+r_2}{2}.
\]

For output logits, a consistent choice across window lengths is:

\[
z_c=10\left(\frac{1}{T}\sum_{t=1}^{T}s_{3,c,t}\right).
\]

The fixed scale of 10 is an explicit implementation choice. Normalizing counts by \(T\) avoids changing the logit scale automatically when the window changes.

Start with:

\[
\lambda\in\{0,\;0.1,\;1.0\}.
\]

During a small validation-only pilot, you may rescale this grid if both nonzero settings have negligible effects or both destroy learning. Record the change, then freeze the grid before the main experiments.

Important details:

- Apply the penalty to hidden spikes, leaving output spikes unpenalized initially.
- Keep the penalty connected to the computational graph.
- Use surrogate gradients for spike-producing neurons.
- Reset all neuron states between independent images/batches.
- Generate fresh training spike realizations.
- Use the same loss normalization in every condition.

**Baselines and controls**

| Method | Role |
|---|---|
| Unregularized SNN, \(\lambda=0\) | Establishes the effect of adding the penalty. |
| Regularized SNNs | Test whether learned adaptation improves the trade-off. |
| Frozen unregularized SNN with higher hidden thresholds | Tests whether a simpler deployment adjustment achieves similar benefits. |
| ReLU ANN with widths 784–128–64–10 | Provides a familiar accuracy and model-size reference. |

For threshold adjustment, initially test a common multiplier:

\[
a\in\{1.0,\;1.25,\;1.5\}
\]

on both hidden-layer thresholds. Keep the output threshold fixed.

Maintain the same data split, architecture, optimizer settings, epoch budget, and checkpoint-selection rule across SNN training conditions.

Train each window length separately. A network trained for 32 steps and evaluated after eight steps answers a different question from a network trained for eight steps. Treat truncation as a separate diagnostic.

Give the ANN a reasonable, documented validation-tuning budget. Equal parameter count is useful, but does not make the models equivalent in temporal computation or representational behavior.

**Experiment matrix and selection**

The core SNN experiment is:

\[
3\text{ windows}\times3\text{ penalties}\times5\text{ seeds}
=45\text{ training runs}.
\]

Add five ANN runs. Threshold adjustment uses existing checkpoints and requires no additional training.

Before the main runs:

- Designate **16 steps, moderate penalty versus no penalty** as the primary contrast.
- Freeze the accuracy tolerance and candidate settings.
- Define checkpoint selection consistently, such as highest validation accuracy with earliest epoch breaking ties.
- Specify that all grid results will be reported, including failures.

For the practical comparison between strategies, select settings using validation data: choose the lowest-activity candidate satisfying the validation accuracy budget. Lock those choices before testing.

Do not choose the best-looking test configuration and then describe it as a planned finding.

**Metrics and hardware-related evaluation**

Record these separately:

| Metric | Meaning |
|---|---|
| Accuracy and macro-F1 | Overall and class-balanced predictive performance. |
| Accuracy difference | Performance change relative to the same-window baseline. |
| Input and per-layer spike counts | Events per image over the complete window. |
| Per-layer occupancy | Fraction of neuron–time positions containing a spike. |
| Effective synaptic operations | Activity weighted by outgoing connections. |
| Simulation steps | Algorithmic observation length. |
| Inactive-neuron fraction | Neurons that never fire over the specified evaluation set. |
| Output tie/all-zero rates | Whether spike-count decisions become ambiguous. |
| Parameter and state storage | Model footprint and temporal-state overhead. |
| CPU/GPU execution time | Simulator performance on the reported machine. |

Use a documented tie rule, such as deterministic `argmax`, and report tie frequency. Do not exclude all-zero-output examples.

For this fully connected network, define \(S_0,S_1,S_2\) as input and hidden-layer spike counts for one image. An idealized effective synaptic accumulation count is:

\[
\operatorname{SynOps}
=128S_0+64S_1+10S_2.
\]

This counts the number of destination synapses activated by emitted spikes, assuming dense nonzero connectivity and event-based zero skipping.

Several implications follow:

- Input events can dominate the total.
- Reducing \(S_1\) or \(S_2\) leaves the \(128S_0\) term unchanged.
- Lower occupancy can coexist with more total events if \(T\) increases.
- Output spikes still incur generation and decoding costs, even though this formula has no downstream output layer.

Also report **\(202T\) neuron updates per image** for this timestep-based model, where 202 is the number of hidden and output neurons. Keep this separate from SynOps.

The architecture has **109,386 trainable weights and biases**, approximately **0.42 MiB at float32**, before fixed neuron settings, state, and other buffers.

For the ANN, report dense MACs and, if making a sparsity-aware comparison, effective MACs accounting for zero activations. Do not assume that ANNs lack exploitable sparsity.

These are activity and storage measurements. Actual device energy additionally depends on memory access, routing, precision, neuron updates, control, and hardware mapping. NeuroBench explicitly separates algorithmic and system evaluation; the 2026 analytical energy study reinforces the importance of data movement. [NeuroBench](https://www.nature.com/articles/s41467-025-56739-4), [Yan et al., version 6](https://arxiv.org/abs/2409.08290v6)

A defensible conclusion would concern **reduced effective operations under stated assumptions**. Measured energy savings or neuromorphic latency would require an identified hardware implementation and appropriate measurements.

**Randomness and uncertainty**

Use five independently trained seeds for the main experiment.

Within each trained model:

- Evaluate three independently generated input-spike realizations.
- Share the evaluation realizations across methods for paired comparisons.
- For different windows, use prefixes of the same generated input sequence where practical.
- Keep validation and test realization banks separate.

Average realization-level measurements within each training seed. Then analyze the five seed-level results. Three realizations from one trained model are not three independent training experiments.

Report:

- Every seed’s result.
- Mean and standard deviation.
- Paired accuracy differences.
- Paired relative operation reductions.
- A confidence interval for the primary paired differences, with its assumptions stated.

Five seeds provide limited precision. These intervals principally describe training variability conditional on your data split and test set.

For the one-percentage-point tolerance, a favorable average alone is insufficient to establish negligible accuracy loss. If the uncertainty interval extends beyond the tolerance, describe the evidence as inconclusive at that margin.

Keep other grid comparisons descriptive unless you plan an appropriate multiple-comparison analysis.

**Ablations and sensitivity checks**

The required ablations are already embedded in the design:

1. Remove the penalty: \(\lambda=0\).
2. Compare learned suppression with threshold adjustment.
3. Vary training/inference window together.
4. Compare hidden activity with whole-network SynOps.
5. Inspect inactivity and output ambiguity.

If the core experiment is complete, choose **one** additional extension:

| Extension | Purpose |
|---|---|
| Repeat baseline and selected regularized setting on MNIST | Tests limited dataset transfer. |
| Evaluate selected models under predefined Gaussian noise | Tests whether the trade-off survives a simple input change. |
| Repeat selected comparisons at \(\beta=0.8\) and 0.95 | Tests dependence on the chosen membrane retention. |

A full factorial experiment over encoding, leak, threshold, noise, window, and penalty would overwhelm a first project and a short manuscript.

**Compute requirements**

The small network can run on a laptop CPU. The total grid, rather than model size, is the main expense.

Measure one representative 16-step run before committing to the full matrix. If its training time is \(R_{16}\), a rough estimate under approximately linear scaling with window length is:

\[
\text{SNN training time}
\approx15R_{16}\left(\frac8{16}+1+\frac{32}{16}\right)
=52.5R_{16}.
\]

For illustration, \(R_{16}=30\) minutes implies roughly 26 hours of SNN training; \(R_{16}=2\) hours implies roughly 105 hours. These are planning examples, not benchmark measurements.

Train one condition at a time and generate spikes per batch. An 8 GB RAM laptop is a reasonable starting target for this small setup, subject to actual memory profiling.

If necessary, reduce to **two window lengths before the main study**, preserving five seeds and the comparative controls. That produces 30 SNN runs. Optional cloud GPU use can shorten training, but the project should remain viable without it.

**Reproducibility**

Use PyTorch, torchvision, snnTorch, NumPy, pandas, SciPy, scikit-learn, and Matplotlib. Keep one SNN framework throughout the main study.

Your repository should contain:

- An environment lock recording exact installed versions.
- Saved dataset split indices and dataset provenance.
- Configuration files for every condition.
- Separate training and evaluation entry points.
- Checkpoints, logs, and raw per-seed measurements.
- Figure-generation scripts.
- Definitions of every activity metric.
- A README explaining how to reproduce the principal comparison.

Record initialization, data-shuffling, training-encoding, and evaluation seeds separately. Document available deterministic settings and any remaining nondeterminism.

Verify the measurement code with a tiny hand-checkable network. Confirm that the spike penalty produces gradients, neuron states reset correctly, and event counts match known inputs. Preserve failed runs and explain exclusion rules in advance.

**7. A paper structure that fits SURJ**

Aim for approximately **2,900 words of main text**, leaving room for captions and references.

| Section | Approximate words | Purpose |
|---|---:|---|
| Abstract | 200 | Question, approach, actual principal findings, bounded implication. |
| Introduction | 350 | Motivate efficient AI and state the comparison. |
| Background and related work | 350 | Explain LIF neurons, spikes, and established sparsity methods. |
| Research question and methodology | 250 | Define hypotheses, accuracy tolerance, and comparison logic. |
| Experimental setup | 400 | Data, models, training, seeds, selection, and metric definitions. |
| Results | 650 | Present observed trade-offs and uncertainty. |
| Discussion and implications | 400 | Explain mechanisms and hardware relevance. |
| Limitations | 200 | Bound the evidence and identify omitted costs. |
| Conclusion | 100 | State the supported answer. |
| References | Separate budget | Use the journal’s numbered style. |

The exact counting treatment of captions and references should be checked when preparing the submission; the allocation above is a planning choice.

Use six essential figures/tables:

1. **Conceptual figure:** one LIF neuron and the two sparsification approaches.
2. **Setup table:** architecture, data split, training conditions, and seeds.
3. **Accuracy–SynOps plot:** points with seed variation, colored by method.
4. **Window/penalty figure:** accuracy and activity shown in aligned panels.
5. **Layer-cost figure:** input, hidden, and readout contributions.
6. **Comparison table:** validation-selected regularization and threshold settings, accuracy differences, operations, and uncertainty.

A **Pareto frontier** is useful here: configurations for which another measured configuration does not simultaneously achieve higher accuracy and lower activity.

To maintain accessibility:

- Explain the bucket analogy before the neuron equation.
- Define “spike,” “timestep,” and “synaptic operation” at first use.
- Include a worked counting example.
- Distinguish absolute spike count from normalized occupancy.
- Put extensive configuration details and reusable code in the repository.
- Keep the introduction focused on the experiment’s motivation; avoid trying to survey the entire field.

**8. A realistic six-week roadmap**

Assume roughly **10–15 hours of active work per week**, with longer experiments running unattended.

| Week | Work | Milestone and deliverable |
|---|---|---|
| **1** | Learn encoding, LIF dynamics, and surrogate gradients; read the core sources; set up the environment. | Explain the neuron model in your own words; produce a working small SNN and a source-summary table. |
| **2** | Implement ANN/SNN baselines, spike logging, operation counting, and the minimum experiment. Run validation-only pilots. | Audited measurement pipeline; baseline learning curves; frozen main protocol and runtime estimate. |
| **3** | Run the main training grid with five seeds; retain checkpoints and failures. | Complete training logs and condition inventory. |
| **4** | Finish evaluation and threshold comparisons; quantify uncertainty; conduct one optional extension if feasible. | Final raw results and analysis tables. |
| **5** | Produce figures; draft methods, results, discussion, and limitations; then write the introduction. | Complete first manuscript with traceable claims. |
| **6** | Reproduce a principal comparison from a clean environment; check citations, figures, anonymization, and word count. | Reproducible repository and submission-ready manuscript. |

For a four-week version, omit the optional extension, reduce window lengths before freezing the study, and begin writing methods during experimentation. Preserve the primary controls and uncertainty reporting.

**9. The minimum viable experiment to do first**

Before the full matrix:

1. Use **5,000 MNIST training images** and **1,000 separate validation images** drawn from its training partition.
2. Use the proposed small network and **16 steps**.
3. Train two conditions for three epochs with one seed: \(\lambda=0\) and \(\lambda=1\).
4. Evaluate using identical input-spike realizations.
5. Apply the three threshold multipliers to the unregularized model.
6. Measure validation accuracy, all layer spike counts, SynOps, and output ambiguity.
7. Make one accuracy–activity plot and one layer-cost breakdown.

This is a pipeline check, not sufficient publication evidence.

The first milestone is demonstrating that:

- The baseline learns.
- The penalty reaches the intended spikes through the gradient.
- Counts include input processing.
- Threshold changes are applied consistently.
- Repeated evaluation is reproducible.

**If the hypothesis is unsupported**

Do not keep altering the study until regularization appears superior.

Possible outcomes still support useful analysis:

| Outcome | What to investigate |
|---|---|
| Regularization reduces spikes but harms accuracy beyond the tolerance | Whether the task has little redundancy at that window, or whether the chosen penalty prevents adequate training. |
| Threshold adjustment matches regularization | Whether retraining provides sufficient benefit to justify its additional cost. |
| Hidden spikes decrease substantially but total SynOps barely changes | Whether fixed input processing dominates the workload. |
| The effect appears only at longer windows | Whether activity suppression requires more time to accumulate evidence. |
| Results vary strongly across seeds | Whether the effect is too unstable for a confident practical recommendation. |
| Neither approach helps | Whether the baseline is already sparse, the penalty has little influence, or the proposed comparison has limited practical value. |

Separate implementation faults from scientific outcomes using the predefined checks. Any later experiments motivated by observed test results should be labeled exploratory.

Your immediate next step is to complete the three snnTorch tutorials and the minimum experiment. The full project becomes worth pursuing when that pilot demonstrates a functioning comparison, trustworthy measurements, and a manageable runtime—not when it produces a favorable result.
