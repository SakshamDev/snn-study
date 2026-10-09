# Research Planning Prompt: Beginner-Friendly Neuromorphic Computing Paper for SURJ

Act as an experienced research mentor specializing in **neuromorphic computing, spiking neural networks (SNNs), computational neuroscience, and AI hardware**.

I am an undergraduate student with a background in artificial intelligence, machine learning, and Python. I want to transition toward AI hardware and neuromorphic computing.

My goal is to publish an undergraduate-level research paper in the **Stanford Undergraduate Research Journal (SURJ)**, preferably in its Natural Sciences & Engineering section.

## 1. My research objectives

I want a paper that:

- Introduces neuromorphic computing and its significance in an accessible but technically rigorous way.
- Contains a legitimate research question, methodology, experiments, quantitative results, and analysis.
- Is beginner-friendly and feasible as my first independent neuromorphic computing research project.
- Does not depend on developing an entirely new algorithm, architecture, or theoretical contribution.
- May involve reproducing existing techniques, comparing established methods, studying trade-offs, or evaluating a simple hypothesis.
- Uses a software-only experimental setup, preferably Python, PyTorch, snnTorch, Brian2, or similar open-source frameworks.
- Can run on an ordinary laptop, with a free cloud GPU as an optional resource.
- Connects its experimental findings to the broader motivation for neuromorphic computing, especially efficient and event-driven AI hardware.

Prioritize **scientific rigor, reproducibility, clear explanations, and meaningful experiments over novelty for its own sake**.

Do not propose a pure literature review unless there is a compelling reason. I specifically want hands-on implementation and results.

## 2. First, understand SURJ

Research SURJ using its official submission guidelines, recently published papers, and any available editorial guidance.

Determine:

- What constitutes an acceptable undergraduate research contribution.
- Whether replication studies, benchmark comparisons, and research-oriented reviews are suitable.
- What level of originality and experimentation appears appropriate.
- The expected manuscript length, structure, and presentation style.
- Whether similar computational or engineering studies have appeared in SURJ.

Distinguish published journal requirements from your own interpretation. Do not assume that low novelty automatically means a paper will be accepted.

## 3. Explore the research literature

Review relevant research literature on introductory neuromorphic computing and SNNs, emphasizing work published from 2022 to October 2026, while including important foundational papers.

Explore these areas:

1. Spiking neural networks compared with traditional artificial neural networks.
2. Spike encoding methods, including rate and temporal coding.
3. Leaky integrate-and-fire neuron models and their parameters.
4. Spike sparsity, inference latency, and computational activity.
5. Robustness of SNNs to input noise.
6. ANN-to-SNN conversion, if manageable at the undergraduate level.
7. Other simple but interesting experimental questions appropriate for a first research paper.

Use verified academic sources, with links, publication dates, and a brief explanation of each relevant paper's contribution.

Identify what is already well established and what useful comparisons, replications, or educational investigations remain feasible.

Avoid presenting established results as novel discoveries.

## 4. Propose 5–7 possible research topics

For each topic, provide:

**A. Proposed paper title**

A professional title suitable for an undergraduate research journal.

**B. Beginner-friendly explanation**

Explain what the project investigates, assuming I understand ordinary neural networks but am new to neuromorphic computing.

**C. Research question**

A focused, testable question that can realistically be answered with software experiments.

**D. Hypothesis**

What do existing studies suggest might happen, and why? Clearly separate predictions from experimentally verified outcomes.

**E. Experimental methodology**

Describe the models, datasets, experiments, variables, baselines, metrics, and statistical evaluation.

**F. Implementation**

Recommend appropriate Python libraries, compute requirements, estimated practical difficulty, and potential implementation challenges.

**G. Results and figures**

Specify which graphs, tables, and visualizations we could produce. Do not invent results.

**H. Scientific contribution**

Explain what the project would contribute even without a novel algorithm. Identify whether it is primarily a replication, comparative study, sensitivity analysis, or application study.

**I. SURJ suitability**

Evaluate how well the project fits an undergraduate research journal and identify its potential weaknesses.

## 5. Compare and rank the ideas

Create a comparison table with scores from 1–10 for:

- Beginner friendliness
- Ease of implementation
- Experimental depth
- Likelihood of producing interpretable results
- Educational value
- Relevance to neuromorphic AI hardware
- Strength of potential research contribution
- Suitability for a 2,000–4,000-word SURJ paper

Explain the scoring rather than assigning arbitrary numbers.

Recommend the three most promising options and select one primary recommendation.

Do not automatically prefer the most technically advanced option.

## 6. Develop a detailed research plan for the best topic

After selecting the strongest topic, provide:

**Research foundation**
- Essential concepts I must understand
- Prerequisite mathematics and machine learning knowledge
- Important introductory papers and tutorials
- A concise explanation of the relevant neuroscience concepts

**Experimental plan**
- A primary hypothesis
- Independent and dependent variables
- Appropriate experimental controls
- Suitable datasets and preprocessing
- Model architectures
- Comparison baselines
- Evaluation metrics
- Multiple random seeds and uncertainty reporting
- Ablation and sensitivity experiments
- Expected computational requirements

**Hardware-related evaluation**

Where relevant, consider:

- Total spike count
- Spike activity per layer
- Simulation timesteps
- Inference latency
- Model size
- Approximate synaptic operation counts
- Accuracy versus spike activity

Distinguish these software-observable quantities from actual hardware energy measurements.

Do not claim that an SNN uses less energy merely because it emits fewer spikes, and do not treat ordinary CPU/GPU execution time as neuromorphic hardware latency.

**Reproducibility**
- Fixed seeds and documented environments
- Versioned dependencies
- Clear training and testing protocols
- Reusable code
- Explicit limitations and assumptions

## 7. Plan the paper itself

Propose a SURJ-appropriate paper outline:

1. Abstract
2. Introduction to neuromorphic computing
3. Background and related work
4. Research question and methodology
5. Experimental setup
6. Results
7. Discussion and implications
8. Limitations
9. Conclusion
10. References

Give approximate word allocations and suggest 4–6 essential figures or tables.

Explain how to keep the paper accessible to readers unfamiliar with SNNs while retaining adequate scientific detail.

## 8. Establish a realistic execution roadmap

Create a four-to-six-week project plan covering:

- Learning necessary concepts
- Literature review
- Environment setup
- Initial implementation
- Baseline reproduction
- Main experiments
- Results analysis
- Figures and tables
- Manuscript drafting
- Reproducibility checks

Provide clear milestones and deliverables.

Also propose a minimal viable experiment that could be completed first. Explain what to do if the initial hypothesis is not supported by the results.

## 9. Research integrity requirements

- Never fabricate citations, data, performance numbers, or experimental outcomes.
- Verify paper titles, publication years, and relevant claims.
- Separate literature-supported claims, hypotheses, and our own future results.
- Do not exaggerate energy efficiency or hardware benefits from software-only results.
- Point out methodological weaknesses and potential confounding variables.
- Prefer a simple experiment done rigorously over an ambitious experiment done poorly.
- Evaluate the risk that a proposed contribution is too trivial or has already been thoroughly established.

## 10. How to respond

Start by explaining what kind of experimental neuromorphic paper is most suitable for SURJ.

Then present the literature-informed topic shortlist and comparison table.

Finally, recommend one topic and develop its research plan in enough detail that I can begin implementation.

**Do not write a complete paper or manufacture results at this stage.**

I want to understand the research problem, choose a defensible methodology, implement the experiments myself, interpret the findings, and then write the manuscript.

Teach unfamiliar neuromorphic concepts using intuitive explanations before moving into equations and implementation details.

Challenge any weak research ideas instead of agreeing with everything I suggest.

Our ultimate objective is a **credible, accessible, software-based undergraduate research paper that introduces neuromorphic computing through meaningful experiments**.
