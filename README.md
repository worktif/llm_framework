# Experimental Validation of the L-Function for Task Optimization in Multi-Agent LLM Systems

## Overview
This documentation outlines the experimental validation of the L-function, a novel mathematical construct designed to optimize task prioritization, contextual coherence, and resource management within Large Language Model (LLM)-driven Multi-Agent Systems (MAS). The experimental section systematically evaluates the operational behavior of the L-function under various simulated and empirical conditions, focusing on the interplay between historical context alignment, task-specific response optimization, and system adaptability.

By applying the L-function to diverse agent scenarios, the experiments aim to validate its practical relevance, robustness, and dynamic responsiveness to resource constraints, task criticality, and semantic deviation. This work bridges theoretical modeling with empirical evidence to assess the L-function’s role in enhancing MAS efficiency, coherence, and decision-making precision.

---

## Prerequisites
1. Python 3.11 or higher.
2. Required Python packages (install using `pip install -r requirements.txt`):
   - `matplotlib`
   - `numpy`
   - `pandas`
   - `sentence_transformers`
   - ...

# Project Structure


This section presents a structured overview of the project directory layout, detailing the purpose and functional role of each component in the context of experimental validation of the L-function within Multi-Agent Systems (MAS).

### Root Directory

The root directory serves as the entry point of the project and contains the primary source directory (src/) that encapsulates all experimental modules, computational utilities, and supporting scripts required for empirical evaluation.

### src/ Directory

The src/ directory constitutes the core of the experimental framework. It is systematically organized to support reproducible experimentation and modular analysis. The folder includes:

```
src/
├── experiments/
│   ├── .data/
│   ├── a.task_priority/
│   │   ├── experimental_data/
│   │   ├── a.a.task_priority.histograms.py
│   │   ├── a.b.task_priority.histograms.analysis.py
│   │   ├── a.c.task_priority.heatmaps.py
│   ├── b.deviation_task/
│   │   ├── experimental_data/
│   │   ├── b.a.deviation_task.synth.py
│   │   ├── b.b.deviation_task.experiment.empirical.py
│   ├── c.deviation_historical/
│   │   ├── experimental_data/
│   │   ├── c.a.deviation_historical.model.by_agent.py
│   │   ├── c.b.deviation_historical.analysis.py
│   ├── d.llm_formula/
│   │   ├── experimental_data/
│   │   ├── d.a.llm_formula.empirical.py
├── utils/
│   ├── files.py
│   ├── objects.py
│   ├── strings.py
```
---

### experiments/ Directory

This directory encapsulates all experimental modules, systematically organized by experimental focus areas. Each subdirectory is dedicated to a specific class of experiments and contains corresponding execution scripts along with an experimental_data/ folder for localized data storage and output archiving.

#### .data/
- Contains foundational datasets required for initializing and reproducing experimental runs across various modules.

#### a.task_priority/
- Dedicated to experiments investigating task prioritization mechanisms and their impact on the L-function parameters.
- **Subdirectories**:
    - `experimental_data/`: Stores task-priority-specific empirical data, including computed priority weights and raw observations.
- **Key Scripts**:
    - `a.a.task_priority.histograms.py`: Generates histogram visualizations of task prioritization coefficients (α, β, γ) under varying load and task importance levels.
    - `a.b.task_priority.histograms.analysis.py`: Performs statistical analysis on histogram outputs to identify distributional patterns and anomalies in prioritization behavior.
    - `a.c.task_priority.heatmaps.py`: Constructs two-dimensional heatmaps depicting the relationships and interactions among the prioritization coefficients.

#### b.deviation_task/
- Addresses empirical validation of task-specific deviation metrics as defined in the L-function formulation.
- **Subdirectories:**
    - `experimental_data/`: Archives generated and observed datasets for deviation computation in task-specific contexts.
- **Key Scripts:**
    - `b.a.deviation_task.synth.py`: Produces synthetic experimental datasets to simulate task-based deviation dynamics and validate DT(O, T).
    - `b.b.deviation_task.experiment.empirical.py`: Conducts controlled empirical experiments to assess task deviation impact and optimize the corresponding λ-weighted term.

#### c.deviation_historical/
- Focused on modeling and analyzing historical context deviation, particularly evaluating the DH(O, H) component within agent communication.
- **Subdirectories:**
    - `experimental_data/`: Retains structured datasets produced during historical deviation simulations.
- **Key Scripts:**
    - `c.a.deviation_historical.model.by_agent.py`: Simulates multiple agent scenarios to quantify historical context misalignment and deviation propagation.
    - `c.b.deviation_historical.analysis.py`: Executes post-hoc analysis on deviation distributions, including cosine similarity computations and error profiling.

#### d.llm_formula/
- Reserved for the empirical validation of the complete L-function, integrating both task-specific and contextual deviation components under resource-aware conditions.
- **Subdirectories:**
    - `experimental_data/`: Contains high-level evaluation datasets capturing full-system behavior under experimental constraints.
- **Key Scripts:**
    - `d.a.llm_formula.empirical.py`: Executes full-stack experimental procedures validating the theoretical formulation of the L-function in live MAS simulations.

#### utils/ Directory

This directory provides a suite of shared utility scripts designed to support experimental reproducibility, code modularity, and data pipeline stability.
- `files.py`: Implements I/O management utilities for structured file handling, dataset loading, and experiment logging.
- `objects.py`: Defines reusable object-oriented abstractions, including agent wrappers, vector containers, and lambda-weight structures.
- `strings.py`: Provides standardized string operations for data formatting, annotation parsing, and human-readable output generation.

---


This structure ensures modularity, clarity, and ease of navigation for all contributors engaging with the experimental validation framework.

## Getting Started

### Task Prioritization with lambda
1. **Generate Priority Histograms**
   File: `a.a.task_priority.histograms.py`
- Generates histograms and initializes datasets for the priority coefficients `\alpha`, `\beta`, and `\gamma`.
- Default coefficient values: `[0.01, 0.2, 0.5, 0.9, 1]`
- Execution:
    ```shell
      python src/experiments/a.task_priority/a.a.task_priority.histograms.py
    ```

2. **Analyze Histogram Data**
   File: `a.b.task_priority.histograms.analysis.py`
- Processes the histogram datasets to construct heatmaps reflecting inter-coefficient dependencies.
- Execution:
    ```shell
      python src/experiments/a.task_priority/a.b.task_priority.histograms.analysis.py
    ```


3. **Create Priority Heatmaps**
   File: `a.c.task_priority.heatmaps.py`
- Generates visual heatmaps illustrating the relationships among the priority coefficients.
- Execution:
    ```shell
      python src/experiments/a.task_priority/a.c.task_priority.heatmaps.py
    ```


### Task-Specific Deviation Experiments
1.	**Independent Task Experiment**
      File: `b.a.deviation_task.synth.py`
- Generates randomized values for controlled experimental evaluation of the L-function in isolation.
- Outputs include task-specific deviation metrics and alignment scores.
- Execution:
  ```shell
      python src/experiments/b.deviation_task/b.a.deviation_task.synth.py
    ```


2. **Synchronized Task Experiment**
   File: `b.b.deviation_task.experiment.empirical.py`
- Executes empirical evaluation where lambda values follow sinusoidal functions, and model output vectors align with cosine-based ground truth for maximal consistency.
- Execution:
  ```shell
      python src/experiments/b.deviation_task/b.b.deviation_task.experiment.empirical.py
    ```


### Historical Context Deviation Experiments
1. **Progressive Experiment with Fibonacci Sequences**
   File: `c.a.deviation_historical.model.by_agent.py`
- Evaluates historical context deviation across expanding context windows, incremented by Fibonacci sequence steps to ensure gradational argumentation validity.
- Execution:
  ```shell
      python src/experiments/c.deviation_historical/c.a.deviation_historical.model.by_agent.py
    ```

2. **Mean Historical Context Deviation by Window Size**
   File: `c.b.deviation_historical.analysis.py`
- Computes statistical summaries (min, max, mean) of historical deviations across varying window sizes based on Fibonacci increments.
- Execution:
  ```shell
      python src/experiments/c.deviation_historical/c.b.deviation_historical.analysis.py
    ```

### L-Formula Optimization Experiments
1. **Experimental Parameter Tuning**
   File: `d.a.llm_formula.empirical.py`
- Executes parameter optimization procedures for the **L-function** by dynamically adjusting evaluation windows for core coefficients.
- The **L-function** validation workflow includes:
    - Managing adaptive optimization intervals to determine whether current values satisfy local or global minima criteria.
    - Assessing the scalability and robustness of the formula across a broad spectrum of system-defined value distributions.
    - Direct application of theoretical foundations supporting trade-offs between output brevity, task-aligned deviation, and historical-contextual fidelity.
- Execution:
  ```shell
      python src/experiments/d.llm_formula/d.a.llm_formula.empirical.py
    ```

---

## Appendix

### Notes
- All script filenames are strictly alphabetically ordered to ensure deterministic execution across heterogeneous operating systems.
- Execution order is critical. Please adhere to the alphabetical script sequence to preserve dependency integrity and result reproducibility.
- All experimental datasets are located within the src/experiments/.data directory. These datasets include reference configurations, default parameter boundaries, and control scenarios for validation.

---

### Contribution

Contributions to this experimental framework are welcome and encouraged. To propose changes or enhancements, please submit a pull request accompanied by a comprehensive and technically detailed description of your modifications.

---

## License

This project is distributed under the terms of the [Apache License 2.0](LICENSE), ensuring open access, collaborative development, and reproducible scientific validation.