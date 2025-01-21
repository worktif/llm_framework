# Documentation for LLM Framework Experimental Section

## Overview
This project provides an LLM framework experimental section for analyzing and optimizing task prioritization and contextual dependencies using the novel mathematical formulation of the **L-function**. This system is designed to enhance performance in Multi-Agent Systems (MAS), emphasizing resource allocation, historical context, and task-specific deviations.

## Prerequisites
1. Python 3.11 or higher.
2. Required Python packages (install using `pip install -r requirements.txt`):
   - `matplotlib`
   - `numpy`
   - `pandas`
   - `sentence_transformers`
   - ...

# Project Structure

This section outlines the directory structure of the project, providing an overview of the organization and purpose of each folder and file.

## Root Directory
The root directory contains the main source folder (`src`) where all experiments and utility scripts are located.

### `src/` Directory
The `src/` folder is the main working directory, structured as follows:

src/
├── experiments/
│   ├── .data/
│   ├── a.task_priority/
│   │   ├── experimental_data/
│   │   ├── a.task_priority.histograms.py
│   │   ├── b.task_priority.histograms.analysis.py
│   │   ├── c.task_priority.heatmaps.py
│   ├── b.deviation_task/
│   │   ├── experimental_data/
│   │   ├── a.deviation_task.synth.py
│   │   ├── b.deviation_task.experiment.empirical.py
│   ├── c.deviation_historical/
│   │   ├── experimental_data/
│   │   ├── a.deviation_historical.model.by_agent.py
│   │   ├── b.deviation_historical.analysis.py
│   ├── d.llm_formula/
│   │   ├── experimental_data/
│   │   ├── a.llm_formula.empirical.py
├── utils/
│   ├── files.py
│   ├── objects.py
│   ├── strings.py

---

### `experiments/` Directory
This directory contains all experimental modules grouped by specific experiment types. Each subdirectory contains scripts and `experimental_data` folders for data storage.

#### `.data/`
- Holds initial datasets required for experiments.

#### `a.task_priority/`
- Focused on task prioritization experiments and visualization.
- **Subdirectories:**
  - `experimental_data/`: Stores priority-related data by experiments.
- **Key Scripts:**
  - `a.task_priority.histograms.py`: Generates histograms for priority coefficients (`\alpha`, `\beta`, `\gamma`).
  - `b.task_priority.histograms.analysis.py`: Analyzes generated histogram data.
  - `c.task_priority.heatmaps.py`: Creates heatmaps to visualize priority coefficient relationships.

#### `b.deviation_task/`
- Handles experiments related to task-specific deviations.
- **Subdirectories:**
  - `experimental_data/`: Stores task-related data by experiments.
- **Key Scripts:**
  - `a.deviation_task.synth.py`: Generates synthetic data for task-specific experiments.
  - `b.deviation_task.experiment.empirical.py`: Executes empirical validation experiments for task deviations.

#### `c.deviation_historical/`
- Focused on analyzing historical context deviations.
- **Subdirectories:**
  - `experimental_data/`: Stores data by deviation historical experiments.
- **Key Scripts:**
  - `a.deviation_historical.model.by_agent.py`: Simulates historical deviation models for agents.
  - `b.deviation_historical.analysis.py`: Performs data analysis for historical context deviations.

#### `d.llm_formula/`
- Dedicated to the experimental validation of the **L-formula**.
- **Subdirectories:**
  - `experimental_data/`: Stores empirical datasets resulted by experiments.
- **Key Scripts:**
  - `a.llm_formula.empirical.py`: Runs empirical analysis for L-formula experiments.

---

### `utils/` Directory
Utility scripts for shared functionality across experiments.
- `files.py`: Manages file-related operations.
- `objects.py`: Defines reusable object structures and methods.
- `strings.py`: Handles string manipulations and formatting.

---

## Notes on Directory Organization
- **Alphabetical Order**: File and directory names are alphabetically prefixed (e.g., `a.`, `b.`, `c.`) to define the order of execution, ensuring consistency across operating systems.
- **Experimental Data**: Each experiment has a dedicated `experimental_data` folder to store datasets used during execution.

---

This structure ensures modularity, clarity, and ease of navigation for contributors to the project.

Let me know if you need additional refinements!

## Getting Started
### Task Prioritization with `lambda`
1. **Generate Priority Histograms**:
   File: `a.task_priority.histograms.py`
   - Generates histograms and initial datasets for priority coefficients `\alpha`, `\beta`, `\gamma`.
   - Default values: `[0.01, 0.2, 0.5, 0.9, 1]`.
   - Run:
     ```bash
     python src/experiments/a.task_priority/a.task_priority.histograms.py
     ```

2. **Analyze Histogram Data**:
   File: `histogram.analysis.prod.py`
   - Processes histogram datasets to produce heatmaps of dependencies.
   - Run:
     ```bash
     python src/experiments/a.task_priority/b.task_priority.histograms.analysis.py
     ```

3. **Create Priority Heatmaps**:
   File: `priorities.heatmap.prod.py`
   - Generates heatmaps visualizing the relationship between priority coefficients.
   - Run:
     ```bash
     python src/experiments/a.task_priority/c.task_priority.heatmaps.py
     ```

### Task-Specific Deviation Experiments
1. **Independent Task Experiment**:
   - Generate random values for experimental conditions to test the formula independently.
   - Output includes deviation metrics and alignment values.
   - Run:
     ```bash
     python src/experiments/b.deviation_task/a.deviation_task.synth.py
     ```
     
2. **Synchronized Task Experiment**:
   - `lambda` values are defined as sinusoidal dependencies, with model outputs aligned to cosine patterns for maximal consistency.
   - Run:
     ```bash
     python src/experiments/b.deviation_task/b.deviation_task.experiment.empirical.py
     ```

### Historical Context Deviation Experiments
1. **Progressive Experiment with Fibonacci Sequences**:
   - Evaluate deviations using context windows based on Fibonacci numbers for argumentation validity.
   - Run:
     ```bash
     python src/experiments/c.deviation_historical/a.deviation_historical.model.by_agent.py
     ```
2. **Mean Historical Context Deviation Experiment by Window Size**:
   - Evaluate minimum, maximum and average deviations using context windows based on Fibonacci numbers for argumentation validity.
   - Run:
     ```bash
     python src/experiments/c.deviation_historical/b.deviation_historical.analysis.py
     ```

### L-Formula Optimization Experiments
1. **Experimental Parameter Tuning**:
   - Optimize `L` values by defining dynamic adjustment windows for key parameters.
   - The L function experiments involve:
     - Managing the optimization window to determine if the current value is optimal.
     - Testing the formula’s minimums and scalability across the system's value ranges.
     - Theoretical principles evidence for balancing response brevity, task-specific alignment, and contextual relevance are directly applied here.
   - Run:
     ```bash
     python src/experiments/d.llm_formula/a.llm_formula.empirical.py
     ```

## Appendix

### Notes
- File names are alphabetically ordered to ensure consistent execution across different operating systems.
- The order of execution is critical. Follow the alphabetical file names to ensure proper dependency resolution.
- Experimental datasets are stored in the `src/experiments/.data` directory. These include base values for testing scenarios and parameter ranges.

### Contribution
Contributions are welcome! Please submit a pull request with a detailed description of your changes.

---

## License
This project is open-source and available under the [Apache License 2.0](LICENSE).