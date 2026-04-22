# DP-SGD Privacy Auditing Tool

This repository contains a group-built tool for auditing privacy leakage in differentially private training.
The tool compares theoretical privacy guarantees against empirical leakage estimated via poisoning-based audits.

## Novel Contribution

This project goes beyond direct paper reimplementation in three ways:

1. It adds a clipping-aware, low-variance poisoning mode (`svd_lowvar`) that uses an SVD-derived direction to create harder-to-ignore poison points.
2. It frames a critical research question: which DP-SGD hyperparameters most strongly drive the privacy gap between theoretical epsilon and empirical leakage?
3. It produces a documented auditing toolkit with method-comparison and critical-findings outputs for evaluation and reporting.

## Research Question

Primary question:

- Which combination of clipping norm, noise multiplier, and poisoning strategy creates the largest gap between claimed privacy (theoretical epsilon) and observed leakage (empirical epsilon lower bound)?

Secondary question:

- Does the low-variance SVD poisoning strategy (`svd_lowvar`) reveal stronger privacy leakage than a conventional square-trigger strategy under the same training budget?

## How This Extends the NeurIPS 2020 Paper

- Reproduction baseline: DP-SGD auditing with poisoning and Clopper-Pearson calibrated bounds.
- Extension 1: Multi-method audit framework with direct head-to-head poisoning comparison (`square` vs `svd_lowvar`).
- Extension 2: Critical analysis outputs ranking hyperparameters by their association with privacy gap and utility.
- Extension 3: Privacy-utility reporting artifacts intended for practical model audit workflows.

## Setup Instructions

### Requirements

- Python 3.10+
- Windows PowerShell (or any shell)

### Installation

1. Clone this repository and open it in a terminal.
2. Create and activate a virtual environment.
3. Install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Description of Tool and Experiment Usage

### What the tool does

- Trains DP-SGD models with Opacus on binary Fashion-MNIST.
- Creates clean and poisoned training datasets using multiple poisoning methods.
- Computes theoretical privacy epsilon from the DP accountant.
- Computes empirical epsilon lower bounds from audit outcomes.
- Runs hyperparameter sweeps and reports sensitivity trends.
- Compares poisoning methods and reports privacy-utility tradeoffs.
- Generates report-ready CSV files and plots.

### Main run command

```powershell
python run_experiments.py --config configs/sweep.yaml --output-dir results
```

### Method-comparison run (recommended for report)

```powershell
python run_experiments.py --config configs/sweep.yaml --output-dir results --max-configs 12
```

### Quick test command

```powershell
python run_experiments.py --config configs/sweep.yaml --output-dir results --max-configs 2
```

### Important output files

- results/summary.csv
- results/sensitivity_analysis.csv
- results/method_comparison.csv
- results/critical_findings.csv
- results/plot_eps_emp_vs_noise.png
- results/plot_eps_emp_vs_clip.png
- results/plot_gap_vs_noise.png
- results/plot_theo_vs_empirical_noise.png
- results/plot_theo_vs_empirical_clip.png
- results/plot_empirical_eps_by_method.png

### Metric definitions

- epsilon_theoretical: Privacy estimate from Opacus accountant.
- epsilon_empirical_lb: Empirical lower bound from auditing and confidence intervals.
- gap_ratio: epsilon_theoretical divided by epsilon_empirical_lb when empirical bound is non-zero.
- clean_acc_mean: Average clean test accuracy.
- attack_advantage: Absolute difference in trigger-detection hit rates between clean and poisoned runs.
- utility_drop: Change in clean accuracy under poisoned training.

## How to Interpret Key Results

- Stronger empirical leakage signal:
Higher `epsilon_empirical_lb` and higher `attack_advantage` indicate the audit finds stronger distinguishability.
- Larger theory-practice mismatch:
Higher `gap_ratio` indicates a larger disconnect between claimed and measured privacy.
- Better practical audit finding:
Higher `epsilon_empirical_lb` with low absolute `utility_drop` suggests privacy risk can be exposed without large utility degradation.

## Report Checklist

Use these files directly in your submission:

1. `results/summary.csv`: Full sweep outcomes per configuration.
2. `results/method_comparison.csv`: Mean performance by poisoning method (core novelty evidence).
3. `results/critical_findings.csv`: Ranked high-risk configurations for critical analysis.
4. `results/sensitivity_analysis.csv`: Hyperparameter influence summary.
5. `results/plot_empirical_eps_by_method.png`: Visual proof of method-level differences.

Recommended statement for your methodology section:

"Our work reproduces the core DP-SGD auditing protocol and extends it with a clipping-aware low-variance poisoning method, comparative multi-method evaluation, and report-oriented sensitivity diagnostics to study privacy leakage beyond theoretical guarantees."

## Clear Indication of What Was Implemented by the Group

The following components were implemented by our group in this repository:

1. DP-SGD training pipeline and epsilon extraction (src/train_dp.py).
2. Dataset preparation, trigger insertion, and poisoning utilities with two audit methods (src/data.py).
3. Logistic regression and MLP model definitions (src/model.py).
4. Privacy audit logic with Clopper-Pearson calibrated empirical bounds (src/audit.py).
5. Experiment orchestration, hyperparameter sweeps, method comparison, plotting, and sensitivity analysis (src/experiments.py).
6. Command-line runner for complete experiments (run_experiments.py).
7. Configuration-driven experiment control (configs/sweep.yaml).

## Reproducibility Notes

- Seeds are controlled through the config file.
- Increase `num_trials` for tighter confidence intervals.
- Use `--max-configs` for a quick smoke run before full sweeps.

