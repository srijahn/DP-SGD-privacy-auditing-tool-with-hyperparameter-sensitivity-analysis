# DP-SGD Privacy Auditing Tool

This repository contains a group-built tool for auditing privacy leakage in differentially private training.
The tool compares theoretical privacy guarantees against empirical leakage estimated via poisoning-based audits.

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
- Creates clean and poisoned training datasets.
- Computes theoretical privacy epsilon from the DP accountant.
- Computes empirical epsilon lower bounds from audit outcomes.
- Runs hyperparameter sweeps and reports sensitivity trends.
- Generates report-ready CSV files and plots.

### Main run command

```powershell
python run_experiments.py --config configs/sweep.yaml --output-dir results
```

### Quick test command

```powershell
python run_experiments.py --config configs/sweep.yaml --output-dir results --max-configs 2
```

### Important output files

- results/summary.csv
- results/sensitivity_analysis.csv
- results/plot_eps_emp_vs_noise.png
- results/plot_eps_emp_vs_clip.png
- results/plot_gap_vs_noise.png
- results/plot_theo_vs_empirical_noise.png
- results/plot_theo_vs_empirical_clip.png

### Metric definitions

- epsilon_theoretical: Privacy estimate from Opacus accountant.
- epsilon_empirical_lb: Empirical lower bound from auditing and confidence intervals.
- gap_ratio: epsilon_theoretical divided by epsilon_empirical_lb when empirical bound is non-zero.
- clean_acc_mean: Average clean test accuracy.

## Clear Indication of What Was Implemented by the Group

The following components were implemented by our group in this repository:

1. DP-SGD training pipeline and epsilon extraction (src/train_dp.py).
2. Dataset preparation, trigger insertion, and poisoning utilities (src/data.py).
3. Logistic regression and MLP model definitions (src/model.py).
4. Privacy audit logic with Clopper-Pearson calibrated empirical bounds (src/audit.py).
5. Experiment orchestration, hyperparameter sweeps, plotting, and sensitivity analysis (src/experiments.py).
6. Command-line runner for complete experiments (run_experiments.py).
7. Configuration-driven experiment control (configs/sweep.yaml).

## Reproducibility Notes

- Random seeds are set from the config file.
- Results may vary slightly across machines and PyTorch/Opacus versions.
- Increase num_trials in configs/sweep.yaml for stronger statistical confidence.
