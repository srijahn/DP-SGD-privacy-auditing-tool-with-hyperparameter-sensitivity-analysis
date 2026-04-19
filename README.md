# Ethics of AI Project: Auditing DP-SGD Privacy Leakage

This repository gives you a runnable baseline for your project proposal:

- Train DP-SGD models on binary Fashion-MNIST.
- Run poisoning-based privacy auditing.
- Compare theoretical privacy (`epsilon_theoretical`) and empirical lower bound (`epsilon_empirical_lb`).
- Generate CSV + figures for your report.

## 1) Project novelty claim you can use

This project extends Jagielski et al. (NeurIPS 2020) with a **systematic hyperparameter sensitivity analysis** and a **practical leakage-aware tuning recipe**.

Research question:

> Which DP-SGD hyperparameters most strongly influence empirical privacy leakage at similar utility, and how should we tune them to reduce leakage?

## 2) Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_experiments.py --config configs/sweep.yaml --output-dir results
```

Fast smoke test (fewer configs):

```powershell
python run_experiments.py --config configs/sweep.yaml --output-dir results --max-configs 2
```

## 3) Outputs

After a run, check:

- `results/summary.csv`
- `results/plot_eps_emp_vs_noise.png`
- `results/plot_eps_emp_vs_clip.png`
- `results/plot_gap_vs_noise.png`

## 4) What each metric means

- `epsilon_theoretical`: DP accountant estimate from Opacus.
- `epsilon_empirical_lb`: empirical lower bound inferred by auditing (Clopper-Pearson calibrated).
- `gap_ratio = epsilon_theoretical / epsilon_empirical_lb`: larger means bigger mismatch between theory and measured leakage lower bound.
- `clean_acc_mean`: average clean test accuracy.

## 5) Required report sections (6-10 pages)

Use this exact structure to satisfy the rubric:

1. Introduction and motivation
2. Related work
3. Methodology
4. Experimental setup
5. Results
6. Discussion and limitations
7. Conclusion

## 6) Suggested experiments for novelty

1. Hyperparameter ranking:
   - Sweep noise multiplier, clipping norm, batch size, learning rate, epochs.
2. Fixed-accuracy comparison:
   - Keep accuracy in a narrow band and compare leakage trends.
3. One extra contribution:
   - Initialization randomness study OR improved trigger strategy.

## 7) Team contribution checklist (add names)

- Member A: DP training pipeline and accountant integration
- Member B: Auditing implementation and confidence intervals
- Member C: Hyperparameter sweep automation and plotting
- Member D: Report writing, related work, and result analysis

Replace with your actual names and exact contributions before submission.

## 8) Limitations to mention honestly

- Binary Fashion-MNIST is a simplified setting.
- Backdoor-style auditing gives lower bounds, not full privacy characterization.
- Runtime limits reduce number of trials and confidence tightness.

## 9) Minimum submission package

- Clean code repository with this README
- `results/summary.csv` + plots
- Final PDF report (NeurIPS/ICML style)
- Short section: "Implemented by our group"
