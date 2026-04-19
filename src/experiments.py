from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.audit import run_audit
from src.data import load_binary_fashion_mnist


def _normalize_types(cfg: Dict) -> Dict:
    int_fields = {
        "seed",
        "num_trials",
        "poisoning_k",
        "batch_size",
        "epochs",
        "trigger_size",
        "target_label",
    }
    float_fields = {
        "alpha",
        "delta",
        "learning_rate",
        "weight_decay",
        "noise_multiplier",
        "max_grad_norm",
    }

    base = cfg.get("base", {})
    for key in int_fields:
        if key in base:
            base[key] = int(base[key])
    for key in float_fields:
        if key in base:
            base[key] = float(base[key])

    dataset_cfg = cfg.get("dataset", {})
    for key in ("max_train_per_class", "max_test_per_class"):
        if key in dataset_cfg:
            dataset_cfg[key] = int(dataset_cfg[key])

    sweep = cfg.get("sweep", {})
    for key, values in sweep.items():
        if key in int_fields:
            sweep[key] = [int(v) for v in values]
        elif key in float_fields:
            sweep[key] = [float(v) for v in values]

    cfg["base"] = base
    cfg["dataset"] = dataset_cfg
    cfg["sweep"] = sweep
    return cfg


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _normalize_types(cfg)


def _grid_from_sweep(sweep: Dict[str, List]) -> Iterable[Dict]:
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _plot_line(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> None:
    if df.empty:
        return
    grp = df.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)
    plt.figure(figsize=(6, 4))
    plt.plot(grp[x_col], grp[y_col], marker="o")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_theo_vs_empirical(df: pd.DataFrame, x_col: str, out_path: Path) -> None:
    """Plot theoretical and empirical epsilon on the same graph to show the gap."""
    if df.empty:
        return
    grp = df.groupby(x_col, as_index=False)[["epsilon_theoretical", "epsilon_empirical_lb"]].mean().sort_values(x_col)
    
    plt.figure(figsize=(8, 5))
    plt.plot(grp[x_col], grp["epsilon_theoretical"], marker="o", label="Theoretical ε", linewidth=2)
    plt.plot(grp[x_col], grp["epsilon_empirical_lb"], marker="s", label="Empirical ε (lower bound)", linewidth=2)
    plt.title(f"Theoretical vs Empirical Privacy: {x_col.replace('_', ' ').title()}")
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Privacy Parameter ε")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _compute_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sensitivity analysis: correlation of each numeric hyperparameter with gap_ratio."""
    if df.empty:
        return pd.DataFrame()
    
    # Identify numeric hyperparameter columns (exclude results columns)
    result_cols = {
        "epsilon_theoretical",
        "epsilon_empirical_lb",
        "gap_ratio",
        "threshold",
        "clean_trigger_mean",
        "poison_trigger_mean",
        "clean_acc_mean",
        "poison_acc_mean",
        "n_trials",
        "n_estimation_trials",
        "debug",
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    hyperparam_cols = [col for col in numeric_cols if col not in result_cols]
    
    sensitivities = []
    for col in hyperparam_cols:
        # Remove NaN values for correlation calculation
        valid_idx = ~(df[col].isna() | df["gap_ratio"].isna())
        if valid_idx.sum() > 1:
            corr = df.loc[valid_idx, col].corr(df.loc[valid_idx, "gap_ratio"])
            variance_contribution = df.loc[valid_idx, col].std() if df.loc[valid_idx, col].std() > 0 else 0
            sensitivities.append({
                "hyperparameter": col,
                "correlation_with_gap_ratio": float(corr) if not np.isnan(corr) else 0.0,
                "normalized_variance": float(variance_contribution),
            })
    
    sensitivity_df = pd.DataFrame(sensitivities)
    if not sensitivity_df.empty:
        sensitivity_df = sensitivity_df.sort_values("correlation_with_gap_ratio", key=abs, ascending=False)
    return sensitivity_df


def run_all(config_path: str, output_dir: str, max_configs: int | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_binary_fashion_mnist(
        data_dir=cfg["dataset"]["data_dir"],
        max_train_per_class=cfg["dataset"]["max_train_per_class"],
        max_test_per_class=cfg["dataset"]["max_test_per_class"],
    )

    base = cfg["base"]
    combos = list(_grid_from_sweep(cfg["sweep"]))
    if max_configs is not None:
        combos = combos[:max_configs]

    rows = []
    for combo in tqdm(combos, desc="Running configs"):
        run_cfg = {**base, **combo}
        result = run_audit(x_train, y_train, x_test, y_test, run_cfg)
        rows.append({**run_cfg, **result})

    df = pd.DataFrame(rows)
    summary_csv = out_dir / "summary.csv"
    df.to_csv(summary_csv, index=False)

    _plot_line(
        df,
        x_col="noise_multiplier",
        y_col="epsilon_empirical_lb",
        out_path=out_dir / "plot_eps_emp_vs_noise.png",
        title="Empirical epsilon LB vs noise multiplier",
    )
    _plot_line(
        df,
        x_col="max_grad_norm",
        y_col="epsilon_empirical_lb",
        out_path=out_dir / "plot_eps_emp_vs_clip.png",
        title="Empirical epsilon LB vs clipping norm",
    )
    _plot_line(
        df,
        x_col="noise_multiplier",
        y_col="gap_ratio",
        out_path=out_dir / "plot_gap_vs_noise.png",
        title="Gap ratio (eps_th / eps_emp) vs noise multiplier",
    )

    # NEW: Plot theoretical vs empirical epsilon side-by-side
    _plot_theo_vs_empirical(
        df,
        x_col="noise_multiplier",
        out_path=out_dir / "plot_theo_vs_empirical_noise.png",
    )
    _plot_theo_vs_empirical(
        df,
        x_col="max_grad_norm",
        out_path=out_dir / "plot_theo_vs_empirical_clip.png",
    )

    # NEW: Compute and save hyperparameter sensitivity ranking
    sensitivity_df = _compute_sensitivity(df)
    sensitivity_csv = out_dir / "sensitivity_analysis.csv"
    sensitivity_df.to_csv(sensitivity_csv, index=False)
    print(f"\n=== Hyperparameter Sensitivity Analysis ===")
    print(sensitivity_df.to_string(index=False))
    print(f"Saved to {sensitivity_csv}")

    return df
