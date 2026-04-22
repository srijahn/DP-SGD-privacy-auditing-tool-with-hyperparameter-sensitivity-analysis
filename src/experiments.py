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
        "svd_scale",
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
    requested = Path(config_path)
    candidates = [requested]

    # Allow shorthand like "sweep.yaml" by also checking under configs/.
    if not requested.is_absolute():
        candidates.append(Path("configs") / requested)

    resolved = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            resolved = candidate
            break

    if resolved is None:
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Config file not found. Tried: {searched}. "
            "Use --config configs/sweep.yaml or provide an absolute path."
        )

    with open(resolved, "r", encoding="utf-8") as f:
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
    """Compute sensitivity of hyperparameters against key privacy and utility metrics."""
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
        "attack_advantage",
        "utility_drop",
        "n_trials",
        "n_estimation_trials",
        "debug",
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    hyperparam_cols = [col for col in numeric_cols if col not in result_cols]
    
    sensitivities = []
    for col in hyperparam_cols:
        valid_gap = ~(df[col].isna() | df["gap_ratio"].isna())
        valid_emp = ~(df[col].isna() | df["epsilon_empirical_lb"].isna())
        valid_utility = ~(df[col].isna() | df["clean_acc_mean"].isna())

        if valid_gap.sum() > 1 and valid_emp.sum() > 1 and valid_utility.sum() > 1:
            corr_gap = df.loc[valid_gap, col].corr(df.loc[valid_gap, "gap_ratio"])
            corr_emp = df.loc[valid_emp, col].corr(df.loc[valid_emp, "epsilon_empirical_lb"])
            corr_util = df.loc[valid_utility, col].corr(df.loc[valid_utility, "clean_acc_mean"])
            variance_contribution = df.loc[valid_gap, col].std() if df.loc[valid_gap, col].std() > 0 else 0
            sensitivities.append(
                {
                    "hyperparameter": col,
                    "corr_gap_ratio": float(corr_gap) if not np.isnan(corr_gap) else 0.0,
                    "corr_empirical_epsilon": float(corr_emp) if not np.isnan(corr_emp) else 0.0,
                    "corr_clean_accuracy": float(corr_util) if not np.isnan(corr_util) else 0.0,
                    "normalized_variance": float(variance_contribution),
                }
            )
    
    sensitivity_df = pd.DataFrame(sensitivities)
    if not sensitivity_df.empty:
        sensitivity_df = sensitivity_df.sort_values("corr_gap_ratio", key=abs, ascending=False)
    return sensitivity_df


def _method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "poison_method" not in df.columns:
        return pd.DataFrame()
    cols = [
        "epsilon_theoretical",
        "epsilon_empirical_lb",
        "gap_ratio",
        "attack_advantage",
        "clean_acc_mean",
        "utility_drop",
    ]
    present_cols = [c for c in cols if c in df.columns]
    return df.groupby("poison_method", as_index=False)[present_cols].mean().sort_values("epsilon_empirical_lb", ascending=False)


def _critical_findings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["privacy_utility_score"] = out["epsilon_empirical_lb"] * out["clean_acc_mean"]
    cols = [
        "poison_method",
        "model_name",
        "noise_multiplier",
        "max_grad_norm",
        "epsilon_theoretical",
        "epsilon_empirical_lb",
        "gap_ratio",
        "attack_advantage",
        "clean_acc_mean",
        "utility_drop",
        "privacy_utility_score",
    ]
    cols = [c for c in cols if c in out.columns]
    return out.sort_values(["epsilon_empirical_lb", "attack_advantage"], ascending=False)[cols]


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

    if "poison_method" in df.columns and df["poison_method"].nunique() > 1:
        plt.figure(figsize=(8, 5))
        grouped = df.groupby("poison_method", as_index=False)["epsilon_empirical_lb"].mean()
        plt.bar(grouped["poison_method"], grouped["epsilon_empirical_lb"])
        plt.title("Empirical epsilon lower bound by poisoning method")
        plt.xlabel("Poisoning method")
        plt.ylabel("Empirical epsilon lower bound")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_empirical_eps_by_method.png", dpi=180)
        plt.close()

    # NEW: Compute and save hyperparameter sensitivity ranking
    sensitivity_df = _compute_sensitivity(df)
    sensitivity_csv = out_dir / "sensitivity_analysis.csv"
    sensitivity_df.to_csv(sensitivity_csv, index=False)

    method_df = _method_comparison(df)
    method_csv = out_dir / "method_comparison.csv"
    method_df.to_csv(method_csv, index=False)

    critical_df = _critical_findings(df)
    critical_csv = out_dir / "critical_findings.csv"
    critical_df.to_csv(critical_csv, index=False)

    print(f"\n=== Hyperparameter Sensitivity Analysis ===")
    print(sensitivity_df.to_string(index=False))
    print(f"Saved to {sensitivity_csv}")
    if not method_df.empty:
        print("\n=== Poison Method Comparison ===")
        print(method_df.to_string(index=False))
        print(f"Saved to {method_csv}")
    print(f"Saved to {critical_csv}")

    return df
