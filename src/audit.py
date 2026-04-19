from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import beta

from src.data import poison_dataset
from src.train_dp import train_dp_once


def cp_lower(k: int, n: int, alpha: float) -> float:
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def cp_upper(k: int, n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def eps_lb_from_counts(
    clean_hits: int,
    poison_hits: int,
    n: int,
    poisoning_k: int,
    alpha: float,
) -> Tuple[float, Dict[str, float]]:
    # We compute bounds in both directions and keep the stronger one.
    p_clean_lo = cp_lower(clean_hits, n, alpha / 2)
    p_clean_hi = cp_upper(clean_hits, n, alpha / 2)
    p_poison_lo = cp_lower(poison_hits, n, alpha / 2)
    p_poison_hi = cp_upper(poison_hits, n, alpha / 2)

    ratio_a = p_clean_lo / max(p_poison_hi, 1e-12)
    ratio_b = p_poison_lo / max(p_clean_hi, 1e-12)

    ratio = max(ratio_a, ratio_b, 1.0)
    eps_lb = max(0.0, math.log(ratio) / max(poisoning_k, 1))

    debug = {
        "p_clean_lo": p_clean_lo,
        "p_clean_hi": p_clean_hi,
        "p_poison_lo": p_poison_lo,
        "p_poison_hi": p_poison_hi,
        "ratio": ratio,
    }
    return eps_lb, debug


def choose_threshold(cal_clean: List[float], cal_poison: List[float], poisoning_k: int, alpha: float) -> float:
    vals = np.array(cal_clean + cal_poison)
    if vals.size == 0:
        return 0.5

    candidates = np.unique(np.quantile(vals, q=np.linspace(0.1, 0.9, 17)))
    best_eps = -1.0
    best_t = float(np.median(vals))

    n = min(len(cal_clean), len(cal_poison))
    for t in candidates:
        c_hits = int(np.sum(np.array(cal_clean[:n]) >= t))
        p_hits = int(np.sum(np.array(cal_poison[:n]) >= t))
        eps, _ = eps_lb_from_counts(c_hits, p_hits, n, poisoning_k, alpha)
        if eps > best_eps:
            best_eps = eps
            best_t = float(t)
    return best_t


def run_audit(
    x_train,
    y_train,
    x_test,
    y_test,
    cfg: Dict,
) -> Dict:
    clean_scores: List[float] = []
    poison_scores: List[float] = []
    eps_th_values: List[float] = []
    acc_clean_values: List[float] = []
    acc_poison_values: List[float] = []

    n_trials = cfg["num_trials"]

    for i in range(n_trials):
        trial_seed = cfg["seed"] + i

        clean_out = train_dp_once(
            x_train,
            y_train,
            x_test,
            y_test,
            cfg,
            seed=trial_seed,
        )

        x_poison, y_poison = poison_dataset(
            x_train,
            y_train,
            poisoning_k=cfg["poisoning_k"],
            target_label=cfg["target_label"],
            seed=trial_seed,
            trigger_size=cfg["trigger_size"],
        )

        poison_out = train_dp_once(
            x_poison,
            y_poison,
            x_test,
            y_test,
            cfg,
            seed=trial_seed + 10_000,
        )

        clean_scores.append(clean_out["trigger_success"])
        poison_scores.append(poison_out["trigger_success"])
        eps_th_values.append((clean_out["epsilon_theoretical"] + poison_out["epsilon_theoretical"]) / 2.0)
        acc_clean_values.append(clean_out["clean_accuracy"])
        acc_poison_values.append(poison_out["clean_accuracy"])

    split = max(2, n_trials // 2)
    cal_clean = clean_scores[:split]
    cal_poison = poison_scores[:split]
    est_clean = clean_scores[split:]
    est_poison = poison_scores[split:]

    if len(est_clean) == 0 or len(est_poison) == 0:
        est_clean = clean_scores
        est_poison = poison_scores

    threshold = choose_threshold(cal_clean, cal_poison, cfg["poisoning_k"], cfg["alpha"])

    est_n = min(len(est_clean), len(est_poison))
    clean_hits = int(np.sum(np.array(est_clean[:est_n]) >= threshold))
    poison_hits = int(np.sum(np.array(est_poison[:est_n]) >= threshold))

    eps_emp, debug = eps_lb_from_counts(
        clean_hits,
        poison_hits,
        est_n,
        cfg["poisoning_k"],
        cfg["alpha"],
    )

    eps_th = float(np.mean(eps_th_values)) if eps_th_values else float("nan")
    gap_ratio = float("nan") if eps_emp <= 0.0 else (eps_th / eps_emp)

    return {
        "epsilon_theoretical": eps_th,
        "epsilon_empirical_lb": eps_emp,
        "gap_ratio": gap_ratio,
        "threshold": threshold,
        "clean_trigger_mean": float(np.mean(clean_scores)),
        "poison_trigger_mean": float(np.mean(poison_scores)),
        "clean_acc_mean": float(np.mean(acc_clean_values)),
        "poison_acc_mean": float(np.mean(acc_poison_values)),
        "n_trials": n_trials,
        "n_estimation_trials": est_n,
        "debug": debug,
    }
