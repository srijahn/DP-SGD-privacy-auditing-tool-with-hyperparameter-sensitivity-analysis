from __future__ import annotations

import argparse

from src.experiments import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DP-SGD privacy auditing sweeps")
    parser.add_argument("--config", type=str, default="configs/sweep.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-configs", type=int, default=None)
    args = parser.parse_args()

    df = run_all(args.config, args.output_dir, max_configs=args.max_configs)
    print("Finished. Rows:", len(df))
    if not df.empty:
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
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
