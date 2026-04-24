from __future__ import annotations

import argparse

from src.experiments import regenerate_outputs, run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DP-SGD privacy auditing sweeps")
    parser.add_argument("--config", type=str, default="configs/sweep.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--plots-only", action="store_true", help="Regenerate plots and CSV summaries from an existing summary.csv")
    parser.add_argument("--results-csv", type=str, default=None, help="Path to an existing summary.csv for plot-only mode")
    args = parser.parse_args()

    if args.plots_only:
        results_csv = args.results_csv or f"{args.output_dir}/summary.csv"
        df = regenerate_outputs(results_csv, args.output_dir)
    else:
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
