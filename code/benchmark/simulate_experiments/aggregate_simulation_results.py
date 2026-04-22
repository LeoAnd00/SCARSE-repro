import os
import glob
import argparse
import pandas as pd

def save_summary(out_dir, save_path):
    """
    Aggregate per-seed protein regression model performance CSVs into a summary table.

    This function searches for all `raw_simulation_summary.csv` files
    produced during simulation runs, merges them, and computes mean ±
    standard deviation metrics across random seeds.

    Parameters
    ----------
    out_dir : str
        Root directory containing per-dataset, per-seed simulation outputs.
        Expected directory structure:
            out_dir/
                <dataset>/
                    seed_<seed>/
                        train_size_<n>/
                            raw_simulation_summary.csv
    save_path : str
        File path where the aggregated summary CSV will be written.

    Returns
    -------
    None
        Writes the aggregated summary CSV to disk.
    """
    perf_files = glob.glob(
        os.path.join(out_dir, "**", "seed_*", "train_size_*", "raw_simulation_summary.csv"),
        recursive=True
    )

    if not perf_files:
        raise ValueError(f"No raw_simulation_summary.csv files found under {out_dir}")

    df_list = [pd.read_csv(f) for f in perf_files]
    df = pd.concat(df_list, ignore_index=True)

    if "Best Params" in df.columns:
        df = df.drop("Best Params", axis=1)

    df.to_csv(save_path.split(".")[0] + "_all.csv", index=False)

    metrics = [
        "CV MSE",
        "Test MSE",
        "Test RMSE",
        "Test MAE",
        "Test R2",
        "Test Spearman Correlation",
        "Top 20 Accuracy"
    ]

    group_cols = ["Dataset", "Label", "Baseline",
                  "Foundation", "Model", "Train Size"]

    def agg_group(g):
        result = {}

        for m in metrics:
            s = g[m]

            if m == "Test R2" and out_dir.endswith("substitutions"):
                s = s[s >= -10]

            result[f"{m} Mean"] = s.mean()
            result[f"{m} Std"] = s.std()

        return pd.Series(result)

    summary_df = (
        df.groupby(group_cols)
          .apply(agg_group)
          .reset_index()
    )

    summary_df = summary_df.groupby(["Dataset", "Label"], group_keys=False).apply(
        lambda d: d.sort_values(["Train Size", "Test MSE Mean"])
    ).reset_index(drop=True)

    summary_df.to_csv(save_path, index=False)
    print(f"[Saved] Aggregated summary CSV: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate per-seed protein simulation results")
    parser.add_argument("out_dir", type=str, help="Base output folder containing per-seed CSVs")
    parser.add_argument("summary_csv", type=str, help="Path to save the aggregated summary CSV")
    args = parser.parse_args()

    save_summary(args.out_dir, args.summary_csv)
