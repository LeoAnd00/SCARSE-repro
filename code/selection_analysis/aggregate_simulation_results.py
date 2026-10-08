import os
import glob
import argparse
import pandas as pd

def save_summary(out_dir, save_path):
    """
    Aggregate per-seed protein model performance CSVs into a summary table.

    This function searches for all `raw_simulation_summary.csv` files
    produced during simulation runs, merges them, and computes mean ±
    standard deviation metrics across random seeds.

    Aggregation is performed per:
        Dataset * Label * Representation * Foundation * Model * Train Size

    Parameters
    ----------
    out_dir : str
        Root directory containing per-dataset, per-seed simulation outputs.
        Expected directory structure:
            out_dir/
                <dataset>/
                    <representation_tag>/
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
    perf_files = glob.glob(os.path.join(out_dir, "**", "seed_*", "train_size_*", "raw_simulation_summary.csv"), recursive=True)
    if not perf_files:
        raise ValueError(f"No raw_simulation_summary.csv files found under {out_dir}")

    df_list = [pd.read_csv(f) for f in perf_files]
    df = pd.concat(df_list, ignore_index=True)

    if "Best Params" in df.columns:
        df = df.drop("Best Params", axis=1)

    df.to_csv(save_path.split(".")[0] + "_all.csv", index=False)

    metrics = [
        "CV MSE", "CV RMSE", "CV MAE", "CV R2", "CV Spearman Correlation",
        "Test MSE", "Test RMSE", "Test MAE", "Test R2", "Test Spearman Correlation",
    ]
    metrics = [m for m in metrics if m in df.columns]

    group_cols = ["Dataset", "Label", "Representation", "Foundation", "Model", "Train Size"]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Summary CSVs are missing expected columns: {missing}. Found: {df.columns.tolist()}. "
            "Results produced before the representation refactor need to be re-run."
        )

    summary_df = (
        df.groupby(group_cols)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten the MultiIndex columns into "<metric> Mean" / "<metric> Std"
    summary_df.columns = [
        col[0] if col[1] == "" else f"{col[0]} {col[1].capitalize()}"
        for col in summary_df.columns.to_flat_index()
    ]

    # Sort directly rather than via groupby().apply(): pandas 3 no longer passes
    # the grouping columns into the applied function, which silently dropped
    # "Dataset" and "Label" from the output.
    summary_df = summary_df.sort_values(
        ["Dataset", "Label", "Train Size", "Test MSE Mean"]
    ).reset_index(drop=True)

    summary_df.to_csv(save_path, index=False)
    print(f"[Saved] Aggregated summary CSV: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate per-seed protein simulation results")
    parser.add_argument("out_dir", type=str, help="Base output folder containing per-seed CSVs")
    parser.add_argument("summary_csv", type=str, help="Path to save the aggregated summary CSV")
    args = parser.parse_args()

    save_summary(args.out_dir, args.summary_csv)
