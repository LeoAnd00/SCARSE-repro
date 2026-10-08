import os
import glob
import argparse
import pandas as pd

#: Columns that identify one simulation run. Used to drop rows that appear both
#: in this benchmark's own output and in a reused foundation-model run.
RUN_ID_COLS = ["Dataset", "Label", "Representation", "Foundation",
               "Model", "Train Size", "Seed"]


def save_summary(out_dir, save_path, reuse_esm_from=None,
                 reuse_tag="esm2_t33_650M_UR50D"):
    """
    Aggregate per-seed regression model performance CSVs into a summary table.

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
                    <representation_tag>/
                        seed_<seed>/
                            train_size_<n>/
                                raw_simulation_summary.csv
    save_path : str
        File path where the aggregated summary CSV will be written.
    reuse_esm_from : str, optional
        A foundation-model benchmark output root for the same family, e.g.
        ``simulation_output/foundations/substitutions``. The `reuse_tag`
        subtree is read from there instead of requiring this benchmark to
        re-run the ESM2 representation, which is the identical computation:
        both benchmarks call this module with the same datasets, seeds, train
        sizes and `--representation esm --foundation_model facebook/<tag>`.
        Only that one checkpoint is taken, so the smaller checkpoints stay out
        of the representation comparison.
    reuse_tag : str
        Checkpoint folder to take from `reuse_esm_from`. Must be the checkpoint
        this benchmark would otherwise have run for `--representation esm`.

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

    own_files = list(perf_files)
    reused_files = []

    if reuse_esm_from:
        # Exactly <dataset>/<reuse_tag>/seed_*/train_size_*, so the 8M/35M/150M
        # checkpoints in the same tree are not pulled in.
        reused_files = glob.glob(
            os.path.join(reuse_esm_from, "*", reuse_tag,
                         "seed_*", "train_size_*", "raw_simulation_summary.csv")
        )
        if not reused_files:
            raise ValueError(
                f"--reuse_esm_from {reuse_esm_from} has no "
                f"*/{reuse_tag}/seed_*/train_size_*/raw_simulation_summary.csv files.\n"
                "Run the foundation-model benchmark for that family and checkpoint "
                "first, or drop the flag and submit this benchmark with "
                "'esm' as the representation."
            )
        perf_files = own_files + reused_files

    df_list = [pd.read_csv(f) for f in perf_files]
    df = pd.concat(df_list, ignore_index=True)

    if reused_files:
        n_reused = sum(len(pd.read_csv(f)) for f in reused_files)
        print(f"[Reuse] {len(reused_files)} run folders ({n_reused} rows) taken from "
              f"{reuse_esm_from} for {reuse_tag}")

        bad = df[(df["Representation"] == "esm")
                 & (~df["Foundation"].astype(str).str.endswith(reuse_tag))]
        if not bad.empty:
            raise ValueError(
                "Reused rows carry unexpected checkpoints: "
                f"{sorted(bad['Foundation'].unique())}. Only {reuse_tag} belongs in "
                "the representation comparison."
            )

        # Running esm here as well as reusing it would count every run twice.
        have = [c for c in RUN_ID_COLS if c in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=have, keep="first")
        dropped = before - len(df)
        if dropped:
            print(f"[Reuse] dropped {dropped} duplicate runs already present locally")

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

    group_cols = ["Dataset", "Label", "Representation",
                  "Foundation", "Model", "Train Size"]

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Summary CSVs are missing expected columns: {missing}. Found: {df.columns.tolist()}. "
            "Results produced before the representation refactor used a 'Baseline' column and "
            "need to be re-run or renamed."
        )

    def agg_group(g):
        result = {}

        for m in metrics:
            s = g[m]
            result[f"{m} Mean"] = s.mean()
            result[f"{m} Std"] = s.std()

        return pd.Series(result)

    summary_df = (
        df.groupby(group_cols)[metrics]
          .apply(agg_group)
          .reset_index()
    )

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
    parser.add_argument(
        "--reuse_esm_from", type=str, default=None,
        help="Foundation-model benchmark output root for the same family, e.g. "
             "simulation_output/foundations/substitutions. Its --reuse_tag subtree "
             "is used as this benchmark's esm representation instead of re-running it."
    )
    parser.add_argument(
        "--reuse_tag", type=str, default="esm2_t33_650M_UR50D",
        help="Checkpoint folder to take from --reuse_esm_from (default: the 650M model)."
    )
    args = parser.parse_args()

    save_summary(args.out_dir, args.summary_csv,
                 reuse_esm_from=args.reuse_esm_from,
                 reuse_tag=args.reuse_tag)