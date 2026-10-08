#!/usr/bin/env python3
"""
Rebuild simulation_results/df_all.csv from the per-run files in runs/.

The pipeline stores one CSV per (dataset, representation, strategy, seed) in
<results_dir>/runs/, and load_results() stitches those together (optionally with
a legacy df_all.csv). For reproducibility we also want a single df_all.csv that
holds every run, so a fresh checkout that re-ran everything ends up with the same
df_all.csv the notebooks expect.

This script builds that file by calling load_results() itself, so the output is
identical to what the notebooks consume: the same columns, the same torn-row
dropping, the same Acquisition backfill, and the same "re-run replaces its earlier
rows" de-duplication. To avoid folding an existing df_all.csv back into itself, it
reads ONLY the runs/ files (it points load_results at a non-existent legacy path).

Usage (from the workflow_simulation folder):
    python3 rebuild_df_all.py                       # writes simulation_results/df_all.csv
    python3 rebuild_df_all.py --results-dir simulation_results
    python3 rebuild_df_all.py --out some/other.csv  # write elsewhere
    python3 rebuild_df_all.py --check               # verify only, do not write
"""
import argparse
import os
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="simulation_results",
                    help="Results dir holding runs/ (default: simulation_results).")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <results-dir>/df_all.csv).")
    ap.add_argument("--check", action="store_true",
                    help="Only report what would be written; do not write the file.")
    args = ap.parse_args()

    results_dir = args.results_dir
    runs_dir = os.path.join(results_dir, "runs")
    out = args.out or os.path.join(results_dir, "df_all.csv")

    if not os.path.isdir(runs_dir):
        sys.exit(f"ERROR: no runs/ folder at {runs_dir}. Nothing to rebuild from.")

    n_run_files = len([f for f in os.listdir(runs_dir) if f.endswith(".csv")])
    if n_run_files == 0:
        sys.exit(f"ERROR: {runs_dir} contains no .csv run files.")

    # Import the pipeline's own loader so the rebuilt file matches it exactly.
    sys.path.insert(0, os.path.abspath("."))
    try:
        from functions.model_optimization import load_results
    except Exception as e:
        sys.exit(f"ERROR: could not import load_results from functions.model_optimization: {e}")

    # Read ONLY the runs/ files: point the legacy-csv argument at a path that does
    # not exist, so an existing df_all.csv is not folded back into the rebuild.
    sentinel = os.path.join(results_dir, "__no_legacy_df_all__.csv")
    df = load_results(results_dir=results_dir, results_csv=sentinel)

    print(f"Run files read : {n_run_files}")
    print(f"Rows assembled : {len(df)}")
    print(f"Datasets       : {df['Dataset'].nunique()} "
          f"({', '.join(sorted(df['Dataset'].unique()))})")
    if "Acquisition" in df.columns:
        print(f"Acquisitions   : {', '.join(sorted(df['Acquisition'].unique()))}")
    print(f"Metrics        : {', '.join(sorted(df['Metric'].unique()))}")

    if args.check:
        print(f"\nCheck only. Would write {len(df)} rows to {out}.")
        return

    # Back up an existing df_all.csv before overwriting it.
    if os.path.exists(out):
        from datetime import datetime
        bak = f"{out}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.replace(out, bak)
        print(f"Backed up existing file to {bak}")

    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
