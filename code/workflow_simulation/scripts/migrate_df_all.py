#!/usr/bin/env python3
"""One-off migration: split df_all.csv into one file per run, and list what is missing.

Older runs all appended to a single ``simulation_results/df_all.csv``. Array
tasks that finished at the same moment could interleave their writes there,
leaving torn lines. The simulation now writes one file per run into
``simulation_results/runs/`` instead.

This script:
  1. reads df_all.csv, dropping any damaged rows;
  2. writes one CSV per (representation, strategy, dataset, seed) into runs/;
  3. reports which runs are missing or incomplete, with the sbatch commands
     that would produce them.

Existing files in runs/ are kept: a run already there is assumed to be newer
than the same run in df_all.csv, and is not overwritten unless --overwrite is
given. Nothing is deleted, and df_all.csv is left untouched.

Usage, from this scripts/ folder:

    python migrate_df_all.py
    python migrate_df_all.py --results_dir ../simulation_results --dry_run
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functions.model_optimization import (  # noqa: E402
    ACQUISITION_STRATEGIES,
    REPRESENTATION_LABELS,
    RESULT_ID_COLUMNS,
    RUNS_SUBDIR,
    run_file_name,
)

#: Datasets in the order simulation_per_seed.sh lists them: the array task id is
#: dataset index * NUM_SEEDS + seed index, and the seed is 42 + seed index.
DATASETS = [
    "A0A247D711_LISMN", "DN7A_SACS2", "ENVZ_ECOLI", "FKBP3_HUMAN",
    "MAFG_MOUSE_sub", "POLG_PESV_sub", "SBI_STAAM", "SDA_BACSU",
    "SOX30_HUMAN", "YNZC_BACSU_sub", "Ecoli_AMPs", "Saureus_AMPs",
    "Paeruginosa_AMPs", "HemoPI2",
]
NUM_SEEDS = 10
FIRST_SEED = 42


def task_id(dataset, seed):
    """Array task id for one (dataset, seed), or None if the dataset is unknown."""
    if dataset not in DATASETS:
        return None
    return DATASETS.index(dataset) * NUM_SEEDS + (int(seed) - FIRST_SEED)


def compact(ids):
    """[1, 2, 3, 7] -> '1-3,7', the form sbatch --array expects."""
    ids = sorted(ids)
    out, start, prev = [], None, None
    for i in ids + [None]:
        if start is None:
            start = prev = i
            continue
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = i
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="../simulation_results",
                    help="Folder holding df_all.csv (default: ../simulation_results)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite run files that already exist.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report only; write nothing.")
    args = ap.parse_args()

    src = os.path.join(args.results_dir, "df_all.csv")
    run_dir = os.path.join(args.results_dir, RUNS_SUBDIR)
    if not os.path.exists(src):
        sys.exit(f"No such file: {src}")

    df = pd.read_csv(src, sep=None, engine="python")
    print(f"Read {len(df)} rows from {src}")

    if "Acquisition" not in df.columns:
        df["Acquisition"] = "greedy"
    df["Acquisition"] = df["Acquisition"].fillna("greedy")

    # Torn lines leave fields empty or shifted; such rows cannot be repaired.
    ok = (df["Representation"].isin(REPRESENTATION_LABELS)
          & df["Acquisition"].isin(ACQUISITION_STRATEGIES)
          & pd.to_numeric(df["Value"], errors="coerce").notna()
          & pd.to_numeric(df["Num_samples"], errors="coerce").notna())
    if (~ok).any():
        print(f"Dropping {int((~ok).sum())} damaged row(s):")
        print(df[~ok].to_string(index=False))
        df = df[ok]

    df["Value"] = pd.to_numeric(df["Value"])
    df["Num_samples"] = pd.to_numeric(df["Num_samples"]).astype(int)
    df["Seed"] = pd.to_numeric(df["Seed"]).astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=RESULT_ID_COLUMNS, keep="last")
    if len(df) < before:
        print(f"Dropped {before - len(df)} duplicate row(s).")

    # ---------------------------------------------------------------- write
    groups = list(df.groupby(["Representation", "Acquisition", "Dataset", "Seed"],
                             sort=True))
    written = skipped = 0
    if not args.dry_run:
        os.makedirs(run_dir, exist_ok=True)
    for (rep, acq, ds, seed), part in groups:
        path = os.path.join(run_dir, run_file_name(ds, rep, acq, seed))
        if os.path.exists(path) and not args.overwrite:
            skipped += 1
            continue
        if not args.dry_run:
            part.to_csv(path, index=False)
        written += 1
    verb = "Would write" if args.dry_run else "Wrote"
    print(f"\n{verb} {written} run file(s) to {run_dir}"
          + (f"; {skipped} already existed and were kept." if skipped else "."))

    # ---------------------------------------------------------------- report
    # A complete run has the same number of rows as most runs do.
    sizes = pd.Series({k: len(v) for k, v in groups})
    full = int(sizes.mode().iloc[0])
    print(f"\nA complete run has {full} rows "
          f"({int((sizes == full).sum())} of {len(sizes)} runs).")

    missing = []
    for rep in REPRESENTATION_LABELS:
        for acq in ACQUISITION_STRATEGIES:
            for ds in DATASETS:
                for seed in range(FIRST_SEED, FIRST_SEED + NUM_SEEDS):
                    n = int(sizes.get((rep, acq, ds, seed), 0))
                    if n < full:
                        missing.append({"Representation": rep, "Acquisition": acq,
                                        "Dataset": ds, "Seed": seed, "rows": n,
                                        "task": task_id(ds, seed)})

    if not missing:
        print("\nEvery run is complete. Nothing to re-run.")
        return

    miss = pd.DataFrame(missing)
    print(f"\n{len(miss)} run(s) missing or incomplete:\n")
    print(miss.to_string(index=False))

    print("\nRe-run them with (from this scripts/ folder):\n")
    for (rep, acq), part in miss.groupby(["Representation", "Acquisition"]):
        tasks = sorted(t for t in part["task"] if t is not None)
        if tasks:
            print(f"  sbatch --array={compact(tasks)} simulation_per_seed.sh {rep} {acq}")
    unknown = miss[miss["task"].isna()]["Dataset"].unique()
    if len(unknown):
        print(f"\nNot in the script's dataset list, so no task id: {sorted(unknown)}")

    print("\nIncomplete runs are re-run whole; their new file replaces the old one.")


if __name__ == "__main__":
    main()
