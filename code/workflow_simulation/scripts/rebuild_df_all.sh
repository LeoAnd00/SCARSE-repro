#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-62
#SBATCH -t 00:15:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J rebuild-df-all
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts. Submit from this folder
# after a one-off `mkdir -p logs`, or the job dies before printing anything.

# Rebuild simulation_results/df_all.csv from the per-run files in runs/.
#
# =====================================================================
# HOW TO RUN  (submit from this scripts/ folder)
#
#   mkdir -p logs                 # one-off, so SLURM can open the log files
#   sbatch rebuild_df_all.sh      # rebuild and write df_all.csv
#
#   CHECK=1 sbatch rebuild_df_all.sh   # preview only, write nothing
#   bash rebuild_df_all.sh             # run directly on a login node (no SLURM)
# =====================================================================
#
# For reproducibility: after re-running the whole pipeline (which writes one CSV
# per run into runs/), this collapses those into a single df_all.csv identical to
# what the notebooks consume. It reads ONLY runs/ (never folds an existing
# df_all.csv into itself) and reuses the pipeline's own load_results() so the
# torn-row dropping, Acquisition backfill and re-run de-duplication all match.
#
# The notebooks still read both df_all.csv AND runs/ via load_results(); this
# script does not change that. It just (re)creates df_all.csv so a fresh checkout
# has the file the notebooks expect. Any existing df_all.csv is backed up first.
#
# Run it after a full pipeline re-run, and after the HemoPI2 merge if you did one.

module purge > /dev/null 2>&1 || true

# ------------------------------------------------------------------
# PATHS - edit PROJECT_ROOT once; everything below follows from it.
# ------------------------------------------------------------------
PROJECT_ROOT="/proj/berzelius-2026-62/users/${USER}/reproducibility_code"
CONTAINER="${PROJECT_ROOT}/env.sif"
CODE_DIR="${PROJECT_ROOT}/code"
MODULE_DIR="${CODE_DIR}/workflow_simulation"

cd "${MODULE_DIR}" || {
    echo "No such directory: ${MODULE_DIR}"
    echo "Check that PROJECT_ROOT above points at your copy of the repo."
    exit 1
}
mkdir -p "${MODULE_DIR}/scripts/logs"

export PYTHONUNBUFFERED=1

RESULTS_DIR="simulation_results"

# Set CHECK=1 to report only (no file written); leave unset to write.
CHECK_FLAG=""
if [ -n "${CHECK:-}" ]; then
    CHECK_FLAG="--check"
    echo ">> CHECK: reporting only, df_all.csv will not be written."
fi

echo "Results dir: ${RESULTS_DIR}"

apptainer exec "$CONTAINER" \
python rebuild_df_all.py \
  --results-dir "${RESULTS_DIR}" \
  ${CHECK_FLAG}

echo ">>> Done."
