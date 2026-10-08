#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-62
#SBATCH -t 01:00:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J fnd-agg
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts, so no mkdir inside the
# script can help. Submit from this folder after a one-off `mkdir -p logs`, or
# the job dies before printing anything.

# Aggregate the ESM2 foundation-model benchmark, ONE DATASET FAMILY PER RUN.
#
#   sbatch run_aggregate_foundations.sh <family>
#
# <family> names one of the three dataset families, and is the same word as in
# the foundations_<family>.sh script that produced the results:
#
#   substitutions   10 ProteinGym substitution DMS assays
#   indels          10 ProteinGym indel DMS assays
#   peptides        E.coli / S.aureus / P.aeruginosa AMPs + HemoPI2
#
# So, after all of foundations_<family>.sh have finished:
#   sbatch run_aggregate_foundations.sh substitutions
#   sbatch run_aggregate_foundations.sh indels
#   sbatch run_aggregate_foundations.sh peptides
#
# Reads  simulation_output/foundations/<family>/
# Writes simulation_output/foundations/<family>/simulation_summary.csv
#
# One run covers all four ESM2 checkpoints, because the aggregation walks every
# checkpoint folder under that family. It reads whatever is on disk when it
# runs, so a family whose array jobs are still going is summarised half-done -
# either wait, or queue it behind them:
#   sbatch --dependency=afterok:<jobid1>:<jobid2>:<jobid3>:<jobid4> \
#          run_aggregate_foundations.sh substitutions

# Everything runs inside the container, so nothing is needed from the host
# environment. NSC's mpprun/nsc modules are sticky and a plain purge cannot
# unload them, which is harmless - purge quietly and carry on.
module purge > /dev/null 2>&1 || true

# ------------------------------------------------------------------
# PATHS - edit PROJECT_ROOT once; everything below follows from it.
# ------------------------------------------------------------------
PROJECT_ROOT="/proj/berzelius-2026-62/users/${USER}/reproducibility_code"
CONTAINER="${PROJECT_ROOT}/env.sif"
DATA_DIR="${PROJECT_ROOT}/data"
CODE_DIR="${PROJECT_ROOT}/code"

# SLURM copies the batch script into its own spool directory before running it,
# so ${BASH_SOURCE[0]} is /var/lib/slurm/... and cannot be used to find this
# file. Derive the module directory from PROJECT_ROOT instead and cd there, so
# every relative path below is the same no matter where sbatch was run from.
MODULE_DIR="${CODE_DIR}/benchmark/simulate_experiments"
cd "${MODULE_DIR}" || {
    echo "No such directory: ${MODULE_DIR}"
    echo "Check that PROJECT_ROOT above points at your copy of the repo."
    exit 1
}
mkdir -p "${MODULE_DIR}/scripts/logs"

# The ESM2 checkpoints are pre-downloaded by code/setup/cache_esm_models.sh.
# Offline mode means no array task ever contacts huggingface.co.
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Keep BLAS inside the CPU allocation instead of grabbing the whole node.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ------------------------------------------------------------------
# FAMILY - which dataset family to aggregate. Required: defaulting here
# would silently summarise a family you did not ask for.
# ------------------------------------------------------------------
FAMILY="${1:-}"
case "$FAMILY" in
    substitutions|indels|peptides) ;;
    "") echo "Missing argument: which dataset family to aggregate."
        echo "Usage: sbatch $(basename "$0") <family>"
        echo "  <family> = substitutions | indels | peptides"
        exit 1 ;;
    *)  echo "Unknown family: '$FAMILY'"
        echo "Expected one of: substitutions indels peptides"
        exit 1 ;;
esac

BASE="${CODE_DIR}/benchmark/simulate_experiments/simulation_output/foundations"
OUT_DIR="${BASE}/${FAMILY}"                       # Root folder with per-seed CSVs
SUMMARY_CSV="${OUT_DIR}/simulation_summary.csv"   # Output aggregated summary CSV

if [ ! -d "$OUT_DIR" ]; then
    echo "No results for family '$FAMILY' at: $OUT_DIR"
    echo "That folder is written by foundations_${FAMILY}.sh - has it finished?"
    exit 1
fi

echo ">> Aggregating family '$FAMILY' from $OUT_DIR"

apptainer exec "$CONTAINER" python aggregate_simulation_results.py \
    "$OUT_DIR" "$SUMMARY_CSV"