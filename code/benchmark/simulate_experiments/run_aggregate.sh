#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-62
#SBATCH -t 01:00:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J rep-agg
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts, so no mkdir inside the
# script can help. Submit from this folder after a one-off `mkdir -p logs`, or
# the job dies before printing anything.

# Aggregate the representation benchmark, ONE DATASET FAMILY PER RUN.
#
#   sbatch run_aggregate.sh <family>
#
# <family> names one of the three dataset families, and is the same word as in
# the <family>.sh script that produced the results:
#
#   substitutions   10 ProteinGym substitution DMS assays
#   indels          10 ProteinGym indel DMS assays
#   peptides        E.coli / S.aureus / P.aeruginosa AMPs + HemoPI2
#
# So, after the three baseline submissions for a family have finished:
#   sbatch run_aggregate.sh substitutions
#   sbatch run_aggregate.sh indels
#   sbatch run_aggregate.sh peptides
#
# Reads  simulation_output/<family>/              (physchem, morgan, ngram)
#  plus  simulation_output/foundations/<family>/  (esm, reused - see below)
# Writes simulation_output/<family>/simulation_summary.csv
#
# One run covers every representation, because the aggregation walks all the
# representation folders under that family. It reads whatever is on disk when
# it runs, so queue it behind the array jobs with
# --dependency=afterok:<jobid>... if you do not want to wait and submit later.

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

# Set variables
# One family per run, passed as the first argument:
#   substitutions | indels | peptides
# The aggregation walks every representation folder underneath, so a single run
# covers physchem, morgan and ngram together.
# Required: defaulting here would silently summarise a family you did not ask for.
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

SIM_OUT="${CODE_DIR}/benchmark/simulate_experiments/simulation_output"
OUT_DIR="${SIM_OUT}/${FAMILY}"                    # Root folder with per-seed CSVs
SUMMARY_CSV="${OUT_DIR}/simulation_summary.csv"   # Output aggregated summary CSV

# ------------------------------------------------------------------
# REUSING THE ESM2 650M RUNS FROM THE FOUNDATION-MODEL BENCHMARK
# ------------------------------------------------------------------
# The esm arm of this benchmark is the identical computation to the 650M arm of
# the foundation-model benchmark: same datasets, same seeds 42-51, same train
# sizes, same optimize_models.py call. So run the foundation benchmark first and
# read its 650M results here instead of spending another 1 380 tasks on them.
# Only the 650M subtree is taken; 8M/35M/150M stay out of this comparison.
#
# Set REUSE_ESM=0 to switch this off, in which case you must also submit
# <family>.sh with 'esm' as the representation.
REUSE_ESM="${REUSE_ESM:-1}"
REUSE_TAG="${REUSE_TAG:-esm2_t33_650M_UR50D}"
FOUNDATIONS_DIR="${SIM_OUT}/foundations/${FAMILY}"

REUSE_ARGS=()
if [ "$REUSE_ESM" = "1" ]; then
    REUSE_ARGS=(--reuse_esm_from "$FOUNDATIONS_DIR" --reuse_tag "$REUSE_TAG")
    echo ">> Reusing ${REUSE_TAG} from ${FOUNDATIONS_DIR}"
else
    echo ">> REUSE_ESM=0: expecting this benchmark's own esm runs under ${OUT_DIR}"
fi

# Run the Python aggregation script, e.g.:  bash run_aggregate.sh peptides
apptainer exec "$CONTAINER" python aggregate_simulation_results.py \
    "$OUT_DIR" "$SUMMARY_CSV" "${REUSE_ARGS[@]}"