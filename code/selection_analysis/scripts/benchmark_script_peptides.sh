#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-62
#SBATCH -t 48:00:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -J sel-pept
#SBATCH --array=0-359   # 4 datasets x 10 seeds x 9 train sizes
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts, so no mkdir inside the
# script can help. Submit from this folder after a one-off `mkdir -p logs`, or
# the job dies before printing anything.

# Selection analysis: cross-validation quality vs end-point selection
# performance. One representation per submission, passed as the first argument:
#   sbatch benchmark_script_peptides.sh esm
#   sbatch benchmark_script_peptides.sh physchem
#   sbatch benchmark_script_peptides.sh morgan
#   sbatch benchmark_script_peptides.sh ngram

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
MODULE_DIR="${CODE_DIR}/selection_analysis"
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

# ------------------------
# REPRESENTATION
# ------------------------
REPRESENTATION=${1:-esm}
case "$REPRESENTATION" in
    esm|physchem|morgan|ngram) ;;
    *) echo "Unknown representation: $REPRESENTATION (expected esm, physchem, morgan or ngram)"; exit 1 ;;
esac

# ------------------------
# DATASETS
# ------------------------

# Dataset names (14 entries) - must match the workflow simulation's names so
# the two result sets can be joined in selecting_workflow_eval.ipynb
dataset_names=(
  Ecoli_AMPs
  Saureus_AMPs
  Paeruginosa_AMPs
  HemoPI2
)

# Corresponding file names
file_names=(
  ecoli_amps/processed/EC_all_40
  saureus_amps/processed/SA_all_40
  paeruginosa_amps/processed/PA_all_40
  hemopi2/processed/hemopi2_all
)
NUM_DATASETS=${#dataset_names[@]}

# ------------------------
# SEEDS
# ------------------------
NUM_SEEDS=10
SEED_BASE=42

# ------------------------
# TRAIN SIZES
# ------------------------
TRAIN_SIZES=(20 30 40 50 60 70 80 90 100)
NUM_TRAIN_SIZES=${#TRAIN_SIZES[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

TS_IDX=$((TASK_ID % NUM_TRAIN_SIZES))
TASK_ID=$((TASK_ID / NUM_TRAIN_SIZES))

SEED_IDX=$((TASK_ID % NUM_SEEDS))
TASK_ID=$((TASK_ID / NUM_SEEDS))

DATA_IDX=$((TASK_ID % NUM_DATASETS))

SEED=$((SEED_BASE + SEED_IDX))
TRAIN_SIZE=${TRAIN_SIZES[$TS_IDX]}

echo ">> Representation: $REPRESENTATION"
echo ">> Dataset:        ${dataset_names[$DATA_IDX]}"
echo ">> Seed:           $SEED"
echo ">> Train size:     $TRAIN_SIZE"

# ------------------------
# PATHS
# ------------------------
DATA="${DATA_DIR}/${file_names[$DATA_IDX]}.csv"

OUT="${CODE_DIR}/selection_analysis/simulation_output"

# ------------------------
# RUN
# ------------------------
apptainer exec \
    "$CONTAINER" \
    python optimize_models.py \
        "$DATA" "$OUT" \
        --n_seeds $SEED \
        --train_sizes $TRAIN_SIZE \
        --representation "$REPRESENTATION" \
        --foundation_model "facebook/esm2_t33_650M_UR50D"