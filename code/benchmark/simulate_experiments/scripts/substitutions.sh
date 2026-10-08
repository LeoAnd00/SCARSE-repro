#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-62
#SBATCH -t 48:00:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -J rep-subs
#SBATCH --array=0-699
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts, so no mkdir inside the
# script can help. Submit from this folder after a one-off `mkdir -p logs`, or
# the job dies before printing anything.

# One representation per submission, passed as the first argument:
#   sbatch substitutions.sh physchem
#   sbatch substitutions.sh morgan
#   sbatch substitutions.sh ngram
#
# RUN THE FOUNDATION-MODEL BENCHMARK FIRST, and only the three baselines here.
# The esm arm of this benchmark is the identical computation to the 650M arm of
# foundations_substitutions.sh - same datasets, same seeds, same train sizes, same
# optimize_models.py call - so run_aggregate.sh reads those results instead
# (--reuse_esm_from) and there is nothing to gain from repeating them.
#
# To queue the three baselines one after another:
#   jid=""
#   for rep in physchem morgan ngram; do
#       dep=""; [ -n "$jid" ] && dep="--dependency=afterok:$jid"
#       jid=$(sbatch $dep substitutions.sh $rep | awk '{print $4}')
#   done
#
# 'esm' is still accepted, for the case where you want this benchmark to stand
# on its own. Then also aggregate with REUSE_ESM=0:
#   sbatch substitutions.sh esm
#   REUSE_ESM=0 bash run_aggregate.sh substitutions

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
DATASETS=(
    "ENVZ_ECOLI_Ghose_2023.csv"
    "A0A247D711_LISMN_Stadelmann_2021.csv"
    "DN7A_SACS2_Tsuboyama_2023_1JIC.csv"
    "FKBP3_HUMAN_Tsuboyama_2023_2KFV.csv"
    "MAFG_MOUSE_Tsuboyama_2023_1K1V.csv"
    "POLG_PESV_Tsuboyama_2023_2MXD.csv"
    "SBI_STAAM_Tsuboyama_2023_2JVG.csv"
    "SDA_BACSU_Tsuboyama_2023_1PV0.csv"
    "SOX30_HUMAN_Tsuboyama_2023_7JJK.csv"
    "YNZC_BACSU_Tsuboyama_2023_2JVD.csv"
)
NUM_DATASETS=${#DATASETS[@]}

# ------------------------
# SEEDS
# ------------------------
NUM_SEEDS=10
SEED_BASE=42

# ------------------------
# TRAIN SIZES
# ------------------------
TRAIN_SIZES=(20 50 75 100 200 350 500)
NUM_TRAIN_SIZES=${#TRAIN_SIZES[@]}

# ------------------------
# FOUNDATION MODEL
# ------------------------
FOUNDATIONS=(
    "facebook/esm2_t33_650M_UR50D"
)
NUM_FOUNDATIONS=${#FOUNDATIONS[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

TS_IDX=$((TASK_ID % NUM_TRAIN_SIZES))
TASK_ID=$((TASK_ID / NUM_TRAIN_SIZES))

SEED_IDX=$((TASK_ID % NUM_SEEDS))
TASK_ID=$((TASK_ID / NUM_SEEDS))

DATA_IDX=$((TASK_ID % NUM_DATASETS))
FOUND_IDX=$((TASK_ID / NUM_DATASETS))

DATAFILE=${DATASETS[$DATA_IDX]}
SEED=$((SEED_BASE + SEED_IDX))
TRAIN_SIZE=${TRAIN_SIZES[$TS_IDX]}
FOUND=${FOUNDATIONS[$FOUND_IDX]}

echo ">> Representation: $REPRESENTATION"
echo ">> Foundation:     $FOUND"
echo ">> Dataset:        $DATAFILE"
echo ">> Seed:           $SEED"
echo ">> Train size:     $TRAIN_SIZE"

# ------------------------
# PATHS
# ------------------------
DATA="${DATA_DIR}/proteingym_dms/processed/substitutions/$DATAFILE"

OUT="${CODE_DIR}/benchmark/simulate_experiments/simulation_output/substitutions"

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
        --foundation_model "$FOUND"