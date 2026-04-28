#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-32
#SBATCH -p vera
#SBATCH -t 48:00:00
#SBATCH -C ZEN4
#SBATCH -n 1
#SBATCH --array=0-69   

#jid1=$(sbatch mbc_esm.sh | awk '{print $4}'); sbatch --dependency=afterok:$jid1 mbc_descriptors.sh

ml purge  # Ensure we don't have any conflicting modules loaded

# HuggingFace cache directory
export HF_HOME="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

# List datasets
DATASETS=(
    "EC.csv"
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

echo ">> Foundation: $FOUND"
echo ">> Dataset:    $DATAFILE"
echo ">> Seed:       $SEED"
echo ">> Train size: $TRAIN_SIZE"

# Construct dataset path
DATA="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/data/mbc/processed/$DATAFILE"

# Outputs
OUT="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/code/benchmark/simulate_experiments/simulation_output/mbc"

# ------------------------
# RUN
# ------------------------
apptainer exec \
    /cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/env.sif \
    python ../optimize_models.py \
        "$DATA" "$OUT" \
        --n_seeds $SEED \
        --train_sizes $TRAIN_SIZE \
        --levenshtein_split \
        --baseline

