#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-32
#SBATCH -p vera
#SBATCH -t 48:00:00
#SBATCH -C ZEN4
#SBATCH -n 1
#SBATCH --array=0-399 


ml purge  # Ensure we don't have any conflicting modules loaded

# HuggingFace cache directory
export HF_HOME="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

# List datasets
DATASETS=(
    "PKN1_HUMAN_Tsuboyama_2023_1URF_indels.csv"
    "RPC1_BP434_Tsuboyama_2023_1R69_indels.csv"
    "POLG_PESV_Tsuboyama_2023_2MXD_indels.csv"
    "SDA_BACSU_Tsuboyama_2023_1PV0_indels.csv"
    "RD23A_HUMAN_Tsuboyama_2023_1IFY_indels.csv"
    "MAFG_MOUSE_Tsuboyama_2023_1K1V_indels.csv"
    "SQSTM_MOUSE_Tsuboyama_2023_2RRU_indels.csv"
    "VG08_BPP22_Tsuboyama_2023_2GP8_indels.csv"
    "YNZC_BACSU_Tsuboyama_2023_2JVD_indels.csv"
    "PIN1_HUMAN_Tsuboyama_2023_1I6C_indels.csv"
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
TRAIN_SIZES=(20 40 60 80)
NUM_TRAIN_SIZES=${#TRAIN_SIZES[@]}

# ------------------------
# FOUNDATION MODELS
# (progen2 excluded)
# ------------------------
FOUNDATIONS=(
    "facebook/esm2_t33_650M_UR50D"
)
NUM_FOUNDATIONS=${#FOUNDATIONS[@]}

# ------------------------
# GRID MATH (4D ARRAY)
# ------------------------
# Order: foundation → dataset → seed → train_size
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
DATA="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/data/proteingym_dms/processed/indels/$DATAFILE"

# Outputs
OUT="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/code/benchmark/simulate_experiments/simulation_output/indels"

# ------------------------
# RUN
# ------------------------
apptainer exec \
    /cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/env.sif \
    python ../optimize_models.py \
        "$DATA" "$OUT" \
        --n_seeds $SEED \
        --train_sizes $TRAIN_SIZE \
        --test_sizes 20

