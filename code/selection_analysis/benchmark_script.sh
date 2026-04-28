#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-32
#SBATCH -p vera
#SBATCH -t 24:00:00
#SBATCH -C ZEN4
#SBATCH -n 1
#SBATCH --array=0-989

ml purge

# HuggingFace cache directory
export HF_HOME="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

# ------------------------
# DATASETS
# ------------------------
# Base data directory
DATA_DIR="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/data"

# Dataset names (13 entries)
dataset_names=(
  A0A247D711_LISMN
  DN7A_SACS2
  ENVZ_ECOLI
  FKBP3_HUMAN
  MAFG_MOUSE_sub
  POLG_PESV_sub
  SBI_STAAM
  SDA_BACSU
  SOX30_HUMAN
  YNZC_BACSU_sub
  AMP_Ecoli
)

# Corresponding file names
file_names=(
  proteingym_dms/processed/substitutions/A0A247D711_LISMN_Stadelmann_2021
  proteingym_dms/processed/substitutions/DN7A_SACS2_Tsuboyama_2023_1JIC
  proteingym_dms/processed/substitutions/ENVZ_ECOLI_Ghose_2023
  proteingym_dms/processed/substitutions/FKBP3_HUMAN_Tsuboyama_2023_2KFV
  proteingym_dms/processed/substitutions/MAFG_MOUSE_Tsuboyama_2023_1K1V
  proteingym_dms/processed/substitutions/POLG_PESV_Tsuboyama_2023_2MXD
  proteingym_dms/processed/substitutions/SBI_STAAM_Tsuboyama_2023_2JVG
  proteingym_dms/processed/substitutions/SDA_BACSU_Tsuboyama_2023_1PV0
  proteingym_dms/processed/substitutions/SOX30_HUMAN_Tsuboyama_2023_7JJK
  proteingym_dms/processed/substitutions/YNZC_BACSU_Tsuboyama_2023_2JVD
  mbc/processed/EC
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

SEED=$((SEED_BASE + SEED_IDX))
TRAIN_SIZE=${TRAIN_SIZES[$TS_IDX]}
FOUND=${FOUNDATIONS[$FOUND_IDX]}

echo ">> Foundation: $FOUND"
echo ">> Dataset:    $DATAFILE"
echo ">> Seed:       $SEED"
echo ">> Train size: $TRAIN_SIZE"

# ------------------------
# PATHS
# ------------------------
DATA="${DATA_DIR}/${file_names[$DATA_IDX]}.csv"

OUT="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/code/selection_analysis/simulation_output"

# ------------------------
# RUN
# ------------------------
apptainer exec \
    /cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/env.sif \
    python optimize_models.py \
        "$DATA" "$OUT" \
        --n_seeds $SEED \
        --train_sizes $TRAIN_SIZE
