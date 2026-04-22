#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-32
#SBATCH -p vera
#SBATCH -t 48:00:00
#SBATCH -C ZEN4
#SBATCH -n 1
#SBATCH --array=0-699

#jid1=$(sbatch substitutions_esm.sh | awk '{print $4}'); sbatch --dependency=afterok:$jid1 substitutions_descriptors.sh

ml purge

# HuggingFace cache directory
export HF_HOME="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

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

# ------------------------
# PATHS
# ------------------------
DATA="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/data/proteingym_dms/processed/substitutions/$DATAFILE"

OUT="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/code/benchmark/simulate_experiments/simulation_output/substitutions"

# ------------------------
# RUN
# ------------------------
apptainer exec \
    /cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/env.sif \
    python ../optimize_models.py \
        "$DATA" "$OUT" \
        --n_seeds $SEED \
        --train_sizes $TRAIN_SIZE \
        --baseline \
