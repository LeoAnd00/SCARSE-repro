#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-32
#SBATCH -p vera
#SBATCH -t 48:00:00
#SBATCH -C ZEN4
#SBATCH -n 1
#SBATCH --array=0-129   # 13 datasets × 10 seeds = 130 jobs

ml purge

# HuggingFace cache
export HF_HOME="/cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

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
  CellPPD
  ToxinPred3
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
  cellppd/processed/cellppd
  toxinpred3/processed/toxinpred3
)

NUM_DATASETS=13
NUM_SEEDS=10

# Global array index
GLOBAL_IDX=${SLURM_ARRAY_TASK_ID}

# Compute dataset index and seed index
DATASET_IDX=$(( GLOBAL_IDX / NUM_SEEDS ))
SEED_IDX=$(( GLOBAL_IDX % NUM_SEEDS ))

# Set actual random seed (42 → 51)
RANDOM_SEED=$(( 42 + SEED_IDX ))

DATA_PATH="${DATA_DIR}/${file_names[$DATASET_IDX]}.csv"
DATASET_NAME="${dataset_names[$DATASET_IDX]}"

echo "Running dataset ${DATASET_NAME}"
echo "Using data file ${DATA_PATH}"
echo "Seed index ${SEED_IDX}"
echo "Random seed ${RANDOM_SEED}"

# Detect classification datasets
if [[ "$DATASET_NAME" == "CellPPD" || "$DATASET_NAME" == "ToxinPred3" ]]; then

  echo "Running in classification mode"

  apptainer exec /cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/env.sif \
  python ../simulation.py \
    --data_paths "${DATA_PATH}" \
    --dataset_names "${DATASET_NAME}" \
    --initial_train_size 20 \
    --random_seed ${RANDOM_SEED} \
    --model_name "facebook/esm2_t33_650M_UR50D" \
    --max_num_samp 200 \
    --new_samp_per_step 20 \
    --n_trials 100 \
    --n_seeds 1 \
    --classification \
    --score_col "classes"

else

  echo "Running in regression mode"

  apptainer exec /cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/env.sif \
  python ../simulation.py \
    --data_paths "${DATA_PATH}" \
    --dataset_names "${DATASET_NAME}" \
    --initial_train_size 20 \
    --random_seed ${RANDOM_SEED} \
    --model_name "facebook/esm2_t33_650M_UR50D" \
    --max_num_samp 200 \
    --new_samp_per_step 20 \
    --n_trials 100 \
    --n_seeds 1

fi