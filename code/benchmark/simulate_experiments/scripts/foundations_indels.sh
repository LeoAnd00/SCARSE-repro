#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-62
#SBATCH -t 48:00:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -J fnd-indels
#SBATCH --array=0-399
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts, so no mkdir inside the
# script can help. Submit from this folder after a one-off `mkdir -p logs`, or
# the job dies before printing anything.

# ESM2 foundation-model benchmark: which ESM2 variant (up to 650M) gives the
# best downstream accuracy on average across datasets.
#
# The representation is always "esm"; what varies is the checkpoint. One
# checkpoint per submission, passed as the first argument:
#   sbatch foundations_indels.sh esm2_t6_8M_UR50D
#   sbatch foundations_indels.sh esm2_t12_35M_UR50D
#   sbatch foundations_indels.sh esm2_t30_150M_UR50D
#   sbatch foundations_indels.sh esm2_t33_650M_UR50D
#
# Or queue them smallest-first so the cheap ones finish early:
#   jid=""
#   for m in esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D; do
#       if [ -z "$jid" ]; then
#           jid=$(sbatch foundations_indels.sh $m | awk '{print $4}')
#       else
#           jid=$(sbatch --dependency=afterok:$jid foundations_indels.sh $m | awk '{print $4}')
#       fi
#   done
#
# Results land in simulation_output/foundations/indels, separate from the
# representation benchmark, one folder per checkpoint.

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
# ESM2 CHECKPOINT (up to 650M; the 3B and 15B models are out of scope)
# ------------------------
MODEL=${1:-esm2_t33_650M_UR50D}
case "$MODEL" in
    esm2_t6_8M_UR50D|esm2_t12_35M_UR50D|esm2_t30_150M_UR50D|esm2_t33_650M_UR50D) ;;
    *) echo "Unknown ESM2 checkpoint: $MODEL"; \
       echo "Expected one of: esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D"; \
       exit 1 ;;
esac
FOUND="facebook/$MODEL"

# ------------------------
# DATASETS
# ------------------------
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

TASK_ID=$SLURM_ARRAY_TASK_ID

TS_IDX=$((TASK_ID % NUM_TRAIN_SIZES))
TASK_ID=$((TASK_ID / NUM_TRAIN_SIZES))

SEED_IDX=$((TASK_ID % NUM_SEEDS))
TASK_ID=$((TASK_ID / NUM_SEEDS))

DATA_IDX=$((TASK_ID % NUM_DATASETS))

DATAFILE=${DATASETS[$DATA_IDX]}
SEED=$((SEED_BASE + SEED_IDX))
TRAIN_SIZE=${TRAIN_SIZES[$TS_IDX]}

echo ">> Checkpoint:     $FOUND"
echo ">> Dataset:        $DATAFILE"
echo ">> Seed:           $SEED"
echo ">> Train size:     $TRAIN_SIZE"

# ------------------------
# PATHS
# ------------------------
DATA="${DATA_DIR}/proteingym_dms/processed/indels/$DATAFILE"

OUT="${CODE_DIR}/benchmark/simulate_experiments/simulation_output/foundations/indels"

# ------------------------
# RUN
# ------------------------
apptainer exec \
    "$CONTAINER" \
    python optimize_models.py \
        "$DATA" "$OUT" \
        --n_seeds $SEED \
        --train_sizes $TRAIN_SIZE \
        --test_sizes 20 \
        --representation esm \
        --foundation_model "$FOUND"