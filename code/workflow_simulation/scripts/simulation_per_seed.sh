#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-62
#SBATCH -t 48:00:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -J al-sim
#SBATCH --array=0-139   # 14 datasets x 10 seeds = 140 jobs
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts, so no mkdir inside the
# script can help. Submit from this folder after a one-off `mkdir -p logs`, or
# the job dies before printing anything.

# Active-learning workflow simulation.
#
# One representation and one acquisition strategy per submission, passed as the
# first and second arguments:
#   sbatch simulation_per_seed.sh esm greedy
#   sbatch simulation_per_seed.sh esm mixed
#   sbatch simulation_per_seed.sh physchem greedy
#   ... (4 representations x 2 strategies = 8 submissions)
#
# The strategies are:
#   greedy - acquire the 20 highest-predicted sequences each round
#   mixed  - acquire 14 highest-predicted + 6 lowest-predicted each round
#            (70/30, set by --explore_frac below), a more exploratory approach
#
# Or queue them all one after another:
#   jid=""
#   for rep in esm physchem morgan ngram; do
#     for acq in greedy mixed; do
#       dep=""; [ -n "$jid" ] && dep="--dependency=afterok:$jid"
#       jid=$(sbatch $dep simulation_per_seed.sh $rep $acq | awk '{print $4}')
#     done
#   done
#
# Every submission appends to the same simulation_results/df_all.csv, tagged by
# the "Representation" and "Acquisition" columns.

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
MODULE_DIR="${CODE_DIR}/workflow_simulation"
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

# Write Python's prints to the log straight away instead of in large chunks,
# so the log always shows which round a running job is in.
export PYTHONUNBUFFERED=1

# ------------------------------------
# REPRESENTATION AND SELECTION STRATEGY
# ------------------------------------
REPRESENTATION=${1:-esm}
case "$REPRESENTATION" in
    esm|physchem|morgan|ngram) ;;
    *) echo "Unknown representation: $REPRESENTATION (expected esm, physchem, morgan or ngram)"; exit 1 ;;
esac

ACQUISITION=${2:-greedy}
case "$ACQUISITION" in
    greedy|mixed) ;;
    *) echo "Unknown acquisition strategy: $ACQUISITION (expected greedy or mixed)"; exit 1 ;;
esac

# Fraction of each batch taken from the lowest-predicted sequences when
# ACQUISITION=mixed. 0.3 of a 20-peptide batch is 14 best + 6 worst.
EXPLORE_FRAC=0.3

# Run-time limits for the GPR hyperparameter search (seconds, 0 = no limit).
# TRIAL_TIMEOUT: one Optuna trial (all CV folds); a trial over the limit is
#                discarded, and the log names its kernel and alpha.
# ROUND_TIMEOUT: the whole Optuna search of one acquisition round; when reached
#                the best trial so far is used.
# Neither changes a result unless it is reached, and the log says when it is.
TRIAL_TIMEOUT=600
ROUND_TIMEOUT=3600

# Dataset names (14 entries)
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
  Ecoli_AMPs
  Saureus_AMPs
  Paeruginosa_AMPs
  HemoPI2
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
  ecoli_amps/processed/EC_all_40
  saureus_amps/processed/SA_all_40
  paeruginosa_amps/processed/PA_all_40
  hemopi2/processed/hemopi2_all
)

# Optimisation direction per dataset (same order as dataset_names above).
# Almost every endpoint is "higher is better" (maximize). HemoPI2 is the one
# exception: its score is -log(HC50), where a LOWER value means less haemolytic
# and thus more desirable, so it is optimised as a minimize objective. The score
# is negated once inside prep_data, so selection and the "true top 10%" set stay
# consistent. Keep this array aligned with dataset_names if you add a dataset.
directions=(
  maximize   # A0A247D711_LISMN
  maximize   # DN7A_SACS2
  maximize   # ENVZ_ECOLI
  maximize   # FKBP3_HUMAN
  maximize   # MAFG_MOUSE_sub
  maximize   # POLG_PESV_sub
  maximize   # SBI_STAAM
  maximize   # SDA_BACSU
  maximize   # SOX30_HUMAN
  maximize   # YNZC_BACSU_sub
  maximize   # Ecoli_AMPs
  maximize   # Saureus_AMPs
  maximize   # Paeruginosa_AMPs
  minimize   # HemoPI2  (-log(HC50): lower is less haemolytic, i.e. better)
)

NUM_DATASETS=${#dataset_names[@]}
NUM_SEEDS=10

# Guard against the parallel arrays drifting out of alignment.
if [ "${#file_names[@]}" -ne "$NUM_DATASETS" ] || [ "${#directions[@]}" -ne "$NUM_DATASETS" ]; then
    echo "dataset_names (${NUM_DATASETS}), file_names (${#file_names[@]}) and directions (${#directions[@]}) must be the same length"
    exit 1
fi

# Global array index
# Outside SLURM - a manual `bash <script> <args>` run - there is no array
# index. Default to task 0 and say so, rather than silently evaluating an empty
# variable as 0. One task = one (dataset, seed, train size) configuration.
GLOBAL_IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo ">> SLURM_ARRAY_TASK_ID is not set: running array task $GLOBAL_IDX only."
    echo ">> Pick another with e.g. SLURM_ARRAY_TASK_ID=137 bash $(basename "$0") $*"
fi

# Compute dataset index and seed index
DATASET_IDX=$(( GLOBAL_IDX / NUM_SEEDS ))
SEED_IDX=$(( GLOBAL_IDX % NUM_SEEDS ))

# Set actual random seed (42 -> 51)
RANDOM_SEED=$(( 42 + SEED_IDX ))

DATA_PATH="${DATA_DIR}/${file_names[$DATASET_IDX]}.csv"
DATASET_NAME="${dataset_names[$DATASET_IDX]}"
DIRECTION="${directions[$DATASET_IDX]}"

echo "Representation ${REPRESENTATION}"
if [ "$ACQUISITION" = "mixed" ]; then
    echo "Acquisition ${ACQUISITION} (explore_frac ${EXPLORE_FRAC})"
else
    echo "Acquisition ${ACQUISITION} (top-k only; explore_frac does not apply)"
fi
echo "Running dataset ${DATASET_NAME}"
echo "Using data file ${DATA_PATH}"
echo "Optimisation direction ${DIRECTION}"
echo "Seed index ${SEED_IDX}"
echo "Random seed ${RANDOM_SEED}"

apptainer exec "$CONTAINER" \
python simulation.py \
  --data_paths "${DATA_PATH}" \
  --dataset_names "${DATASET_NAME}" \
  --initial_train_size 20 \
  --random_seed ${RANDOM_SEED} \
  --representation "${REPRESENTATION}" \
  --acquisition "${ACQUISITION}" \
  --direction "${DIRECTION}" \
  --explore_frac ${EXPLORE_FRAC} \
  --model_name "facebook/esm2_t33_650M_UR50D" \
  --max_num_samp 200 \
  --new_samp_per_step 20 \
  --n_trials 100 \
  --n_seeds 1 \
  --trial_timeout ${TRIAL_TIMEOUT} \
  --round_timeout ${ROUND_TIMEOUT} \
  --results_dir "./simulation_results"