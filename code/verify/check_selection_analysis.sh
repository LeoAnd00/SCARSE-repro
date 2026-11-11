#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-299
#SBATCH -t 00:10:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J sel-check
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts. Submit from this folder
# after a one-off `mkdir -p logs`, or run it straight on a login node with bash.
# ==========================================================================
# Completeness check for the SELECTION ANALYSIS
# (code/selection_analysis, the fine-grained training-size sweep used for the
#  "selecting workflow" evaluation).
#
#   bash check_selection_analysis.sh                      # login node (instant)
#   mkdir -p logs && sbatch check_selection_analysis.sh   # as a batch job
#
# One run writes one folder:
#
#   simulation_output/<dataset>/<tag>/seed_<seed>/train_size_<size>/
#       raw_simulation_summary.csv
#       simulation_all_predictions.csv
#
# (this module's optimize_models.py writes these two files only; timing.csv and
#  split_sequences.csv belong to the model-benchmark module, not here.)
#
# <dataset> is the data file's basename (no extension); <tag> is the
# representation name (esm, physchem, morgan, ngram) — or the ESM2 checkpoint
# short name if you ran esm with a non-default --foundation_model.
#
# Grid per family (fixed by the benchmark_script_*.sh scripts):
#   substitutions : 10 datasets x 10 seeds x 9 train sizes (20..100 step 10)
#   peptides      :  4 datasets x 10 seeds x 9 train sizes
#
# Inspects the tree only (no container, no Python); one CPU and a few seconds
# are plenty, and it is safe to run any time.
# Exit status 0 only if every expected run folder has both CSVs, non-empty.
# ==========================================================================
set -uo pipefail

# ------------------------------------------------------------------
# EDIT THIS ONE LINE for your Berzelius project (same as every run script).
# ------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/proj/berzelius-2026-62/users/${USER:-$(id -un)}/reproducibility_code}"

# Root of the selection-analysis output tree. Defaults to the copy under
# PROJECT_ROOT; override with ROOT=... if your output lives elsewhere.
ROOT="${ROOT:-${PROJECT_ROOT}/code/selection_analysis/simulation_output}"

# Representations you ran. selection_analysis accepts esm, physchem, morgan, ngram.
# For esm the folder tag is the ESM2 checkpoint short name, not "esm".
TAGS=(esm2_t33_650M_UR50D physchem morgan ngram)

SEED_BASE=42
NUM_SEEDS=10
TRAIN_SIZES=(20 30 40 50 60 70 80 90 100)
MIN_BYTES="${MIN_BYTES:-20}"
REQUIRED_FILES=(raw_simulation_summary.csv simulation_all_predictions.csv)

SUBS_DATASETS=(
  A0A247D711_LISMN_Stadelmann_2021
  DN7A_SACS2_Tsuboyama_2023_1JIC
  ENVZ_ECOLI_Ghose_2023
  FKBP3_HUMAN_Tsuboyama_2023_2KFV
  MAFG_MOUSE_Tsuboyama_2023_1K1V
  POLG_PESV_Tsuboyama_2023_2MXD
  SBI_STAAM_Tsuboyama_2023_2JVG
  SDA_BACSU_Tsuboyama_2023_1PV0
  SOX30_HUMAN_Tsuboyama_2023_7JJK
  YNZC_BACSU_Tsuboyama_2023_2JVD
)
PEPTIDES_DATASETS=(EC_all_40 SA_all_40 PA_all_40 hemopi2_all)

SEEDS=()
for ((i=0; i<NUM_SEEDS; i++)); do SEEDS+=($((SEED_BASE + i))); done

echo "=================================================================="
echo " Selection-analysis completeness check"
echo " output root : ${ROOT}"
echo "=================================================================="
if [ ! -d "$ROOT" ]; then
  echo "!! No simulation_output at ${ROOT}"
  exit 2
fi

total_expected=0
total_missing_folders=0
total_incomplete=0
problem_list=""

check_group() {
  local tag="$1"; local -n ds="$2"; local label="$3"
  local present=0 expected=0
  for d in "${ds[@]}"; do
    for sd in "${SEEDS[@]}"; do
      for sz in "${TRAIN_SIZES[@]}"; do
        expected=$((expected+1)); total_expected=$((total_expected+1))
        local folder="${ROOT}/${d}/${tag}/seed_${sd}/train_size_${sz}"
        if [ ! -d "$folder" ]; then
          total_missing_folders=$((total_missing_folders+1))
          problem_list+="  MISSING FOLDER  ${folder#$ROOT/}"$'\n'
          continue
        fi
        local ok=1
        for f in "${REQUIRED_FILES[@]}"; do
          local p="${folder}/${f}"
          if [ ! -f "$p" ]; then
            ok=0; problem_list+="  MISSING ${f}  ${folder#$ROOT/}"$'\n'
          else
            local szb; szb=$(stat -c%s "$p" 2>/dev/null || echo 0)
            if [ "$szb" -lt "$MIN_BYTES" ]; then
              ok=0; problem_list+="  EMPTY ${f} (${szb}B)  ${folder#$ROOT/}"$'\n'
            fi
          fi
        done
        if [ "$ok" -eq 1 ]; then present=$((present+1)); else total_incomplete=$((total_incomplete+1)); fi
      done
    done
  done
  printf "  %-14s %-24s : %4d / %4d run folders complete\n" "$label" "$tag" "$present" "$expected"
}

for tag in "${TAGS[@]}"; do
  check_group "$tag" SUBS_DATASETS     "substitutions"
  check_group "$tag" PEPTIDES_DATASETS "peptides"
done

echo "------------------------------------------------------------------"
echo " Expected run folders  : ${total_expected}"
echo " Missing folders       : ${total_missing_folders}"
echo " Incomplete folders    : ${total_incomplete}"

if [ -n "$problem_list" ]; then
  echo "------------------------------------------------------------------"
  echo " Problems (first 60 lines):"
  printf '%s' "$problem_list" | head -60
  more=$(printf '%s' "$problem_list" | wc -l)
  [ "$more" -gt 60 ] && echo "  ... and $((more-60)) more"
fi

echo "=================================================================="
if [ "$total_missing_folders" -eq 0 ] && [ "$total_incomplete" -eq 0 ]; then
  echo " ALL SELECTION-ANALYSIS RUNS PRESENT AND HEALTHY."
  exit 0
else
  echo " INCOMPLETE: see the list above for what to re-run."
  echo " (If you ran esm with the default 650M model, the tag is"
  echo "  esm2_t33_650M_UR50D as listed; trim TAGS to the reps you launched.)"
  exit 1
fi
