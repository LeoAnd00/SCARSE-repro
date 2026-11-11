#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-299
#SBATCH -t 00:10:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J bench-check
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts. Submit from this folder
# after a one-off `mkdir -p logs`, or run it straight on a login node with bash.
# ==========================================================================
# Completeness check for the MODEL BENCHMARK (code/benchmark/simulate_experiments).
#
#   bash check_benchmark.sh                      # on a login node (instant)
#   mkdir -p logs && sbatch check_benchmark.sh   # as a batch job
#
# The benchmark has two parts, each writing one folder per run:
#
#   simulation_output/<dataset>/<tag>/seed_<seed>/train_size_<size>/
#       raw_simulation_summary.csv
#       simulation_all_predictions.csv
#       split_sequences.csv
#       timing.csv
#
# where <dataset> is the data file's basename (no extension), and <tag> is:
#   * the ESM2 checkpoint short name for the FOUNDATION sweep
#       (esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D),
#       written under simulation_output/foundations/{substitutions,indels,peptides}/
#   * the representation name for the BASELINE sweep (esm, pepbert, physchem,
#       morgan, ngram), written under simulation_output/{substitutions,indels,peptides}/
#
# Grids (datasets x seeds x train_sizes) per family are fixed by the .sh scripts
# and encoded below. This script only inspects the tree, so it is safe to run
# any time, including while jobs are still running.
#
# Inspects the tree only (no container, no Python); one CPU and a few seconds
# are plenty. Exit status is 0 only if every expected run folder has all four
# CSVs, non-empty.
# ==========================================================================
set -uo pipefail

# ------------------------------------------------------------------
# EDIT THIS ONE LINE for your Berzelius project (same as every run script).
# ------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/proj/berzelius-2026-62/users/${USER:-$(id -un)}/reproducibility_code}"

# Root of the benchmark output tree. Defaults to the copy under PROJECT_ROOT;
# override with ROOT=... if your output lives elsewhere.
ROOT="${ROOT:-${PROJECT_ROOT}/code/benchmark/simulate_experiments/simulation_output}"

# Which tags to check. Edit to the subset you actually ran.
FOUNDATION_TAGS=(esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D)
BASELINE_TAGS=(esm pepbert physchem morgan ngram)

SEED_BASE=42
NUM_SEEDS=10
MIN_BYTES="${MIN_BYTES:-20}"
REQUIRED_FILES=(raw_simulation_summary.csv simulation_all_predictions.csv split_sequences.csv timing.csv)

# --- dataset basenames per family (match the DATASETS arrays in the scripts) ---
SUBS_DATASETS=(
  ENVZ_ECOLI_Ghose_2023
  A0A247D711_LISMN_Stadelmann_2021
  DN7A_SACS2_Tsuboyama_2023_1JIC
  FKBP3_HUMAN_Tsuboyama_2023_2KFV
  MAFG_MOUSE_Tsuboyama_2023_1K1V
  POLG_PESV_Tsuboyama_2023_2MXD
  SBI_STAAM_Tsuboyama_2023_2JVG
  SDA_BACSU_Tsuboyama_2023_1PV0
  SOX30_HUMAN_Tsuboyama_2023_7JJK
  YNZC_BACSU_Tsuboyama_2023_2JVD
)
INDELS_DATASETS=(
  PKN1_HUMAN_Tsuboyama_2023_1URF_indels
  RPC1_BP434_Tsuboyama_2023_1R69_indels
  POLG_PESV_Tsuboyama_2023_2MXD_indels
  SDA_BACSU_Tsuboyama_2023_1PV0_indels
  RD23A_HUMAN_Tsuboyama_2023_1IFY_indels
  MAFG_MOUSE_Tsuboyama_2023_1K1V_indels
  SQSTM_MOUSE_Tsuboyama_2023_2RRU_indels
  VG08_BPP22_Tsuboyama_2023_2GP8_indels
  YNZC_BACSU_Tsuboyama_2023_2JVD_indels
  PIN1_HUMAN_Tsuboyama_2023_1I6C_indels
)
PEPTIDES_DATASETS=(EC_all_40 SA_all_40 PA_all_40 hemopi2_all)

# --- train sizes per family (match TRAIN_SIZES in the scripts) ---
SUBS_SIZES=(20 50 75 100 200 350 500)
INDELS_SIZES=(20 40 60 80)
PEPTIDES_SIZES=(20 50 75 100 200 350 500)

SEEDS=()
for ((i=0; i<NUM_SEEDS; i++)); do SEEDS+=($((SEED_BASE + i))); done

echo "=================================================================="
echo " Model-benchmark completeness check"
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

# check_group <base_dir> <tag> : one (family-base, tag) over its datasets/sizes
check_group() {
  local base="$1" tag="$2"
  local -n ds="$3" ; local -n sizes="$4"
  local label="$5"
  local present=0 expected=0
  for d in "${ds[@]}"; do
    for sd in "${SEEDS[@]}"; do
      for sz in "${sizes[@]}"; do
        expected=$((expected+1)); total_expected=$((total_expected+1))
        local folder="${base}/${d}/${tag}/seed_${sd}/train_size_${sz}"
        if [ ! -d "$folder" ]; then
          total_missing_folders=$((total_missing_folders+1))
          problem_list+="  MISSING FOLDER  ${folder#$ROOT/}"$'\n'
          continue
        fi
        local ok=1
        for f in "${REQUIRED_FILES[@]}"; do
          local p="${folder}/${f}"
          if [ ! -f "$p" ]; then
            ok=0
            problem_list+="  MISSING ${f}  ${folder#$ROOT/}"$'\n'
          else
            local szb; szb=$(stat -c%s "$p" 2>/dev/null || echo 0)
            if [ "$szb" -lt "$MIN_BYTES" ]; then
              ok=0
              problem_list+="  EMPTY ${f} (${szb}B)  ${folder#$ROOT/}"$'\n'
            fi
          fi
        done
        if [ "$ok" -eq 1 ]; then present=$((present+1)); else total_incomplete=$((total_incomplete+1)); fi
      done
    done
  done
  printf "  %-34s %-26s : %4d / %4d run folders complete\n" "$label" "$tag" "$present" "$expected"
}

echo ">> FOUNDATION sweep (simulation_output/foundations/<family>/<esm2_tag>)"
for tag in "${FOUNDATION_TAGS[@]}"; do
  check_group "${ROOT}/foundations/substitutions" "$tag" SUBS_DATASETS     SUBS_SIZES     "foundations/substitutions"
  check_group "${ROOT}/foundations/indels"        "$tag" INDELS_DATASETS   INDELS_SIZES   "foundations/indels"
  check_group "${ROOT}/foundations/peptides"      "$tag" PEPTIDES_DATASETS PEPTIDES_SIZES "foundations/peptides"
done

echo ">> BASELINE sweep (simulation_output/<family>/<representation>)"
for tag in "${BASELINE_TAGS[@]}"; do
  check_group "${ROOT}/substitutions" "$tag" SUBS_DATASETS     SUBS_SIZES     "substitutions"
  check_group "${ROOT}/indels"        "$tag" INDELS_DATASETS   INDELS_SIZES   "indels"
  check_group "${ROOT}/peptides"      "$tag" PEPTIDES_DATASETS PEPTIDES_SIZES "peptides"
done

echo "------------------------------------------------------------------"
echo " Expected run folders  : ${total_expected}"
echo " Missing folders       : ${total_missing_folders}"
echo " Incomplete folders    : ${total_incomplete}  (folder present but a CSV missing/empty)"

if [ -n "$problem_list" ]; then
  echo "------------------------------------------------------------------"
  echo " Problems (first 60 lines):"
  printf '%s' "$problem_list" | head -60
  more=$(printf '%s' "$problem_list" | wc -l)
  [ "$more" -gt 60 ] && echo "  ... and $((more-60)) more"
fi

echo "=================================================================="
if [ "$total_missing_folders" -eq 0 ] && [ "$total_incomplete" -eq 0 ]; then
  echo " ALL BENCHMARK RUNS PRESENT AND HEALTHY."
  exit 0
else
  echo " INCOMPLETE: see the list above for what to re-run."
  echo " (If you did not run every tag/family listed, trim the *_TAGS arrays at"
  echo "  the top of this script to the subset you launched.)"
  exit 1
fi
