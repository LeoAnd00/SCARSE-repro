#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-299
#SBATCH -t 00:10:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J al-check
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts. Submit from this folder
# after a one-off `mkdir -p logs`, or run it straight on a login node with bash.
# ==========================================================================
# Completeness check for the ACTIVE-LEARNING workflow simulation.
#
#   bash check_active_learning.sh            # on a login node (instant)
#   mkdir -p logs && sbatch check_active_learning.sh   # as a batch job
#
# It verifies that every expected per-run file exists and is non-trivial, for
# every (representation x acquisition x dataset x seed) you asked to run. One
# array task writes exactly two files:
#
#   simulation_results/runs/{rep}__{acq}__{dataset}__seed{seed}.csv
#   simulation_results/sequences/{rep}__{acq}__{dataset}__seed{seed}.csv
#
# so the full expected count is
#   |reps| x |acqs| x |datasets| x |seeds|   in EACH of runs/ and sequences/.
#
# Nothing here re-runs or imports the pipeline; it only stats the output tree
# (no container, no Python), so it is safe to run on a login node at any time,
# including while jobs are still finishing (it simply reports what is not there
# yet). One CPU and a few seconds are plenty.
#
# Exit status is 0 only if everything expected is present and healthy.
# ==========================================================================
set -uo pipefail

# ------------------------------------------------------------------
# EDIT THIS ONE LINE for your Berzelius project (same as every run script).
# Everything is located from PROJECT_ROOT, so the check works the same whether
# launched with sbatch (which runs a copy from the spool dir) or with bash.
# ------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/proj/berzelius-2026-62/users/${USER:-$(id -un)}/reproducibility_code}"

# ------------------------------------------------------------------
# EDIT THESE to match what you launched.
# ------------------------------------------------------------------
# Path to the active-learning results directory (contains runs/ and sequences/).
# Defaults to the copy under PROJECT_ROOT; override with RESULTS_DIR=... if your
# results live elsewhere.
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/code/workflow_simulation/simulation_results}"

# Representations and acquisition strategies you submitted.
REPRESENTATIONS=(esm pepbert physchem morgan ngram)
ACQUISITIONS=(greedy mixed ucb)

# Seeds: the sweep uses 10 seeds, 42..51.
SEED_BASE=42
NUM_SEEDS=10

# Datasets: the AL sweep's 14 datasets, as the FILE BASENAMES that become the
# "Dataset" value in the CSV (os.path on the data file). These are exactly the
# dataset_names array in simulation_per_seed.sh.
DATASETS=(
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

# A run CSV smaller than this many bytes is treated as damaged/empty.
MIN_BYTES="${MIN_BYTES:-200}"
# ------------------------------------------------------------------

RUNS_DIR="${RESULTS_DIR}/runs"
SEQ_DIR="${RESULTS_DIR}/sequences"

# same sanitiser as run_file_name() in model_optimization.py
safe() { printf '%s' "$1" | sed 's/[^[:alnum:].-]/_/g'; }

echo "=================================================================="
echo " Active-learning completeness check"
echo " results dir : ${RESULTS_DIR}"
echo "=================================================================="

if [ ! -d "$RUNS_DIR" ]; then
    echo "!! No runs/ directory at ${RUNS_DIR}"
    echo "!! Either RESULTS_DIR is wrong, or nothing has finished yet."
    exit 2
fi

SEEDS=()
for ((i=0; i<NUM_SEEDS; i++)); do SEEDS+=($((SEED_BASE + i))); done

expected=0
miss_runs=0
miss_seqs=0
small_runs=0
small_seqs=0
missing_list=""

for rep in "${REPRESENTATIONS[@]}"; do
  for acq in "${ACQUISITIONS[@]}"; do
    present=0
    for ds in "${DATASETS[@]}"; do
      for sd in "${SEEDS[@]}"; do
        expected=$((expected+1))
        fn="$(safe "$rep")__$(safe "$acq")__$(safe "$ds")__seed$(safe "$sd").csv"
        rp="${RUNS_DIR}/${fn}"
        sp="${SEQ_DIR}/${fn}"

        if [ ! -f "$rp" ]; then
          miss_runs=$((miss_runs+1))
          missing_list+="  [runs] MISSING  ${fn}"$'\n'
        else
          present=$((present+1))
          sz=$(stat -c%s "$rp" 2>/dev/null || echo 0)
          if [ "$sz" -lt "$MIN_BYTES" ]; then
            small_runs=$((small_runs+1))
            missing_list+="  [runs] TINY(${sz}B) ${fn}"$'\n'
          fi
        fi

        if [ ! -f "$sp" ]; then
          miss_seqs=$((miss_seqs+1))
          missing_list+="  [sequences] MISSING  ${fn}"$'\n'
        else
          sz=$(stat -c%s "$sp" 2>/dev/null || echo 0)
          if [ "$sz" -lt "$MIN_BYTES" ]; then
            small_seqs=$((small_seqs+1))
            missing_list+="  [sequences] TINY(${sz}B) ${fn}"$'\n'
          fi
        fi
      done
    done
    printf "  %-8s %-7s : %3d / %3d runs present\n" \
      "$rep" "$acq" "$present" "$((${#DATASETS[@]} * NUM_SEEDS))"
  done
done

echo "------------------------------------------------------------------"
echo " Expected per-run files : ${expected}  (in EACH of runs/ and sequences/)"
echo " runs/      missing     : ${miss_runs}"
echo " runs/      too small   : ${small_runs}"
echo " sequences/ missing     : ${miss_seqs}"
echo " sequences/ too small   : ${small_seqs}"

# Flag any unexpected extra files (typo in a dataset/rep name, stray re-run).
extra_runs=$(find "$RUNS_DIR" -maxdepth 1 -name '*.csv' | wc -l)
echo " runs/ files on disk    : ${extra_runs}  (expected ${expected})"
if [ -d "$SEQ_DIR" ]; then
  extra_seqs=$(find "$SEQ_DIR" -maxdepth 1 -name '*.csv' | wc -l)
  echo " sequences/ files on disk: ${extra_seqs}  (expected ${expected})"
fi

if [ -n "$missing_list" ]; then
  echo "------------------------------------------------------------------"
  echo " Problems (first 60 lines):"
  printf '%s' "$missing_list" | head -60
  more=$(printf '%s' "$missing_list" | wc -l)
  [ "$more" -gt 60 ] && echo "  ... and $((more-60)) more"
fi

echo "=================================================================="
if [ "$miss_runs" -eq 0 ] && [ "$miss_seqs" -eq 0 ] \
   && [ "$small_runs" -eq 0 ] && [ "$small_seqs" -eq 0 ]; then
  echo " ALL ACTIVE-LEARNING RUNS PRESENT AND HEALTHY."
  exit 0
else
  echo " INCOMPLETE: see the list above for what to re-run."
  exit 1
fi
