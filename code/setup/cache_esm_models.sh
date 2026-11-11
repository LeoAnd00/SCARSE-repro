#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-299
#SBATCH -t 02:00:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J esm-cache
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts, so no mkdir inside the
# script can help. Submit from this folder after a one-off `mkdir -p logs`, or
# the job dies before printing anything.

# Download the foundation-model checkpoints into the shared HuggingFace cache,
# ONCE, before submitting any job array. By default this caches the four ESM2
# checkpoints only. To also cache PepBERT (needed for --representation pepbert),
# pass --with_pepbert; every argument is forwarded to cache_esm_models.py.
#
# Easiest is to run it straight on a login node, which definitely has internet
# (recommended, especially for PepBERT, since compute nodes may be offline):
#     bash cache_esm_models.sh                  # ESM2 only
#     bash cache_esm_models.sh --with_pepbert   # ESM2 + PepBERT
#     bash cache_esm_models.sh --pepbert_only   # PepBERT only (ESM2 already cached)
#
# Or submit it and wait for it to finish before submitting anything else
# (the compute node must have outbound internet for the download to work):
#     mkdir -p logs && sbatch cache_esm_models.sh --with_pepbert
#
# Every run script afterwards sets HF_HUB_OFFLINE=1, so no array task ever
# contacts huggingface.co. Without this step, thousands of tasks starting at
# once would each try to download the same 2.5 GB checkpoint: you get rate
# limited, and concurrent writes to one cache directory can leave truncated
# files that fail much later in the run.

set -euo pipefail

# ------------------------------------------------------------------
# EDIT THIS ONE LINE for your Berzelius project
# ------------------------------------------------------------------
PROJECT_ROOT="/proj/berzelius-2026-62/users/${USER}/reproducibility_code"

CONTAINER="${PROJECT_ROOT}/env.sif"

# SLURM runs a copy of this script from its spool directory, so locate the
# python script through PROJECT_ROOT rather than relative to the copy.
SETUP_DIR="${PROJECT_ROOT}/code/setup"
cd "${SETUP_DIR}" || {
    echo "No such directory: ${SETUP_DIR}"
    echo "Check that PROJECT_ROOT above points at your copy of the repo."
    exit 1
}
mkdir -p "${SETUP_DIR}/logs"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

mkdir -p "$HF_HOME"

echo ">> HF_HOME = $HF_HOME"
echo ">> Caching foundation models (args: ${*:-<none> = four ESM2 checkpoints, ~3.2 GB})"
echo ">> Pass --with_pepbert to also cache PepBERT."

if [ -x "$(command -v apptainer)" ] && [ -f "$CONTAINER" ]; then
    apptainer exec "$CONTAINER" python cache_esm_models.py "$@"
else
    # No container available (e.g. running on a login node with a local env)
    python cache_esm_models.py "$@"
fi

echo
echo ">> Done. The cache is now populated; every run script reads it with"
echo ">> HF_HUB_OFFLINE=1 and will not hit the network."