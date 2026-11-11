#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-299
#SBATCH -t 00:20:00
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J pepbert-test
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# NOTE: the two log paths above are relative to the directory you run sbatch
# from, and SLURM opens them before this script starts. Submit from this folder
# after a one-off `mkdir -p logs`, or the job dies before printing anything.

# One-off PepBERT smoke test. Run this ONCE after caching PepBERT
# (cache_esm_models.sh --with_pepbert) and before launching the PepBERT sweep.
# It loads PepBERT from the local HuggingFace cache and embeds a few peptides,
# exercising the exact code path the pipelines use. A "PASSED" means every
# --representation pepbert job will work.
#
#     mkdir -p logs && sbatch test_pepbert.sh
#
# Reads from the cache only (no internet needed), so it runs fine on a compute
# node. The test itself works on CPU; it also works on a GPU node if you prefer
# to confirm the GPU path.

set -euo pipefail

module purge > /dev/null 2>&1 || true

# ------------------------------------------------------------------
# EDIT THIS ONE LINE for your Berzelius project (same as the other scripts).
# ------------------------------------------------------------------
PROJECT_ROOT="/proj/berzelius-2026-62/users/${USER}/reproducibility_code"

CONTAINER="${PROJECT_ROOT}/env.sif"
CODE_DIR="${PROJECT_ROOT}/code"

# Run from a project directory that HAS a functions/ package, so the test's
# `from functions.representations import ...` resolves. The script itself lives
# in code/setup (no functions/ there), which is why it must be launched from
# here rather than from its own folder.
MODULE_DIR="${CODE_DIR}/workflow_simulation"
cd "${MODULE_DIR}" || {
    echo "No such directory: ${MODULE_DIR}"
    echo "Check that PROJECT_ROOT above points at your copy of the repo."
    exit 1
}

# Point HuggingFace at the shared cache and keep it offline: PepBERT was cached
# by cache_esm_models.sh --with_pepbert, so nothing should contact the network.
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo ">> HF_HOME     = $HF_HOME"
echo ">> Running from ${MODULE_DIR} (so 'functions' is importable)"
echo ">> PepBERT smoke test ..."

# PYTHONPATH="." makes the functions/ package in this directory importable even
# though the script file lives elsewhere.
if [ -x "$(command -v apptainer)" ] && [ -f "$CONTAINER" ]; then
    PYTHONPATH="." apptainer exec "$CONTAINER" python ../setup/test_pepbert.py
else
    PYTHONPATH="." python ../setup/test_pepbert.py
fi

echo ">> Done."
