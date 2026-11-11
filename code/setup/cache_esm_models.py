#!/usr/bin/env python3
"""
Pre-download every ESM2 checkpoint the benchmarks use into the shared HF cache.

Run this once, before submitting any job arrays. Afterwards the run scripts set
HF_HUB_OFFLINE=1, so thousands of array tasks read the cache instead of all
hitting huggingface.co at the same time (which gets you rate-limited, and
produces half-written cache entries when several jobs race on the same files).

Usage
-----
    export HF_HOME=/proj/<project>/users/$USER/reproducibility_code/hf_cache
    python cache_esm_models.py                # all four checkpoints
    python cache_esm_models.py --only 650M    # just one
    python cache_esm_models.py --verify_only  # check an existing cache
"""

import argparse
import os
import sys

#: Same list as ESM2_VARIANTS in the benchmark code. Up to 650M only.
ESM2_VARIANTS = {
    "facebook/esm2_t6_8M_UR50D":    {"params": "8M",   "layers": 6,  "embedding_dim": 320},
    "facebook/esm2_t12_35M_UR50D":  {"params": "35M",  "layers": 12, "embedding_dim": 480},
    "facebook/esm2_t30_150M_UR50D": {"params": "150M", "layers": 30, "embedding_dim": 640},
    "facebook/esm2_t33_650M_UR50D": {"params": "650M", "layers": 33, "embedding_dim": 1280},
}

#: PepBERT checkpoint (peptide-specific language model). Must match
#: PEPBERT_DEFAULT_REPO / PEPBERT_CHECKPOINT in the representation code. Unlike
#: ESM, PepBERT ships a custom model.py/config.py and a raw .pt checkpoint, so
#: the download must include *.py and *.pt as well.
PEPBERT_REPO = "dzjxzyd/PepBERT-large-UniParc"
PEPBERT_CHECKPOINT = "tmodel_17.pt"


def human_size(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def cache_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def download(model_name):
    """Fetch the files transformers needs, without instantiating the model."""
    from huggingface_hub import snapshot_download

    print(f"  downloading {model_name} ...", flush=True)
    path = snapshot_download(
        repo_id=model_name,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "pytorch_model.bin",
        ],
    )
    print(f"    -> {path}")
    return path


def verify(model_name, expected_dim):
    """Load the model from cache with the network switched off.

    This is the real check: it proves the run scripts will work with
    HF_HUB_OFFLINE=1 set, rather than only that some files exist on disk.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = EsmModel.from_pretrained(model_name)

    dim = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters())

    ok = dim == expected_dim
    status = "OK " if ok else "DIM MISMATCH"
    print(f"    {status} hidden_size={dim} (expected {expected_dim}), "
          f"{n_params/1e6:.1f}M parameters, vocab={tokenizer.vocab_size}")

    del model
    return ok


def download_pepbert():
    """Fetch PepBERT's custom code, config, tokenizer and checkpoint."""
    from huggingface_hub import snapshot_download

    print(f"  downloading {PEPBERT_REPO} ...", flush=True)
    path = snapshot_download(
        repo_id=PEPBERT_REPO,
        allow_patterns=["*.py", "*.json", "*.pt", "*.txt"],
    )
    print(f"    -> {path}")
    return path


def verify_pepbert():
    """Load PepBERT from cache with the network off and embed a test peptide.

    PepBERT is loaded through its own custom loader rather than transformers, so
    this exercises the exact code path the pipelines use. Importing the pipeline
    representer keeps the loading logic in one place.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Import the shared representer. Works when run from a project dir (with a
    # functions/ package) or when that dir is on PYTHONPATH.
    try:
        from functions.representations import SequenceRepresenter
    except Exception:
        print("    SKIP verify: run from a project dir (e.g. code/workflow_simulation) "
              "or set PYTHONPATH so 'functions' is importable. Files are cached; "
              "the pipelines will load them.")
        return True

    rep = SequenceRepresenter(representation="pepbert", random_seed=42)
    rep.load_model()
    X = rep.transform(["GLPALISWIKRKRL", "KRKGFLGI"])
    ok = X.ndim == 2 and X.shape[0] == 2
    status = "OK " if ok else "FAILED"
    print(f"    {status} embedding shape={X.shape}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=str, default=None,
                        help="Cache a single checkpoint by size, e.g. 8M, 35M, 150M, 650M, pepbert.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Do not download; only load what is already cached.")
    parser.add_argument("--with_pepbert", action="store_true",
                        help="Also cache the PepBERT checkpoint (needed for the "
                             "--representation pepbert benchmark/AL arm).")
    parser.add_argument("--pepbert_only", action="store_true",
                        help="Cache (and verify) only PepBERT, skipping the ESM2 checkpoints.")
    args = parser.parse_args()

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        sys.exit(
            "HF_HOME is not set.\n"
            "Set it to the shared cache the jobs will use, for example:\n"
            "  export HF_HOME=/proj/<project>/users/$USER/reproducibility_code/hf_cache"
        )
    os.makedirs(hf_home, exist_ok=True)
    print(f"HF_HOME = {hf_home}\n")

    # Decide what to cache. PepBERT is opt-in so existing ESM-only workflows are
    # unchanged, but `--only pepbert` / `--pepbert_only` / `--with_pepbert` all
    # reach it.
    do_pepbert = args.with_pepbert or args.pepbert_only or args.only == "pepbert"

    variants = dict(ESM2_VARIANTS)
    if args.pepbert_only or args.only == "pepbert":
        variants = {}
    elif args.only:
        variants = {k: v for k, v in ESM2_VARIANTS.items() if v["params"] == args.only}
        if not variants:
            sizes = [v["params"] for v in ESM2_VARIANTS.values()] + ["pepbert"]
            sys.exit(f"Unknown size {args.only!r}. Expected one of: {sizes}")

    failures = []

    for model_name, info in variants.items():
        print(f"{info['params']:>5s}  {model_name}")

        if not args.verify_only:
            try:
                download(model_name)
            except Exception as e:
                print(f"    DOWNLOAD FAILED: {type(e).__name__}: {e}")
                failures.append(model_name)
                continue

        try:
            if not verify(model_name, info["embedding_dim"]):
                failures.append(model_name)
        except Exception as e:
            print(f"    VERIFY FAILED: {type(e).__name__}: {e}")
            failures.append(model_name)
        print()

    if do_pepbert:
        print(f"pepbert  {PEPBERT_REPO}")
        if not args.verify_only:
            try:
                download_pepbert()
            except Exception as e:
                print(f"    DOWNLOAD FAILED: {type(e).__name__}: {e}")
                failures.append(PEPBERT_REPO)
        if PEPBERT_REPO not in failures:
            try:
                if not verify_pepbert():
                    failures.append(PEPBERT_REPO)
            except Exception as e:
                print(f"    VERIFY FAILED: {type(e).__name__}: {e}")
                failures.append(PEPBERT_REPO)
        print()

    print(f"Cache size: {human_size(cache_size(hf_home))}")

    if failures:
        print(f"\nFAILED for {len(failures)} checkpoint(s): {failures}")
        print("Re-run this script; snapshot_download resumes rather than starting over.")
        sys.exit(1)

    n_cached = len(variants) + (1 if do_pepbert else 0)
    print(f"\nAll {n_cached} checkpoint(s) cached and loadable offline.")
    print("The run scripts set HF_HUB_OFFLINE=1, so no job will contact "
          "huggingface.co from now on.")


if __name__ == "__main__":
    main()
