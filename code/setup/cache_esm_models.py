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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=str, default=None,
                        help="Cache a single checkpoint by size, e.g. 8M, 35M, 150M, 650M.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Do not download; only load what is already cached.")
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

    variants = dict(ESM2_VARIANTS)
    if args.only:
        variants = {k: v for k, v in ESM2_VARIANTS.items() if v["params"] == args.only}
        if not variants:
            sizes = [v["params"] for v in ESM2_VARIANTS.values()]
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

    print(f"Cache size: {human_size(cache_size(hf_home))}")

    if failures:
        print(f"\nFAILED for {len(failures)} checkpoint(s): {failures}")
        print("Re-run this script; snapshot_download resumes rather than starting over.")
        sys.exit(1)

    print(f"\nAll {len(variants)} checkpoint(s) cached and loadable offline.")
    print("The run scripts set HF_HUB_OFFLINE=1, so no job will contact "
          "huggingface.co from now on.")


if __name__ == "__main__":
    main()
