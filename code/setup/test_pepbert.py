#!/usr/bin/env python3
"""Standalone smoke test for the PepBERT representation.

Run this ONCE on the cluster (GPU + internet to reach the Hugging Face Hub)
before launching the full PepBERT benchmark / active-learning sweep. It confirms
that:

  * the custom PepBERT model, config and tokenizer download and load,
  * embeddings are produced for a few peptides of different lengths,
  * the output is a finite, fixed-width matrix.

It deliberately imports the SAME SequenceRepresenter the pipelines use, so a
success here means the pipelines can embed with PepBERT too.

Usage
-----
    cd code/workflow_simulation          # any project dir with functions/
    python ../setup/test_pepbert.py

or point PYTHONPATH at a functions/ directory:

    PYTHONPATH=code/workflow_simulation python code/setup/test_pepbert.py
"""

import sys

import numpy as np


def main():
    try:
        from functions.representations import SequenceRepresenter, PEPBERT_DEFAULT_REPO
    except Exception as exc:  # pragma: no cover - environment dependent
        print("ERROR: could not import SequenceRepresenter. Run this from a "
              "project directory that has a functions/ package next to it "
              "(e.g. code/workflow_simulation).", file=sys.stderr)
        raise

    print(f"Testing PepBERT representation with repo: {PEPBERT_DEFAULT_REPO}")

    rep = SequenceRepresenter(representation="pepbert", random_seed=42)
    rep.load_model()

    seqs = [
        "GLPALISWIKRKRL",             # 14-mer
        "KRKGFLGI",                   # 8-mer
        "ILPILSLIGGLL",               # 12-mer
        "KLKLKLKLKLKLKLKLKLKLKLKLKL", # 26-mer
    ]

    X = rep.transform(seqs)

    print(f"Embedded {len(seqs)} sequences.")
    print(f"Embedding matrix shape: {X.shape}  (n_sequences, embedding_dim)")
    print(f"dtype: {X.dtype}")

    assert X.shape[0] == len(seqs), "wrong number of rows"
    assert X.ndim == 2, "embeddings must be 2D"
    assert np.isfinite(X).all(), "embeddings contain NaN/inf"
    # Different sequences should not collapse to identical vectors.
    assert not np.allclose(X[0], X[1]), "distinct sequences gave identical embeddings"

    print("\nPepBERT smoke test PASSED. The pipelines can now be run with "
          "--representation pepbert.")


if __name__ == "__main__":
    main()
