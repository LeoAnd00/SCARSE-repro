"""
Run peptide/protein fitness prediction simulations using either a foundation
model or one of three baseline representations, with an optimized downstream
regression model.

This script benchmarks the predictive performance of sequence representations
across multiple datasets, training set sizes, and random seeds.

Representations
---------------
Exactly one representation is benchmarked per run, chosen with
``--representation``:

    esm       Mean-pooled ESM2 embeddings (the foundation model)
    physchem  Physicochemical descriptors (modlAMP + composition + HeliQuest)
    morgan    Morgan (circular) fingerprints of the peptide molecule
    ngram     TF-IDF weighted character n-grams, fit on training data only

The task is regression throughout; classification is not part of the
benchmark.

Overview
--------
For each combination of:
    • dataset
    • training size
    • random seed

the pipeline performs:

1. Data loading and preprocessing
2. Train/test split initialization
3. Representation of the sequences
4. Hyperparameter optimization of the downstream model via Optuna
5. Model evaluation using cross-validation and held-out test data
6. Storage of:
       - per-label performance metrics
       - per-sequence predictions
7. Incremental saving of results to disk

Output Structure
----------------
Results are saved incrementally in the following directory layout:

    out_dir/
        dataset_name/
            representation_tag/
                seed_<seed>/
                    train_size_<size>/
                        raw_simulation_summary.csv
                        simulation_all_predictions.csv

``representation_tag`` is the sanitized foundation model name for ``esm``
(e.g. ``esm2_t33_650M_UR50D``) and the baseline name otherwise, so the four
representations never overwrite each other.

Files:
    raw_simulation_summary.csv
        Aggregated performance metrics for each model and label

    simulation_all_predictions.csv
        Per-sequence predictions with ground truth values
"""

import os
import argparse
import pandas as pd
from functions.model_optimization import ModelOptimization, REPRESENTATIONS


def representation_tag(representation, foundation_model_name):
    """Directory-safe name identifying the representation of a run.

    For ESM this is the model name without the organization prefix, so
    different ESM2 sizes stay in separate folders. For the baselines it is
    simply the baseline name.
    """
    if representation == "esm":
        return foundation_model_name.split("/")[-1]
    return representation


def save_results(df, df_all_pred, out_dir):
    """
    Save raw simulation outputs for a single dataset, seed, and training size.

    This function appends results to disk in a structured directory layout
    and ensures incremental accumulation of simulation runs.

    Specifically, it:
        - Saves raw per-run performance summaries
        - Saves per-sequence prediction outputs
        - Appends to existing CSVs if present
        - Preserves reproducibility metadata (dataset, seed, train size)

    Parameters
    ----------
    df : pandas.DataFrame
        Per-run performance results containing metrics for one simulation run.
        Must include columns such as:
            "Dataset", "Seed", "Train Size", "Label", "Model", and metrics.
    df_all_pred : pandas.DataFrame
        Per-sequence predictions produced during the simulation run.
        Must include:
            sequence, prediction, true_value, and metadata columns.
    out_dir : str
        Root directory where simulation outputs will be saved.
    """
    dataset = df["Dataset"].iloc[0]
    seed = df["Seed"].iloc[0]
    tag = df["Representation Tag"].iloc[0]
    train_size = df["Train Size"].iloc[0]
    folder = os.path.join(out_dir, dataset, f"{tag}", f"seed_{seed}", f"train_size_{train_size}")
    os.makedirs(folder, exist_ok=True)

    # Save summary of each run
    input_save_path = os.path.join(folder, "raw_simulation_summary.csv")
    df_raw = df.drop(columns=["Representation Tag"])
    if os.path.exists(input_save_path):
        df_raw.to_csv(input_save_path, mode="a", header=False, index=False)
    else:
        df_raw.to_csv(input_save_path, index=False)

    # Save all predictions. The header is decided by whether the predictions
    # file already exists, not the summary file.
    all_pred_save_path = os.path.join(folder, "simulation_all_predictions.csv")
    if os.path.exists(all_pred_save_path):
        df_all_pred.to_csv(all_pred_save_path, mode="a", header=False, index=False)
    else:
        df_all_pred.to_csv(all_pred_save_path, index=False)


def run_simulations(
    data_paths,
    out_dir,
    target_columns=["score"],
    foundation_model_name="facebook/esm2_t33_650M_UR50D",
    train_sizes=[20, 50, 75, 100, 200, 350, 500],
    test_sizes=500,
    r_seeds=[42, 43, 44, 45, 46],
    n_trials=100,
    folds=10,
    representation="esm",
    morgan_radius=2,
    morgan_bits=2048,
    ngram_range=(1, 3),
    levenshtein_split=False
):
    """
    Run full fitness prediction simulations across datasets, training sizes,
    and random seeds.

    Parameters
    ----------
    data_paths : list[str]
        List of paths to CSV datasets. Each dataset must contain:
            • a sequence column
            • one or more target columns

    out_dir : str
        Root directory where all simulation outputs will be stored.

    target_columns : list[str], default=["score"]
        Names of the target columns to predict. Supports multi-target settings.

    foundation_model_name : str, default="facebook/esm2_t33_650M_UR50D"
        Identifier of the pretrained foundation model used to generate
        sequence embeddings. Only used when ``representation="esm"``.

    train_sizes : list[int]
        List of training set sizes to evaluate. Each size defines the number
        of samples used to initialize the training set.

    test_sizes : int, default=500
        Number of samples allocated to the test set.

    r_seeds : list[int]
        Random seeds controlling:
            • train/test splits
            • Optuna sampling
            • model initialization

    n_trials : int, default=100
        Number of Optuna trials used to optimize each downstream model.

    folds : int, default=10
        Number of cross-validation folds used during model evaluation.

    representation : str, default="esm"
        Sequence representation to benchmark. One of "esm", "physchem",
        "morgan", "ngram".

    morgan_radius : int, default=2
        Morgan fingerprint radius. Only used when ``representation="morgan"``.

    morgan_bits : int, default=2048
        Morgan fingerprint length in bits. Only used when
        ``representation="morgan"``.

    ngram_range : tuple[int, int], default=(1, 3)
        Character n-gram range. Only used when ``representation="ngram"``.

    levenshtein_split : bool, default=False
        If True, split data such that training and test sequences are
        dissimilar based on Levenshtein distance. This reduces information
        leakage from similar sequences.

    Returns
    -------
    None
        Results are written to disk incrementally.
    """

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    tag = representation_tag(representation, foundation_model_name)

    for data_path in data_paths:
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]  # get dataset name from file

        for size in train_sizes:
            for seed in r_seeds:
                print(
                    f"\n=== Running simulation: dataset={dataset_name}, "
                    f"representation={representation}, Train Size={size}, seed={seed} ==="
                )

                sim = ModelOptimization(
                    data_path=data_path,
                    initial_train_size=size,
                    initial_test_size=test_sizes,
                    random_seed=seed,
                    representation=representation,
                    model_name=foundation_model_name,
                    morgan_radius=morgan_radius,
                    morgan_bits=morgan_bits,
                    ngram_range=ngram_range
                )

                # Prep data
                sim.prep_data(score_col=target_columns)
                sim.initialize_test_set(levenshtein_split=levenshtein_split)
                sim.initialize_training_set()

                # Run optimization
                results_df_per_label, all_predictions_per_label = sim.optimize_all_models(
                    random_seed=seed,
                    n_trials=n_trials,
                    folds=folds,
                    optuna_print=False
                )

                # Results for this (dataset, size, seed) run only, so each run
                # is written once into its own folder.
                run_results = []
                run_predictions = []

                # Flatten label results and attach metadata
                for label_idx, df in results_df_per_label.items():
                    df = df.copy()
                    df["Seed"] = seed
                    df["Label"] = label_idx
                    df["Train Size"] = size
                    df["Foundation"] = foundation_model_name if representation == "esm" else "none"
                    df["Dataset"] = dataset_name
                    df["Representation"] = representation
                    df["Representation Tag"] = tag
                    run_results.append(df)

                # Flatten predictions for CSV
                for label_idx, pred_dict in all_predictions_per_label.items():
                    for model_name, predictions in pred_dict.items():
                        for item in predictions:
                            run_predictions.append({
                                "Dataset": dataset_name,
                                "Label": label_idx,
                                "Representation": representation,
                                "Foundation": foundation_model_name if representation == "esm" else "none",
                                "Model": model_name,
                                "Train Size": size,
                                "Seed": seed,
                                "sequence": item["sequence"],
                                "prediction": item["prediction"],
                                "true_value": item["true_value"]
                            })

                save_results(
                    df=pd.concat(run_results, ignore_index=True),
                    df_all_pred=pd.DataFrame(run_predictions),
                    out_dir=out_dir
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sequence representation benchmarks with multiple seeds and training sizes."
    )
    parser.add_argument("data_paths", type=str, help="Path to the input CSV datasets. Seperate paths by ,")
    parser.add_argument("out_dir", type=str, help="Directory to save simulation outputs.")
    parser.add_argument("--target_columns", type=str, nargs="+", default=["score"], help="List of column names of the columns containing targets of interest.")
    parser.add_argument("--foundation_model", type=str, default="facebook/esm2_t33_650M_UR50D", help="HuggingFace ESM2 model. Only used when --representation esm.")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[20, 50, 75, 100, 200, 350, 500], help="List of training set sizes.")
    parser.add_argument("--test_sizes", type=int, default=500, help="Number of data points for testing.")
    parser.add_argument("--n_seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="List of random seeds.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials per model.")
    parser.add_argument("--folds", type=int, default=10, help="Number of outer CV folds.")
    parser.add_argument("--representation", type=str, default="esm", choices=list(REPRESENTATIONS), help="Sequence representation to benchmark.")
    parser.add_argument("--morgan_radius", type=int, default=2, help="Morgan fingerprint radius (--representation morgan).")
    parser.add_argument("--morgan_bits", type=int, default=2048, help="Morgan fingerprint size in bits (--representation morgan).")
    parser.add_argument("--ngram_range", type=int, nargs=2, default=[1, 3], metavar=("MIN", "MAX"), help="Character n-gram range (--representation ngram).")
    parser.add_argument("--levenshtein_split", action="store_true", help="Whether to split data strategically using levenshtein distance to minimize similarity between training data and testing data")

    args = parser.parse_args()
    data_paths_ = args.data_paths.split(",")

    run_simulations(
        data_paths=data_paths_,
        out_dir=args.out_dir,
        target_columns=args.target_columns,
        foundation_model_name=args.foundation_model,
        train_sizes=args.train_sizes,
        test_sizes=args.test_sizes,
        r_seeds=args.n_seeds,
        n_trials=args.n_trials,
        folds=args.folds,
        representation=args.representation,
        morgan_radius=args.morgan_radius,
        morgan_bits=args.morgan_bits,
        ngram_range=tuple(args.ngram_range),
        levenshtein_split=args.levenshtein_split
    )
