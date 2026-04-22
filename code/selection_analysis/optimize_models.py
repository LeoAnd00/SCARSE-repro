import os
import argparse
import pandas as pd
from functions.model_optimization import ModelOptimization


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
    foundation = df["Foundation"].iloc[0]
    train_size = df["Train Size"].iloc[0]
    folder = os.path.join(out_dir, dataset, f"{foundation}", f"seed_{seed}", f"train_size_{train_size}")
    os.makedirs(folder, exist_ok=True)

    # Save summary of each run
    input_save_path = os.path.join(folder, f"raw_simulation_summary.csv")
    df_raw = df.copy()
    if os.path.exists(input_save_path):
        df_raw.to_csv(input_save_path, mode="a", header=False, index=False)
    else:
        df_raw.to_csv(input_save_path, index=False)

    # Save all predictions
    all_pred_save_path = os.path.join(folder, f"simulation_all_predictions.csv")
    if os.path.exists(input_save_path):
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
    levenshtein_split=False
):

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    for data_path in data_paths: 
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]  # get dataset name from file
        dataset_results = []
        all_predictions_list = [] 

        for size in train_sizes:
            for seed in r_seeds:
                print(f"\n=== Running simulation: dataset={dataset_name}, Train Size={size}, seed={seed} ===")

                sim = ModelOptimization(
                    data_path=data_path,
                    initial_train_size=size,
                    initial_test_size=test_sizes,
                    random_seed=seed,
                    model_name=foundation_model_name
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

                # Flatten label results and attach metadata
                for label_idx, df in results_df_per_label.items():
                    df["Seed"] = seed
                    df["Label"] = label_idx
                    df["Train Size"] = size
                    df["Foundation"] = foundation_model_name
                    df["Dataset"] = dataset_name
                    dataset_results.append(df)

                # Flatten predictions for CSV
                for label_idx, pred_dict in all_predictions_per_label.items():
                    for model_name, predictions in pred_dict.items():
                        for item in predictions:
                            all_predictions_list.append({
                                "Dataset": dataset_name,
                                "Label": label_idx,
                                "Foundation": foundation_model_name,
                                "Model": model_name,
                                "Train Size": size,
                                "Seed": seed,
                                "sequence": item["sequence"],  
                                "prediction": item["prediction"],
                                "true_value": item["true_value"]
                            })

            # Combine all results
            dataset_combined_df = pd.concat(dataset_results, ignore_index=True)

            save_results(df=dataset_combined_df, 
                        df_all_pred=pd.DataFrame(all_predictions_list), 
                        out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESM protein sequence simulations with multiple seeds and training sizes.")
    parser.add_argument("data_paths", type=str, help="Path to the input CSV datasets. Seperate paths by ,")
    parser.add_argument("out_dir", type=str, help="Directory to save simulation outputs.")
    parser.add_argument("--target_columns", type=str, nargs="+", default=["score"], help="List of column names of the columns containing targets of interest.")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[20, 50, 75, 100, 200, 350, 500], help="List of training set sizes.")
    parser.add_argument("--test_sizes", type=int, default=500, help="Number of data points for testing.")
    parser.add_argument("--n_seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="List of random seeds.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials per model.")
    parser.add_argument("--folds", type=int, default=10, help="Number of outer CV folds.")
    parser.add_argument("--levenshtein_split", action="store_true", help="Whether to split data strategically using levenshtein distance to minimize similarity between training data and testing data")

    args = parser.parse_args()
    data_paths_ = args.data_paths.split(",") 

    run_simulations(
        data_paths=data_paths_,
        out_dir=args.out_dir,
        target_columns=args.target_columns,
        train_sizes=args.train_sizes,
        test_sizes=args.test_sizes,
        r_seeds=args.n_seeds,
        n_trials=args.n_trials,
        folds=args.folds,
        levenshtein_split=args.levenshtein_split,
    )
