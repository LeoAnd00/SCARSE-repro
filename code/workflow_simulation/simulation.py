#!/usr/bin/env python3

import argparse
import os
from functions.model_optimization import ModelOptimization


def run_hit_rate_simulation(
    data_paths,
    dataset_names,
    initial_train_size,
    random_seed,
    folds,
    model_name,
    max_num_samp,
    new_samp_per_step,
    n_trials,
    n_seeds,
    classification,
    baseline,
    score_col,
    optuna_print
):

    sim = ModelOptimization(
        data_path="",
        initial_train_size=initial_train_size,
        random_seed=random_seed,
        model_name=model_name,
    )

    sim.run_sim_hit_rate(
        data_paths=data_paths,
        dataset_names=dataset_names,
        max_num_samp=max_num_samp,
        folds=folds,
        new_samp_per_step=new_samp_per_step,
        n_trials=n_trials,
        random_seed=random_seed,
        n_seeds=n_seeds,
        classification=classification,
        score_col=score_col,
        baseline=baseline,
        optuna_print=optuna_print,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run active-learning hit-rate simulations"
    )

    parser.add_argument(
        "--data_paths",
        nargs="+",
        required=True,
        help="Paths to dataset CSV files"
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        required=True,
        help="Short dataset names (same order as data_paths)"
    )
    parser.add_argument(
        "--initial_train_size",
        type=int,
        default=20
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t33_650M_UR50D"
    )
    parser.add_argument(
        "--max_num_samp",
        type=int,
        default=200
    )
    parser.add_argument(
        "--new_samp_per_step",
        type=int,
        default=20
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10
    )
    parser.add_argument(
        "--baseline",
        action="store_true"
    )
    parser.add_argument(
        "--use_esmfold_emb",
        action="store_true"
    )
    parser.add_argument(
        "--classification",
        action="store_true"
    )
    parser.add_argument("--score_col", 
                        type=str, nargs="+", 
                        default=["score"], 
                        help="List of column names of the columns containing targets of interest.")
    
    parser.add_argument(
        "--optuna_print",
        action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_hit_rate_simulation(
        data_paths=args.data_paths,
        dataset_names=args.dataset_names,
        initial_train_size=args.initial_train_size,
        random_seed=args.random_seed,
        model_name=args.model_name,
        folds=args.folds,
        max_num_samp=args.max_num_samp,
        new_samp_per_step=args.new_samp_per_step,
        n_trials=args.n_trials,
        n_seeds=args.n_seeds,
        classification=args.classification,
        baseline=args.baseline,
        score_col=args.score_col,
        optuna_print=args.optuna_print,
    )
