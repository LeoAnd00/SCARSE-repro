#!/usr/bin/env python3
"""Run active-learning hit-rate simulations for one dataset, seed and representation."""

import argparse

from functions.model_optimization import ModelOptimization, ACQUISITION_STRATEGIES
from functions.representations import REPRESENTATIONS, NGRAM_VECTORIZERS


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
    representation,
    acquisition,
    explore_frac,
    score_col,
    results_dir,
    morgan_radius,
    morgan_bits,
    ngram_range,
    ngram_vectorizer,
    ngram_features,
    optuna_print,
    trial_timeout,
    round_timeout,
    direction,
):

    sim = ModelOptimization(
        data_path="",
        initial_train_size=initial_train_size,
        random_seed=random_seed,
        representation=representation,
        model_name=model_name,
        morgan_radius=morgan_radius,
        morgan_bits=morgan_bits,
        ngram_range=ngram_range,
        ngram_vectorizer=ngram_vectorizer,
        ngram_features=ngram_features,
        direction=direction,
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
        score_col=score_col,
        representation=representation,
        acquisition=acquisition,
        explore_frac=explore_frac,
        results_dir=results_dir,
        optuna_print=optuna_print,
        trial_timeout=trial_timeout,
        round_timeout=round_timeout,
    )


def _limit(value):
    """Seconds as float, with 0 or less meaning 'no limit'."""
    return value if value and value > 0 else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run active-learning hit-rate simulations"
    )

    parser.add_argument("--data_paths", nargs="+", required=True,
                        help="Paths to dataset CSV files")
    parser.add_argument("--dataset_names", nargs="+", required=True,
                        help="Short dataset names (same order as data_paths)")
    parser.add_argument("--initial_train_size", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model_name", type=str,
                        default="facebook/esm2_t33_650M_UR50D",
                        help="ESM2 checkpoint. Only used when --representation esm.")
    parser.add_argument("--max_num_samp", type=int, default=200)
    parser.add_argument("--new_samp_per_step", type=int, default=20)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--representation", type=str, default="esm",
                        choices=list(REPRESENTATIONS),
                        help="Sequence representation to use.")
    parser.add_argument("--acquisition", type=str, default="greedy",
                        choices=list(ACQUISITION_STRATEGIES),
                        help="Candidate selection strategy: 'greedy' takes the "
                             "top-k predictions, 'mixed' takes (1 - explore_frac) "
                             "of the batch from the best and explore_frac from "
                             "the worst predictions.")
    parser.add_argument("--explore_frac", type=float, default=0.3,
                        help="Fraction of each batch spent on the lowest-predicted "
                             "sequences when --acquisition mixed.")
    parser.add_argument("--morgan_radius", type=int, default=2)
    parser.add_argument("--morgan_bits", type=int, default=2048)
    parser.add_argument("--ngram_range", type=int, nargs=2, default=[1, 3],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--ngram_vectorizer", type=str, default="tfidf",
                        choices=list(NGRAM_VECTORIZERS))
    parser.add_argument("--ngram_features", type=int, default=1024)
    parser.add_argument("--results_dir", type=str, default="./simulation_results",
                        help="Where df_all.csv is written/appended.")
    parser.add_argument("--score_col", type=str, nargs="+", default=["score"],
                        help="Column names of the targets of interest.")
    parser.add_argument("--direction", type=str, default="maximize",
                        choices=["maximize", "minimize"],
                        help="Whether higher (maximize, default) or lower (minimize) "
                             "target values are desirable. Use 'minimize' for targets "
                             "such as -log(HC50) where a lower value is better. The "
                             "score is negated internally so selection and the true "
                             "top-10%% set stay consistent.")
    parser.add_argument("--optuna_print", action="store_true")
    parser.add_argument("--trial_timeout", type=float, default=600,
                        help="Wall-clock seconds one Optuna trial (all CV folds) may "
                             "take before it is discarded. 0 = no limit.")
    parser.add_argument("--round_timeout", type=float, default=3600,
                        help="Wall-clock seconds the Optuna search of one acquisition "
                             "round may take; the best trial so far is then used. "
                             "0 = no limit.")

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
        representation=args.representation,
        acquisition=args.acquisition,
        explore_frac=args.explore_frac,
        score_col=args.score_col,
        results_dir=args.results_dir,
        morgan_radius=args.morgan_radius,
        morgan_bits=args.morgan_bits,
        ngram_range=tuple(args.ngram_range),
        ngram_vectorizer=args.ngram_vectorizer,
        ngram_features=args.ngram_features,
        optuna_print=args.optuna_print,
        trial_timeout=_limit(args.trial_timeout),
        round_timeout=_limit(args.round_timeout),
        direction=args.direction,
    )