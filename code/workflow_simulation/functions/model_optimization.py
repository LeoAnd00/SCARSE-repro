"""
Active-learning workflow simulation.

At each round a Gaussian process is tuned on the sequences acquired so far,
used to score the remaining pool, and the top-predicted sequences are acquired.
The simulation records how good the acquired set is relative to random
selection.

One sequence representation is used per run: the ESM2 foundation model, or one
of the three baselines (physicochemical descriptors, Morgan fingerprints,
character n-grams). See ``representations.py``.

Regression only; classification is not part of the benchmark.
"""

import os
import random
import string
import warnings
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    DotProduct
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from Levenshtein import distance as levenshtein_distance

from functions.representations import SequenceRepresenter, REPRESENTATIONS

#: The 20 standard amino acids, used for sequence Shannon entropy.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


#: Display labels for each representation, used by the figures.
REPRESENTATION_LABELS = {
    "esm":      "GPR + ESM2",
    "pepbert":  "GPR + PepBERT",
    "physchem": "GPR + Physchem",
    "morgan":   "GPR + Morgan",
    "ngram":    "GPR + N-gram",
}

#: Candidate selection strategies.
#:   greedy - acquire the `new_samp_per_step` highest-predicted sequences.
#:            This is exactly UCB with beta = 0 (pure exploitation).
#:   mixed  - acquire (1 - explore_frac) of the batch from the highest-predicted
#:            and explore_frac from the *lowest*-predicted, which deliberately
#:            spends part of the budget on the model's least favourite region.
#:   ucb    - Upper Confidence Bound: rank by mu + beta * sigma, where sigma is
#:            the GPR posterior standard deviation, and take the top-k. beta
#:            trades exploitation (0) against exploration (larger). An
#:            uncertainty-aware acquisition that uses the full GP posterior.
ACQUISITION_STRATEGIES = ("greedy", "mixed", "ucb")

#: Strategies that need the model's predictive standard deviation.
UNCERTAINTY_STRATEGIES = ("ucb",)

#: Default UCB exploration weight.
DEFAULT_UCB_BETA = 1.0

#: Line style per strategy, so a figure can show colour = representation and
#: style = strategy.
ACQUISITION_LINESTYLES = {"greedy": "-", "mixed": "--", "ucb": ":"}


def acquisition_label(strategy, explore_frac, beta=DEFAULT_UCB_BETA):
    """Human-readable strategy name, derived so it follows explore_frac/beta."""
    if strategy == "greedy":
        return "Greedy (top-k)"
    if strategy == "ucb":
        return f"UCB (beta={beta:g})"
    pct = int(round(explore_frac * 100))
    return f"Mixed ({100 - pct}% best / {pct}% worst)"


# ----------------------------------------------------------------------
# Run-time limits for the GPR hyperparameter search
# ----------------------------------------------------------------------
# A single GPR fit can occasionally get stuck in its kernel-hyperparameter
# optimisation. Two limits keep that from stalling a whole simulation:
#
#   trial_timeout - wall-clock seconds one Optuna trial (all CV folds) may take.
#                   Checked inside the optimiser, so it also interrupts a single
#                   fit that is stuck. The trial is then discarded (pruned).
#   round_timeout - wall-clock seconds the Optuna study of one acquisition
#                   round may take. When reached, no new trials start and the
#                   best trial so far is used.
#
# Neither limit changes a result unless it is actually reached: the optimiser
# below makes exactly the call sklearn's default "fmin_l_bfgs_b" makes. Every
# time a limit is reached it is printed, so it can be reported.

#: Deadline (time.monotonic()) for the trial currently being evaluated, or None.
#: Module level on purpose: sklearn's clone() deep-copies estimator parameters,
#: so a deadline stored on the optimiser object would not reach the clones.
_FIT_DEADLINE = None


class TrialTimeLimitExceeded(Exception):
    """Raised inside a GPR fit when the current Optuna trial runs out of time."""


def _time_limited_lbfgs(obj_func, initial_theta, bounds):
    """sklearn's default GPR optimiser (L-BFGS-B), plus the trial deadline."""

    def wrapped(theta, *args, **kwargs):
        if _FIT_DEADLINE is not None and time.monotonic() > _FIT_DEADLINE:
            raise TrialTimeLimitExceeded()
        return obj_func(theta, *args, **kwargs)

    res = scipy.optimize.minimize(wrapped, initial_theta, method="L-BFGS-B",
                                  jac=True, bounds=bounds)
    return res.x, res.fun


#: Used when every trial of a round was pruned or timed out, so the round can
#: still make predictions. sklearn's defaults, with the RBF kernel.
FALLBACK_GPR_PARAMS = {"alpha": 1e-10, "normalize_y": False, "kernel": RBF()}


#: Columns of the results table, and the columns that identify one recorded
#: measurement. The identity columns are used to drop duplicates when a task
#: has been re-run.
RESULT_COLUMNS = ["Dataset", "Seed", "Num_samples", "Metric", "Value",
                  "Representation", "Acquisition"]
RESULT_ID_COLUMNS = ["Dataset", "Seed", "Num_samples", "Metric",
                     "Representation", "Acquisition"]

#: Sub-folder of the results directory holding one CSV per array task.
RUNS_SUBDIR = "runs"

#: Sub-folder holding the per-run acquisition record: every sequence that
#: entered the training set, with the round it was acquired in. This lets any
#: later sequence-level analysis be done without re-running the simulation.
SEQ_SUBDIR = "sequences"

#: Columns of the per-run sequence/acquisition record.
SEQ_COLUMNS = ["Dataset", "Seed", "Representation", "Acquisition",
               "round", "acq_rank", "num_samples_after", "sequence", "score"]


def run_file_name(dataset, representation, acquisition, seed):
    """File name for one (dataset, representation, strategy, seed) run."""
    safe = lambda v: "".join(c if c.isalnum() or c in "-." else "_" for c in str(v))
    return (f"{safe(representation)}__{safe(acquisition)}__{safe(dataset)}"
            f"__seed{safe(seed)}.csv")


def load_results(results_dir="./simulation_results", results_csv=None):
    """Every result row, from the per-run files and any legacy df_all.csv.

    Each array task writes its own file (see RUNS_SUBDIR), because several
    hundred tasks appending to one shared CSV can interleave and leave torn
    lines. Rows that are not readable as a complete record are dropped and
    reported rather than silently plotted.
    """
    frames = []

    legacy = Path(results_csv) if results_csv else Path(results_dir) / "df_all.csv"
    if legacy.exists():
        frames.append(pd.read_csv(legacy, sep=None, engine="python"))

    run_dir = Path(results_dir) / RUNS_SUBDIR
    for path in sorted(run_dir.glob("*.csv")):
        frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError(
            f"No results found: neither {legacy} nor any file in {run_dir}."
        )

    df = pd.concat(frames, ignore_index=True)
    if "Acquisition" not in df.columns:      # written before the mixed strategy
        df["Acquisition"] = "greedy"
    df["Acquisition"] = df["Acquisition"].fillna("greedy")

    # A torn line leaves fields empty or shifted. Such rows are unusable.
    ok = (df["Representation"].isin(REPRESENTATION_LABELS)
          & df["Acquisition"].isin(ACQUISITION_STRATEGIES)
          & pd.to_numeric(df["Value"], errors="coerce").notna()
          & pd.to_numeric(df["Num_samples"], errors="coerce").notna())
    if (~ok).any():
        print(f"Dropped {int((~ok).sum())} damaged row(s) of {len(df)}; these come "
              "from tasks that appended to df_all.csv at the same moment.")
        df = df[ok]

    df["Value"] = pd.to_numeric(df["Value"])
    df["Num_samples"] = pd.to_numeric(df["Num_samples"])

    # A re-run task replaces its earlier rows: per-run files are read last.
    before = len(df)
    df = df.drop_duplicates(subset=RESULT_ID_COLUMNS, keep="last").reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} duplicate row(s) from re-run tasks.")
    return df


#: Colors matching the rest of the analysis code.
REPRESENTATION_COLORS = {
    "GPR + ESM2":     "#4C72B0",
    "GPR + Physchem": "#DD8452",
    "GPR + Morgan":   "#55A868",
    "GPR + N-gram":   "#C44E52",
}


class ModelOptimization:

    def __init__(
        self,
        data_path,
        random_seed=42,
        initial_train_size=20,
        emb_batch_size=64,
        representation="esm",
        model_name="facebook/esm2_t33_650M_UR50D",
        morgan_radius=2,
        morgan_bits=2048,
        ngram_range=(1, 3),
        ngram_vectorizer="tfidf",
        ngram_features=1024,
        direction="maximize",
    ):

        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got {direction!r}"
            )

        if representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {REPRESENTATIONS}, got {representation!r}"
            )

        # Parameters
        self.data_path = data_path
        self.random_seed = random_seed
        self.initial_train_size = initial_train_size
        self.emb_batch_size = emb_batch_size
        self.representation = representation
        self.model_name = model_name
        self.direction = direction

        self.representer = SequenceRepresenter(
            representation=representation,
            model_name=model_name,
            emb_batch_size=emb_batch_size,
            random_seed=random_seed,
            morgan_radius=morgan_radius,
            morgan_bits=morgan_bits,
            ngram_range=ngram_range,
            ngram_vectorizer=ngram_vectorizer,
            ngram_features=ngram_features,
        )

        self.df = None
        self.seq_to_score = {}
        self.training_sequences = []
        self.test_sequences = []

        # Seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    # ------------------------------------------------------------------
    def prep_data(self, seq_col="sequence", score_col=["score"]):

        df = pd.read_csv(self.data_path, sep=None, engine='python')
        required_cols = set([seq_col] + score_col)
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Input file must contain columns: {required_cols}. Found: {df.columns.tolist()}"
            )

        # Store column names of the scores
        self.target_names = score_col

        # Convert sequence to string
        df["sequence"] = df[seq_col].astype(str)

        # Convert scores to float (regression only)
        for col in score_col:
            df[col] = df[col].astype(float)

        # Orient the target so that "higher is better".
        # The whole workflow selects the largest scores (argsort(...)[-k:]) and
        # defines the "true top 10%" the same way. For a minimize target (e.g.
        # -log(HC50) for haemolysis, where a lower value is more desirable) we
        # negate the score once here. Every downstream top-k selection, the true
        # top-10% set, and all normalised means then work unchanged and select the
        # LOWEST original scores. The reported "Top 10% peptides selected (%)"
        # metric carries no raw score, so it stays in interpretable units.
        if self.direction == "minimize":
            for col in score_col:
                df[col] = -df[col]

        # Keep only sequence + score columns
        df = df[["sequence"] + score_col].copy()

        self.df = df

        # Positional numpy array, so the argsort-based indexing below is
        # positional rather than label-based.
        self.y_all = df[score_col[0]].to_numpy()

        # Create a mapping from sequence -> score, or list of scores
        if len(score_col) == 1:
            self.seq_to_score = dict(zip(df["sequence"], df[score_col[0]]))
        else:
            self.seq_to_score = dict(zip(df["sequence"], df[score_col].values.tolist()))

        print("Finished preparing data!")

    def load_model(self):
        """Load the foundation model, if the representation needs one."""
        self.representer.load_model()

    def compute_embeddings(self, sequences, fit=False, batch_size=None):
        """Represent sequences. ``fit`` is only meaningful for TF-IDF n-grams."""
        return self.representer.transform(sequences, fit=fit, batch_size=batch_size)

    def initialize_training_set(self):

        size = max(0, self.initial_train_size)
        initial_training_df = self.df.sample(n=size, random_state=self.random_seed)
        self.training_sequences = initial_training_df["sequence"].tolist()
        self.test_df = self.df.drop(initial_training_df.index).reset_index(drop=True).copy()
        self.test_sequences = self.test_df["sequence"].tolist()

    # ------------------------------------------------------------------
    def optimize_all_models(self, folds=10, random_seed=42, n_trials=100, optuna_print=True,
                            trial_timeout=None, round_timeout=None):
        """Tune the downstream model on the sequences acquired so far.

        trial_timeout / round_timeout are wall-clock limits in seconds (None =
        no limit); see the note on run-time limits at the top of this module.
        """

        # Suppress convergence and feature name warnings
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        ## Seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        y_train = []
        for s in self.training_sequences:
            y_train.append(self.seq_to_score[s])
        y_train = np.array(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        y_test = []
        for s in self.test_sequences:
            y_test.append(self.seq_to_score[s])
        y_test = np.array(y_test)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        n_targets = y_train.shape[1]
        self.n_targets = n_targets
        n_samples = len(self.training_sequences)
        folds = min(folds, n_samples)
        self.folds = folds
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_seed)

        seq_array = np.array(self.training_sequences)

        folds_per_label = {}

        for label_idx in range(n_targets):
            folds_list = []
            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(seq_array)):
                train_df_temp = seq_array[train_idx]
                val_df_temp = seq_array[valid_idx]

                # Represent the fold's training sequences first, with fit=True,
                # so an n-gram vocabulary is learned from the fold's training
                # data only and the validation fold is transformed with it.
                X_train_fold = self.compute_embeddings(list(train_df_temp), fit=True)
                X_val_fold = self.compute_embeddings(list(val_df_temp), fit=False)

                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_val_fold = scaler.transform(X_val_fold)

                train_df = {
                    'sequence': X_train_fold,
                    'score': y_train[train_idx, label_idx]
                }

                val_df = {
                    'sequence': X_val_fold,
                    'score': y_train[valid_idx, label_idx]
                }

                dataset_dict = {
                    'train': train_df,
                    'validation': val_df
                }

                # Store DatasetDict for each fold
                folds_list.append(dataset_dict)

            folds_per_label[f"label_{label_idx}"] = folds_list

        model_configs = {
            "GaussianProcessRegressor": {"class": GaussianProcessRegressor, "params": {
                "alpha": ("float", 1e-10, 1e-6),
                "normalize_y": ("categorical", [True, False]),
                "kernel": ("categorical", [
                    RBF(),
                    Matern(),
                    RationalQuadratic(),
                    DotProduct()])
            }}
        }

        best_model = None

        # Iterate over targets and models
        for label_idx in range(n_targets):

            print(f"\nOptimizing models for target {self.target_names[label_idx]} "
                  f"{label_idx + 1}/{n_targets}...")
            for name, cfg in model_configs.items():
                print(f"\nOptimizing {name}...")
                self.current_name = name

                ModelClass = cfg["class"]
                param_bounds = cfg["params"]

                n_timed_out = [0]

                def objective(trial):
                    global _FIT_DEADLINE

                    _FIT_DEADLINE = (time.monotonic() + trial_timeout
                                     if trial_timeout else None)
                    try:

                        params = {}
                        for name_, cfg in param_bounds.items():
                            ptype = cfg[0]
                            if ptype == "int":
                                params[name_] = trial.suggest_int(name_, cfg[1], cfg[2])
                            elif ptype == "float":
                                params[name_] = trial.suggest_float(name_, cfg[1], cfg[2])
                            elif ptype == "float_log":
                                params[name_] = trial.suggest_float(name_, cfg[1], cfg[2], log=True)
                            elif ptype == "categorical":
                                params[name_] = trial.suggest_categorical(name_, cfg[1])

                        # Same optimiser as sklearn's default, but it honours
                        # the trial deadline.
                        model = ModelClass(**params, optimizer=_time_limited_lbfgs)

                        fold_scores = []

                        for fold_idx, dataset_dict in enumerate(folds_per_label[f"label_{label_idx}"]):
                            if _FIT_DEADLINE is not None and time.monotonic() > _FIT_DEADLINE:
                                raise TrialTimeLimitExceeded()
                            model_fold = clone(model)
                            train_df = dataset_dict["train"]
                            val_df = dataset_dict["validation"]

                            X_train_fold = np.vstack(train_df["sequence"])
                            X_val_fold = np.vstack(val_df["sequence"])

                            y_train_fold = np.vstack(train_df["score"]).ravel()
                            y_val_fold = np.vstack(val_df["score"]).ravel()

                            model_fold.fit(X_train_fold, y_train_fold)
                            y_pred = model_fold.predict(X_val_fold)

                            mse = mean_squared_error(y_val_fold, y_pred)
                            fold_scores.append(mse)

                        return np.mean(fold_scores)
                    except TrialTimeLimitExceeded:
                        n_timed_out[0] += 1
                        kernel = params.get("kernel")
                        print(f"  trial {trial.number} stopped after {trial_timeout} s "
                              f"(kernel={kernel}, alpha={params.get('alpha', float('nan')):.3g}, "
                              f"normalize_y={params.get('normalize_y')})", flush=True)
                        raise optuna.TrialPruned()
                    except Exception:
                        raise optuna.TrialPruned()
                    finally:
                        _FIT_DEADLINE = None

                optuna.logging.set_verbosity(optuna.logging.ERROR)
                study = optuna.create_study(
                    direction='minimize',
                    sampler=optuna.samplers.TPESampler(seed=random_seed)
                )
                t_start = time.monotonic()
                study.optimize(objective, n_trials=n_trials, timeout=round_timeout,
                               show_progress_bar=optuna_print)
                elapsed = time.monotonic() - t_start

                completed = [t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE]
                n_run = len(study.trials)
                note = ""
                if n_run < n_trials:
                    note += f"; round time limit ({round_timeout} s) reached"
                if n_timed_out[0]:
                    note += f"; {n_timed_out[0]} trial(s) hit the trial time limit"
                print(f"  Optuna: {n_run}/{n_trials} trials, {len(completed)} completed, "
                      f"{elapsed:.0f} s{note}", flush=True)

                if completed:
                    best_params = study.best_params
                else:
                    print("  No trial completed; using the fallback GPR "
                          f"{FALLBACK_GPR_PARAMS}", flush=True)
                    best_params = dict(FALLBACK_GPR_PARAMS)

                best_model = ModelClass(**best_params)

        return best_model, y_train, y_test

    # ------------------------------------------------------------------
    # Diversity and entropy of a sequence set
    # ------------------------------------------------------------------
    @staticmethod
    def sequence_shannon_entropy(sequences):
        """Shannon entropy (bits) of the pooled amino-acid composition.

        All residues across ``sequences`` are pooled, the frequency of each of
        the 20 standard amino acids is taken, and the Shannon entropy of that
        distribution is returned. High entropy = the set samples many amino
        acids evenly (compositionally diverse); low entropy = a few residues
        dominate. Non-standard residues are ignored. Returns ``nan`` for an
        empty set.

        This is the "sequence diversity" quantity Reviewer 2 suggested relating
        to active-learning success.
        """
        counts = np.zeros(len(AMINO_ACIDS), dtype=float)
        index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        for seq in sequences:
            for aa in str(seq).upper():
                j = index.get(aa)
                if j is not None:
                    counts[j] += 1.0
        total = counts.sum()
        if total == 0:
            return float("nan")
        p = counts[counts > 0] / total
        return float(-(p * np.log2(p)).sum())

    @staticmethod
    def mean_pairwise_levenshtein(sequences, normalized=True):
        """Mean pairwise (optionally length-normalised) Levenshtein distance.

        A length-robust sequence-diversity measure for a batch, used instead of
        Hamming distance because the peptides are not equal length. With
        ``normalized=True`` each distance is divided by the longer of the two
        sequences, giving a value in [0, 1]. Returns ``nan`` for fewer than two
        sequences.
        """
        seqs = [str(s) for s in sequences]
        n = len(seqs)
        if n < 2:
            return float("nan")
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = levenshtein_distance(seqs[i], seqs[j])
                if normalized:
                    denom = max(len(seqs[i]), len(seqs[j]))
                    d = d / denom if denom > 0 else 0.0
                total += d
                pairs += 1
        return float(total / pairs) if pairs else float("nan")

    @staticmethod
    def embedding_space_variance(embeddings):
        """Mean per-dimension variance of a set of embedding vectors.

        A representation-space spread measure for a batch: the variance of each
        feature across the batch, averaged over features. Equal to the mean
        squared Euclidean distance to the centroid divided by the dimension, so
        it is comparable across representations of different width. Returns
        ``nan`` for fewer than two vectors.
        """
        X = np.asarray(embeddings, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            return float("nan")
        return float(X.var(axis=0).mean())

    @staticmethod
    def select_candidates(pred_scores, n_select, strategy="greedy", explore_frac=0.3,
                          pred_std=None, beta=DEFAULT_UCB_BETA):
        """Indices of the sequences to acquire from the remaining pool.

        Parameters
        ----------
        pred_scores : array
            Predicted mean score for every sequence still in the pool.
        n_select : int
            Batch size. Clipped to the pool size on the final rounds.
        strategy : {"greedy", "mixed", "ucb"}
            ``greedy`` takes the n_select highest-predicted sequences.
            ``mixed`` takes ``round(explore_frac * n_select)`` from the
            lowest-predicted and the rest from the highest-predicted.
            ``ucb`` ranks by ``pred_scores + beta * pred_std`` and takes the
            top-k. ``greedy`` is the special case ``ucb`` with ``beta = 0``.
        explore_frac : float
            Fraction of the batch drawn from the predicted worst. ``mixed`` only.
        pred_std : array, optional
            Posterior standard deviation per pool sequence. Required for
            ``ucb``; ignored otherwise.
        beta : float
            UCB exploration weight. ``ucb`` only.

        Returns
        -------
        numpy.ndarray
            Unique positional indices into ``pred_scores``.
        """
        if strategy not in ACQUISITION_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {ACQUISITION_STRATEGIES}, got {strategy!r}"
            )

        pool = len(pred_scores)
        n_select = int(min(n_select, pool))
        if n_select <= 0:
            return np.array([], dtype=int)

        if strategy == "ucb":
            if pred_std is None:
                raise ValueError("ucb acquisition requires pred_std.")
            acq = np.asarray(pred_scores, dtype=float) + beta * np.asarray(pred_std, dtype=float)
            order = np.argsort(acq)
            return order[-n_select:][::-1]

        order = np.argsort(pred_scores)

        if strategy == "greedy":
            return order[-n_select:][::-1]

        # mixed: split the batch between the predicted best and the predicted worst.
        n_worst = int(round(explore_frac * n_select))
        n_worst = min(n_worst, n_select)
        n_best = n_select - n_worst

        # On the last rounds the pool can be smaller than best + worst, which
        # would make the two slices overlap and acquire the same sequence twice.
        if n_best + n_worst > pool:
            n_worst = max(0, pool - n_best)

        best_idx = order[-n_best:][::-1] if n_best > 0 else np.array([], dtype=int)
        worst_idx = order[:n_worst] if n_worst > 0 else np.array([], dtype=int)

        selected = np.concatenate([best_idx, worst_idx]).astype(int)

        # Belt and braces: never return a duplicate, whatever the pool size.
        _, first = np.unique(selected, return_index=True)
        return selected[np.sort(first)]

    def pred(self, train_seqs, y_train, test_seqs, label_idx, best_model, score_col,
             return_std=False):
        """Fit the tuned model on the acquired set and score the remaining pool.

        Parameters
        ----------
        return_std : bool
            When True, also return the GaussianProcessRegressor posterior
            standard deviation for each pool sequence (used by UCB). If the
            model does not support ``return_std`` (it always does here, but this
            keeps the helper safe) a zero vector is returned, which makes UCB
            degenerate gracefully to greedy.

        Returns
        -------
        numpy.ndarray or tuple(numpy.ndarray, numpy.ndarray)
            Predicted means, or ``(means, stds)`` when ``return_std=True``.
        """

        # fit=True refits the n-gram vocabulary on the current training set,
        # which grows every acquisition round; the pool is then transformed
        # with that vocabulary.
        X_train = self.compute_embeddings(list(train_seqs), fit=True)
        X_test = self.compute_embeddings(list(test_seqs), fit=False)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = clone(best_model)
        model.fit(X_train, y_train[:, label_idx])

        if return_std:
            try:
                mu, sigma = model.predict(X_test, return_std=True)
            except TypeError:
                # Model without a return_std interface: fall back to zero std,
                # so UCB behaves like greedy rather than erroring.
                mu = model.predict(X_test)
                sigma = np.zeros_like(mu, dtype=float)
            return mu, sigma

        return model.predict(X_test)

    # ------------------------------------------------------------------
    def run_sim_hit_rate(
                        self,
                        data_paths,
                        dataset_names,
                        score_col=["score"],
                        new_samp_per_step=20,
                        max_num_samp=300,
                        folds=10,
                        random_seed=42,
                        n_seeds=5,
                        n_trials=100,
                        representation=None,
                        acquisition="greedy",
                        explore_frac=0.3,
                        beta=DEFAULT_UCB_BETA,
                        results_dir="./simulation_results",
                        optuna_print=True,
                        trial_timeout=None,
                        round_timeout=None):
        """
        Run an iterative acquisition simulation to evaluate model-driven
        selection performance versus a random baseline.

        This function simulates an active-learning–style workflow where, at each
        round, a model selects new sequences from a test pool based on predicted
        scores.

        Parameters
        ----------
        acquisition : {"greedy", "mixed"}
            How the `new_samp_per_step` sequences are picked from the predictions.
            "greedy" takes the top-k; "mixed" takes (1 - explore_frac) of the
            batch from the top and explore_frac from the bottom, which is the
            more exploratory strategy.
        explore_frac : float
            Fraction of each batch spent on the lowest-predicted sequences when
            `acquisition="mixed"`. Ignored for "greedy".
        trial_timeout, round_timeout : float or None
            Wall-clock limits (seconds) for one Optuna trial and for the Optuna
            study of one round. None = no limit. See the note on run-time limits
            at the top of this module.
        """
        os.makedirs(results_dir, exist_ok=True)

        if acquisition not in ACQUISITION_STRATEGIES:
            raise ValueError(
                f"acquisition must be one of {ACQUISITION_STRATEGIES}, got {acquisition!r}"
            )
        if not 0.0 <= explore_frac <= 1.0:
            raise ValueError(f"explore_frac must be in [0, 1], got {explore_frac!r}")

        if representation is not None:
            if representation not in REPRESENTATIONS:
                raise ValueError(
                    f"representation must be one of {REPRESENTATIONS}, got {representation!r}"
                )
            self.representation = representation
            self.representer.representation = representation

        # Load the foundation model once, and only if it is the representation.
        self.load_model()

        n_seeds = n_seeds
        start_seed = random_seed
        seeds = [start_seed + i for i in range(n_seeds)]

        rows = []
        seq_rows = []   # full acquisition record: one row per acquired sequence

        for data_idx, data_path in enumerate(data_paths):

            ds_name = dataset_names[data_idx]

            self.data_path = data_path

            for seed in seeds:
                self.random_seed = seed
                self.prep_data(score_col=score_col)
                self.initialize_training_set()

                k = int(np.ceil(len(self.y_all) * 0.1))
                all_sequences = list(self.seq_to_score.keys())

                top10pct_highest_idx = np.argsort(self.y_all)[-k:]
                top10pct_highest_seqs = {all_sequences[i] for i in top10pct_highest_idx}

                all_selected_seqs = set(self.training_sequences)

                # Diversity of the INITIAL seed set, recorded once per (dataset,
                # seed). These are the features we correlate with the eventual
                # active-learning outcome (R2.2: does a more diverse starting
                # pool lead to better campaigns?). Recorded for every
                # representation/strategy run, but the value depends only on the
                # seed set, so it is identical across them for a given seed.
                initial_seqs = list(self.training_sequences)
                initial_entropy = self.sequence_shannon_entropy(initial_seqs)
                initial_lev = self.mean_pairwise_levenshtein(initial_seqs)
                rows.append([ds_name, seed, len(initial_seqs),
                             "Initial Seed Shannon Entropy", initial_entropy,
                             self.representation, acquisition])
                rows.append([ds_name, seed, len(initial_seqs),
                             "Initial Seed Mean Levenshtein", initial_lev,
                             self.representation, acquisition])

                # Record the initial seed set (round 0) in the acquisition log,
                # so the full set of training sequences and the order they were
                # acquired is recoverable for any later sequence-level analysis.
                for rank, s in enumerate(initial_seqs):
                    seq_rows.append([ds_name, seed, self.representation, acquisition,
                                     0, rank, len(initial_seqs), s,
                                     self.seq_to_score.get(s, np.nan)])

                # Simulation

                round_idx = -1
                while len(self.training_sequences) < int(max_num_samp):
                    round_idx += 1
                    t_round = time.monotonic()
                    print(f"[{ds_name} seed {seed}] round {round_idx}: "
                          f"{len(self.training_sequences)} training sequences", flush=True)

                    best_model, y_train, y_test = self.optimize_all_models(
                        folds=folds,
                        random_seed=seed,
                        n_trials=n_trials,
                        optuna_print=optuna_print,
                        trial_timeout=trial_timeout,
                        round_timeout=round_timeout,
                    )

                    # UCB needs the posterior std; the other strategies only
                    # need the mean prediction.
                    needs_std = acquisition in UNCERTAINTY_STRATEGIES
                    pred_out = self.pred(
                        train_seqs=self.training_sequences,
                        y_train=y_train,
                        test_seqs=self.test_sequences,
                        label_idx=0,
                        best_model=best_model,
                        score_col=score_col,
                        return_std=needs_std,
                    )
                    if needs_std:
                        pred_scores, pred_std = pred_out
                    else:
                        pred_scores, pred_std = pred_out, None

                    if round_idx == 0:

                        ### Selection performance normalized between the overall mean value and the highest
                        ### possible mean value based on the selection at each round
                        mean_all = np.mean(self.y_all)
                        highest_idx = np.argsort(self.y_all)[-new_samp_per_step:][::-1]
                        highest_mean = np.mean(self.y_all[np.array(highest_idx)])
                        selected_mean = np.mean(y_train[:, 0])
                        norm_selected_mean = (selected_mean - mean_all) / (highest_mean - mean_all)
                        rows.append([ds_name, seed, len(self.training_sequences),
                                     "Norm. Target Mean (Round)", norm_selected_mean,
                                     self.representation, acquisition])

                    # Select the next batch with the requested strategy: either
                    # the top-k predictions, or a best/worst mix.
                    pred_idx = self.select_candidates(
                        pred_scores,
                        new_samp_per_step,
                        strategy=acquisition,
                        explore_frac=explore_frac,
                        pred_std=pred_std,
                        beta=beta,
                    )

                    print(f"[{ds_name} seed {seed}] round {round_idx} done in "
                          f"{time.monotonic() - t_round:.0f} s", flush=True)

                    # Exhausted pool: nothing left to acquire, so stop rather
                    # than spin on an unchanging training set.
                    if len(pred_idx) == 0:
                        print(f"[{ds_name} seed {seed}] pool exhausted at "
                              f"{len(self.training_sequences)} samples, stopping early.")
                        break

                    ### Selection performance normalized between the overall mean value and the highest
                    ### possible mean value based on the selection at each round
                    # The batch can be shorter than new_samp_per_step if the pool
                    # runs out, so normalise against a batch of the same size.
                    n_sel = len(pred_idx)
                    mean_all = np.mean(y_test[:, 0])
                    highest_idx = np.argsort(y_test[:, 0])[-n_sel:][::-1]
                    highest_mean = np.mean(y_test[:, 0][np.array(highest_idx)])
                    selected_mean = np.mean(y_test[:, 0][np.array(pred_idx)])
                    norm_selected_mean = (selected_mean - mean_all) / (highest_mean - mean_all)
                    rows.append([ds_name, seed, len(self.training_sequences)+n_sel,
                                 "Norm. Target Mean (Round)", norm_selected_mean,
                                 self.representation, acquisition])

                    ### Cumulative performance of entire training data set normalized from overall mean value to
                    ### highest possible mean based on current number of samples in training data
                    current_mean = np.mean(y_train[:, 0])
                    highest_idx = np.argsort(self.y_all)[-len(y_train[:, 0]):][::-1]
                    highest_mean = np.mean(self.y_all[np.array(highest_idx)])
                    mean_all = np.mean(self.y_all)
                    norm_current_mean = (current_mean - mean_all) / (highest_mean - mean_all)
                    rows.append([ds_name, seed, len(self.training_sequences),
                                 "Norm. Target Mean (Cumulative)", norm_current_mean,
                                 self.representation, acquisition])

                    ### Percentage of the true top 10% that has already been acquired
                    overlap = len(top10pct_highest_seqs & all_selected_seqs)
                    pct_top10pct = overlap / k * 100
                    rows.append([ds_name, seed, len(self.training_sequences),
                                 "Top 10% peptides selected (%)", pct_top10pct,
                                 self.representation, acquisition])

                    ### Difference between mean score of the top 10% peptides that are available at each round
                    ### and the mean of all data, normalized between max mean and global mean
                    k_avail = int(np.ceil(len(y_test[:, 0]) * 0.1))

                    top10_idx = np.argsort(y_test[:, 0])[-k_avail:]
                    top10_mean = np.mean(y_test[:, 0][top10_idx])

                    k_global = int(np.ceil(len(self.y_all) * 0.1))
                    global_top10_mean = np.mean(np.sort(self.y_all)[-k_global:])

                    top10_gap_norm = (top10_mean - mean_all) / (global_top10_mean - mean_all)

                    rows.append([ds_name, seed, len(self.training_sequences),
                                 "Remaining Top10 Quality", top10_gap_norm,
                                 self.representation, acquisition])

                    # Update new training set
                    selected_seqs = [self.test_sequences[i] for i in pred_idx]

                    # Record each acquired sequence with the round it was picked
                    # in and its rank within the batch (pred_idx is in
                    # acquisition order, so rank 0 is the top pick). num_samples_after
                    # is the training-set size once this batch is added.
                    n_after = len(self.training_sequences) + len(selected_seqs)
                    for rank, s in enumerate(selected_seqs):
                        seq_rows.append([ds_name, seed, self.representation, acquisition,
                                         round_idx + 1, rank, n_after, s,
                                         self.seq_to_score.get(s, np.nan)])

                    # Batch-diversity metrics for the sequences acquired THIS
                    # round (R3.2): sequence spread via mean pairwise normalised
                    # Levenshtein, and representation-space spread via the mean
                    # per-dimension embedding variance. fit=False keeps the
                    # n-gram vocabulary (fit on the training set) unchanged.
                    batch_lev = self.mean_pairwise_levenshtein(selected_seqs)
                    rows.append([ds_name, seed, len(self.training_sequences) + len(selected_seqs),
                                 "Batch Mean Levenshtein", batch_lev,
                                 self.representation, acquisition])
                    if len(selected_seqs) >= 2:
                        batch_emb = self.compute_embeddings(list(selected_seqs), fit=False)
                        batch_emb_var = self.embedding_space_variance(batch_emb)
                    else:
                        batch_emb_var = float("nan")
                    rows.append([ds_name, seed, len(self.training_sequences) + len(selected_seqs),
                                 "Batch Embedding Variance", batch_emb_var,
                                 self.representation, acquisition])

                    self.training_sequences.extend(selected_seqs)

                    # remove from test set
                    mask = np.ones(len(self.test_sequences), dtype=bool)
                    mask[pred_idx] = False
                    self.test_sequences = [s for i, s in enumerate(self.test_sequences) if mask[i]]
                    self.test_df = self.test_df[
                        self.test_df["sequence"].isin(self.test_sequences)
                    ].reset_index(drop=True)

                    all_selected_seqs.update(selected_seqs)

                ### Final round bookkeeping
                y_train = []
                for s in self.training_sequences:
                    y_train.append(self.seq_to_score[s])
                y_train = np.array(y_train)
                if y_train.ndim == 1:
                    y_train = y_train.reshape(-1, 1)

                current_mean = np.mean(y_train[:, 0])
                highest_idx = np.argsort(self.y_all)[-len(y_train[:, 0]):][::-1]
                highest_mean = np.mean(self.y_all[np.array(highest_idx)])
                mean_all = np.mean(self.y_all)
                norm_current_mean = (current_mean - mean_all) / (highest_mean - mean_all)
                rows.append([ds_name, seed, len(self.training_sequences),
                             "Norm. Target Mean (Cumulative)", norm_current_mean,
                             self.representation, acquisition])

                ### Percentage of the true top 10% that has already been acquired
                overlap = len(top10pct_highest_seqs & all_selected_seqs)
                pct_top10pct = overlap / k * 100
                rows.append([ds_name, seed, len(self.training_sequences),
                             "Top 10% peptides selected (%)", pct_top10pct,
                             self.representation, acquisition])

                ### Remaining pool quality
                k_avail = int(np.ceil(len(y_test[:, 0]) * 0.1))

                top10_idx = np.argsort(y_test[:, 0])[-k_avail:]
                top10_mean = np.mean(y_test[:, 0][top10_idx])

                k_global = int(np.ceil(len(self.y_all) * 0.1))
                global_top10_mean = np.mean(np.sort(self.y_all)[-k_global:])

                top10_gap_norm = (top10_mean - mean_all) / (global_top10_mean - mean_all)

                rows.append([ds_name, seed, len(self.training_sequences),
                             "Remaining Top10 Quality", top10_gap_norm,
                             self.representation, acquisition])

        df_all = pd.DataFrame(rows, columns=RESULT_COLUMNS)

        # One file per (dataset, representation, strategy, seed). Array tasks run
        # at the same time, and appending them all to one CSV interleaves the
        # writes and leaves torn lines. Separate files also mean a re-run task
        # replaces its own results instead of adding a second copy.
        run_dir = Path(results_dir) / RUNS_SUBDIR
        run_dir.mkdir(parents=True, exist_ok=True)
        for (ds_name, seed), part in df_all.groupby(["Dataset", "Seed"], sort=False):
            out = run_dir / run_file_name(ds_name, self.representation,
                                          acquisition, seed)
            part.to_csv(out, index=False)
            print(f"Wrote {len(part)} rows to {out}", flush=True)

        # Full acquisition record: one CSV per (dataset, representation,
        # strategy, seed) in sequences/, same naming as runs/. This stores every
        # training sequence and the round/rank it was acquired in, so later
        # sequence-level analyses need no re-run.
        seq_all = pd.DataFrame(seq_rows, columns=SEQ_COLUMNS)
        seq_dir = Path(results_dir) / SEQ_SUBDIR
        seq_dir.mkdir(parents=True, exist_ok=True)
        for (ds_name, seed), part in seq_all.groupby(["Dataset", "Seed"], sort=False):
            out = seq_dir / run_file_name(ds_name, self.representation,
                                          acquisition, seed)
            part.to_csv(out, index=False)
            print(f"Wrote {len(part)} acquisition rows to {out}", flush=True)

    # ------------------------------------------------------------------
    def visualize(self, dataset_names, dataset_sizes=None,
                  results_dir="./simulation_results",
                  results_csv=None,
                  figures_dir="./simulation_results/figures",
                  n_cols=3, max_num_samp=200, new_samp_per_step=20,
                  explore_frac=0.3, acquisitions=None, figure_suffix=""):
        """Per-dataset acquisition curves.

        Colour encodes the sequence representation and line style the candidate
        selection strategy, so both can be compared in one panel.

        Parameters
        ----------
        dataset_names : list[str]
            Datasets to plot, in panel order.
        results_dir : str
            Results folder. Read are every per-run file in its ``runs``
            sub-folder and, if present, a legacy ``df_all.csv`` beside it.
        results_csv : str, optional
            A specific legacy CSV to read instead of ``<results_dir>/df_all.csv``.
        dataset_sizes : dict[str, int], optional
            Pool size per dataset, used to draw the random-selection reference
            on the "Top 10% peptides selected (%)" panels. Datasets missing
            from the dict simply get no reference line.
        explore_frac : float
            Only used to write the legend label for the "mixed" strategy, so it
            matches the fraction the simulation was run with.
        acquisitions : list[str], optional
            Restrict the figure to these strategies (e.g. ``["greedy"]`` to
            reproduce the single-strategy figure). Default: everything present.
        figure_suffix : str
            Appended to the output file names, handy when writing one figure per
            strategy.
        """

        os.makedirs(figures_dir, exist_ok=True)
        df = load_results(results_dir=results_dir, results_csv=results_csv)
        print(f"{len(df)} rows: "
              f"{df['Dataset'].nunique()} datasets, "
              f"{sorted(df['Representation'].unique())}, "
              f"{sorted(df['Acquisition'].unique())}")

        if acquisitions is not None:
            df = df[df["Acquisition"].isin(acquisitions)]
            if df.empty:
                raise ValueError(f"No rows for acquisition strategies {acquisitions}.")

        df["Model Label"] = df["Representation"].map(REPRESENTATION_LABELS)
        model_order = [REPRESENTATION_LABELS[r] for r in REPRESENTATIONS
                       if REPRESENTATION_LABELS[r] in set(df["Model Label"])]
        acq_order = [a for a in ACQUISITION_STRATEGIES
                     if a in set(df["Acquisition"])]
        acq_labels = {a: acquisition_label(a, explore_frac) for a in acq_order}

        sns.set_theme(
            context="notebook",
            style="white",
            rc={
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "figure.dpi": 300,
                "axes.linewidth": 1.2,
                "axes.edgecolor": "#333333",
                "axes.labelpad": 8,
                "axes.titlepad": 10,
                "font.size": 11,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
            }
        )

        dataset_sizes = dataset_sizes or {}
        n_steps = int(np.ceil((max_num_samp - self.initial_train_size) / new_samp_per_step)) + 1
        data_sizes = np.linspace(self.initial_train_size, max_num_samp, n_steps)
        y_pct_random = {
            name: [i / size * 100 for i in data_sizes]
            for name, size in dataset_sizes.items()
        }

        metrics_groups = {
            "per_round": ["Norm. Target Mean (Round)"],
            "cumulative": ["Norm. Target Mean (Cumulative)"],
            "selection": ["Top 10% peptides selected (%)"],
            "remaining": ["Remaining Top10 Quality"],
        }

        pretty = {
            "Norm. Target Mean (Round)": "Target mean (Round)",
            "Norm. Target Mean (Cumulative)": "Target mean (Cumulative)",
        }

        letters = list(string.ascii_lowercase)

        for group_name, metrics in metrics_groups.items():
            for metric in metrics:

                df_metric = df[df["Metric"] == metric]
                if df_metric.empty:
                    print(f"No rows for metric {metric!r}, skipping.")
                    continue

                ylabel = pretty.get(metric, metric)

                # Compute mean/std across seeds
                summary = (
                    df_metric
                    .groupby(["Dataset", "Num_samples", "Model Label", "Acquisition"])["Value"]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                # ---- Global y-limits across ALL datasets ----
                present = [d for d in dataset_names if d in set(summary["Dataset"])]
                if not present:
                    print(f"None of the requested datasets are in the results for {metric!r}.")
                    continue

                sub = summary[summary["Dataset"].isin(present)]
                y_min = (sub["mean"] - sub["std"].fillna(0)).min()
                y_max = (sub["mean"] + sub["std"].fillna(0)).max()
                padding = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
                y_min = math.floor((y_min - padding) * 10) / 10
                y_max = math.ceil((y_max + padding) * 10) / 10

                n_rows = int(np.ceil(len(present) / n_cols))
                fig = plt.figure(figsize=(4.7 * n_cols, 3.4 * n_rows + 1.2))
                gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.06)

                for i, dataset in enumerate(present):
                    row, col = i // n_cols, i % n_cols
                    ax = fig.add_subplot(gs[row, col])

                    ax.text(0.02, 1.12, letters[i], transform=ax.transAxes,
                            fontsize=11, fontweight="bold", fontfamily="DejaVu Serif",
                            va="top", ha="left")

                    df_plot = summary[summary["Dataset"] == dataset]

                    for model in model_order:
                        for acq in acq_order:
                            df_temp = df_plot[
                                (df_plot["Model Label"] == model)
                                & (df_plot["Acquisition"] == acq)
                            ].sort_values("Num_samples")
                            if df_temp.empty:
                                continue
                            x = df_temp["Num_samples"].values
                            y_mean = df_temp["mean"].values
                            y_std = df_temp["std"].fillna(0).values
                            color = REPRESENTATION_COLORS[model]

                            ax.plot(x, y_mean, linewidth=2.2, color=color,
                                    linestyle=ACQUISITION_LINESTYLES[acq],
                                    label=f"{model} - {acq_labels[acq]}")
                            # Only shade one strategy, otherwise overlapping
                            # bands of the same colour become unreadable.
                            if acq == acq_order[0]:
                                ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                                                alpha=0.22, color=color)

                    # Random selection reference
                    if metric == "Top 10% peptides selected (%)" and dataset in y_pct_random:
                        ax.plot(data_sizes, y_pct_random[dataset], color="red",
                                linewidth=1.8, linestyle="--", label="Random")

                    ax.set_title(dataset)
                    ax.set_ylim(y_min, y_max)

                    if col == 0:
                        ax.set_ylabel(ylabel)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])

                    ax.set_xlabel("n peptides screened")
                    ax.set_xticks(np.arange(self.initial_train_size, max_num_samp + 1,
                                            new_samp_per_step * 2))

                    for spine in ax.spines.values():
                        spine.set_visible(True)

                    ax.grid(True, axis="y")
                    ax.grid(False, axis="x")

                # Colour = representation, line style = selection strategy.
                handles = [Patch(facecolor=REPRESENTATION_COLORS[m], label=m) for m in model_order]
                if len(acq_order) > 1:
                    handles += [
                        plt.Line2D([0], [0], color="#333333",
                                   linestyle=ACQUISITION_LINESTYLES[a],
                                   linewidth=2.2, label=acq_labels[a])
                        for a in acq_order
                    ]
                if metric == "Top 10% peptides selected (%)" and y_pct_random:
                    handles.append(plt.Line2D([0], [0], color="red", linestyle="--",
                                              label="Random"))
                # Wrap onto a second row rather than squeezing the labels, and
                # lift the legend when it does so it clears the top panels.
                ncol = len(handles) if len(handles) <= 5 else int(np.ceil(len(handles) / 2))
                legend_rows = int(np.ceil(len(handles) / ncol))
                fig.legend(handles=handles, loc="upper center",
                           ncol=ncol, frameon=False,
                           bbox_to_anchor=(0.5, 1.0 + 0.02 * (legend_rows - 1)))

                # GridSpec already sets the spacing; tight_layout would fight it.
                out = os.path.join(figures_dir,
                                   f"workflow_{group_name}{figure_suffix}.png")
                plt.savefig(out, dpi=300, bbox_inches="tight")
                print(f"Saved {out}")
                plt.show()


if __name__ == "__main__":
    pass