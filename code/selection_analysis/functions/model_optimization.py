"""
Selection analysis: cross-validation quality vs downstream selection performance.

Fits a Gaussian process on a training set of a given size and records both
cross-validation and held-out test metrics, so CV quality can be related to
end-point active-learning performance.

One sequence representation per run: the ESM2 foundation model, or one of the
three baselines. See ``representations.py``.

Regression only.
"""

import gc
import random
import warnings

import numpy as np
import pandas as pd
import optuna
from Levenshtein import distance as levenshtein_distance
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    DotProduct
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import spearmanr

from functions.representations import SequenceRepresenter, REPRESENTATIONS


class ModelOptimization:
    
    def __init__(
        self,
        data_path,
        random_seed=42,
        initial_train_size=20,
        initial_test_size=500,
        emb_batch_size=64,
        representation="esm",
        model_name="facebook/esm2_t33_650M_UR50D",
        morgan_radius=2,
        morgan_bits=2048,
        ngram_range=(1, 3),
        ngram_vectorizer="tfidf",
        ngram_features=1024,
    ):

        if representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {REPRESENTATIONS}, got {representation!r}"
            )

        # Parameters
        self.data_path = data_path
        self.random_seed = random_seed
        self.initial_train_size = initial_train_size
        self.initial_test_size = initial_test_size
        self.emb_batch_size = emb_batch_size
        self.representation = representation
        self.model_name = model_name

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
        self.levenshtein_split = False

        # Seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def prep_data(self, seq_col="sequence", score_col=["score"]):
        
        df = pd.read_csv(self.data_path, sep=None, engine='python')
        required_cols = set([seq_col] + score_col)
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input file must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        
        # Store column names of the scores
        self.target_names = score_col

        # Convert sequence to string
        df["sequence"] = df[seq_col].astype(str)

        # Convert scores to float
        for col in score_col:
            df[col] = df[col].astype(float)

        # Keep only sequence + score columns
        df = df[["sequence"] + score_col].copy()

        self.df = df

        # Create a mapping from sequence → list of scores if multiple columns
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

    def allocate_counts(self, total_size, proportions):
        raw_counts = proportions * total_size
        
        floored_counts = np.floor(raw_counts).astype(int)
        
        remainder = total_size - floored_counts.sum()

        if remainder > 0:
            fractional_parts = raw_counts - floored_counts
            indices_to_increment = np.argsort(-fractional_parts.values)[:remainder]
            for pos in indices_to_increment:
                floored_counts.iloc[pos] += 1
        
        final_diff = total_size - floored_counts.sum()
        if final_diff != 0:
            floored_counts.iloc[-1] += final_diff
            
        return floored_counts.to_dict()


    def initialize_training_set(self, df_path=None, seq_col="sequence", score_col="score"):

        if df_path:
            df = pd.read_csv(df_path, sep=None, engine='python')
            required_cols = {seq_col, score_col}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Input file must contain columns: {required_cols}. Found: {df.columns.tolist()}")

            df["sequence"] = df[seq_col].astype(str)
            df["score"] = df[score_col].astype(float)
            df = df[["sequence", "score"]].copy()

            train_dict = dict(zip(df["sequence"], df["score"]))
            self.seq_to_score.update(train_dict)
            self.training_sequences = df["sequence"].tolist()
        else:
            remaining_size = max(0, self.initial_train_size)
            
            if self.levenshtein_split:
                seq_length = [len(i) for i in self.df_remaining["sequence"]]
                self.df_remaining["sequence_length"] = seq_length
                length_counts = self.df_remaining["sequence_length"].value_counts()
                length_proportions = length_counts / len(self.df_remaining)
                train_counts = self.allocate_counts(remaining_size, length_proportions)
                selected_rows = []
                # We select the farthest samples per sequence length from the test data
                for seq_len in np.unique(seq_length):
                    subset = self.df_remaining[self.df_remaining["sequence_length"] == seq_len]
                    if len(subset) == 0:
                        continue
                    
                    test_n = min(train_counts.get(seq_len, 0), len(subset))

                    col_name = f"lev_distance_seq_length_{seq_len}"

                    rows = (
                        subset
                        .sort_values(col_name, ascending=False)
                        .head(test_n)
                    )

                    selected_rows.append(rows)
                    self.df_remaining = self.df_remaining.drop(rows.index)

                rest_training_df = pd.concat(selected_rows).reset_index(drop=True)
                if len(rest_training_df) != remaining_size:
                    raise ValueError(f"Number of sequences for training was not achieved {len(rest_training_df)}")

            else:

                rest_training_df = self.df_remaining.sample(n=remaining_size, random_state=self.random_seed).reset_index(drop=True)
            
            self.training_sequences = rest_training_df["sequence"].tolist()

    def initialize_test_set(self, df_path=None, seq_col="sequence", score_col="score", levenshtein_split=False):
        
        if df_path:
            df = pd.read_csv(df_path, sep=None, engine='python')
            required_cols = {seq_col, score_col}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Input file must contain columns: {required_cols}. Found: {df.columns.tolist()}")

            df["sequence"] = df[seq_col].astype(str)
            df["score"] = df[score_col].astype(float)
            df = df[["sequence", "score"]].copy()

            self.seq_to_score = dict(zip(df["sequence"], df["score"]))
            self.test_sequences = df["sequence"].tolist()
        else:
            self.levenshtein_split = levenshtein_split
            test_size = min((len(self.df) - self.initial_train_size), self.initial_test_size)
            if test_size <= 0:
                raise ValueError("Not enough data for testing! Make sure there's enough data for testing")
            print(f"Number of data points for testing: {test_size}")

            if levenshtein_split:
                seq_length = [len(i) for i in self.df["sequence"]]
                self.df["sequence_length"] = seq_length
                length_counts = self.df["sequence_length"].value_counts()
                length_proportions = length_counts / len(self.df)
                test_counts = self.allocate_counts(test_size, length_proportions)

                self.df_remaining = self.df.copy()
                rng = np.random.default_rng(self.random_seed)
                test_indices = []
                # Calc levensthein distance per sequence length group and select most similar sequences per group
                for seq_len in np.unique(seq_length):
                    subset = self.df[self.df["sequence_length"] == seq_len].copy()
                    if len(subset) == 0:
                        continue
                    
                    test_n = min(test_counts.get(seq_len, 0), len(subset))

                    ref_idx = rng.choice(subset.index)
                    ref_seq = subset.loc[ref_idx, "sequence"]

                    subset["lev_distance"] = subset["sequence"].apply(
                        lambda s: levenshtein_distance(ref_seq, s)
                    )
                    
                    test_subset = subset.sort_values("lev_distance", ascending=True).head(test_n)
                    test_indices.extend(test_subset.index.tolist())

                    self.df_remaining.loc[subset.index, f"lev_distance_seq_length_{seq_len}"] = subset["lev_distance"]

                if len(test_indices) != test_size:
                    raise ValueError(f"Number of sequences for testing was not achieved {len(test_indices)}")
                self.test_df = self.df.loc[test_indices]
                self.df_remaining = self.df_remaining.drop(index=test_indices).reset_index(drop=True)
                self.test_df = self.test_df.reset_index(drop=True)
                self.test_sequences = self.test_df["sequence"].tolist()
            else:
                sampled = self.df.sample(n=test_size, random_state=self.random_seed)
                self.df_remaining = self.df.drop(sampled.index).reset_index(drop=True)
                self.test_df = sampled.reset_index(drop=True)
                self.test_sequences = self.test_df["sequence"].tolist()

    def optimize_all_models(self,
                            folds=10,
                            random_seed=42,
                            n_trials=100,
                            optuna_print=True):
        
        # Suppress convergence and feature name warnings
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        

        ## Seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load the foundation model only when it is the representation in use.
        self.load_model()

        # fit=True lets the n-gram vectorizer learn its vocabulary here, on
        # the training sequences only.
        X_train_temp = self.compute_embeddings(self.training_sequences, fit=True)

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
        n_samples = X_train_temp.shape[0]
        folds = min(folds, n_samples)
        
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_seed)

        seq_to_emb = {}
        for idx, seq in enumerate(self.training_sequences):
            seq_to_emb[seq] = X_train_temp[idx]

        seq_array = np.array(self.training_sequences)

        folds_per_label = {}

        for label_idx in range(n_targets):
            folds_list = []
            
            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(seq_array)):
            
                X_train_fold = np.array([seq_to_emb[self.training_sequences[idx]] for idx in train_idx])
                X_val_fold = np.array([seq_to_emb[self.training_sequences[idx]] for idx in valid_idx])

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
        
        # Data to be used for evaluating on test data
        X_train = X_train_temp
        X_test = self.compute_embeddings(self.test_sequences, fit=False)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
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

        results_df_per_label = {}
        all_predictions_per_label = {} 

        # Iterate over targets and models
        for label_idx in range(n_targets):
            y_test_col = y_test[:, label_idx]
            label_results = []
            predictions_per_model = {}

            print(f"\nOptimizing models for target {self.target_names[label_idx]} {label_idx + 1}/{n_targets}...")
            for name, cfg in model_configs.items():
                print(f"\nOptimizing {name}...")
                self.current_name = name

                self.mse_tracker = np.inf
                self.cv_scores = None

                ModelClass = cfg["class"]
                param_bounds = cfg["params"]
                
                def objective(trial):

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
                        
                        model = ModelClass(**params)
                        
                        fold_scores = []
                        y_vall_all = []
                        y_pred_all = []

                        for fold_idx, dataset_dict in enumerate(folds_per_label[f"label_{label_idx}"]):
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
                            y_vall_all.extend(y_val_fold.tolist())
                            y_pred_all.extend(y_pred.tolist())

                        mse = np.mean(fold_scores)

                        if mse < self.mse_tracker:
                            # Pooled across folds, matching how CV R2 is computed.
                            cv_rho, _ = spearmanr(y_vall_all, y_pred_all)
                            self.cv_scores = {
                                "CV RMSE": float(np.sqrt(mse)),
                                "CV MAE": float(mean_absolute_error(y_vall_all, y_pred_all)),
                                "CV R2": float(r2_score(y_vall_all, y_pred_all)),
                                "CV Spearman Correlation": float(cv_rho) if np.isfinite(cv_rho) else np.nan,
                            }
                            self.mse_tracker = mse

                        return mse
                    except Exception as e:
                        print("Trial failed:", e)
                        raise optuna.TrialPruned()

                optuna.logging.set_verbosity(optuna.logging.ERROR) 
                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_seed))
                study.optimize(objective, n_trials=n_trials, show_progress_bar=optuna_print)
                
                best_params = study.best_params

                mean_mse = study.best_value

                best_model = ModelClass(**best_params)
                best_model.fit(X_train, y_train[:, label_idx])

                # Predict on test data
                pred_score_test = best_model.predict(X_test)

                # Save predictions for this model and label
                predictions_per_model[self.current_name] = [
                    {"sequence": seq, "prediction": pred, "true_value": y_true}
                    for seq, pred, y_true in zip(self.test_sequences, pred_score_test, y_test_col)
                ]

                mse_test = mean_squared_error(y_test_col, pred_score_test)
                rmse_test = np.sqrt(mse_test)
                mae_test = mean_absolute_error(y_test_col, pred_score_test)
                r2_test = r2_score(y_test_col, pred_score_test)
                rho, p = spearmanr(y_test_col, pred_score_test)

                # If one selects top 10% predictions, what % of them are in the top 10% of true labels
                k = int(np.ceil(len(pred_score_test) * 0.1))
                top_indices_true = np.argpartition(y_test_col, -k)[-k:]
                top_indices_pred = np.argpartition(pred_score_test, -k)[-k:]
                overlap = len(set(top_indices_pred) & set(top_indices_true))
                pct_top20 = overlap / k * 100
                
                label_results.append({
                    "Model": self.current_name,
                    "Best Params": best_params,
                    "CV MSE": mean_mse,
                    "CV RMSE": self.cv_scores["CV RMSE"],
                    "CV MAE": self.cv_scores["CV MAE"],
                    "CV R2": self.cv_scores["CV R2"],
                    "CV Spearman Correlation": self.cv_scores["CV Spearman Correlation"],
                    "Test MSE": mse_test,
                    "Test RMSE": rmse_test,
                    "Test MAE": mae_test,
                    "Test R2": r2_test,
                    "Test Spearman Correlation": rho,
                    "Test Percent of top 10% pred in top 10% true": pct_top20
                })
                

                del study
                gc.collect()

            results_df = pd.DataFrame(label_results).sort_values("CV MSE", ascending=True)
            results_df_per_label[self.target_names[label_idx]] = results_df
            all_predictions_per_label[self.target_names[label_idx]] = predictions_per_model

        return results_df_per_label, all_predictions_per_label
    

if __name__ == "__main__":
    pass