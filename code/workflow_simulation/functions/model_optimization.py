import os
import random
import warnings
import numpy as np
from modlamp.descriptors import GlobalDescriptor
import pandas as pd
import torch
import optuna
import math
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    DotProduct
)
import string
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from transformers import AutoTokenizer, EsmModel
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path
from matplotlib.patches import Patch


class ModelOptimization:

    def __init__(
        self,
        data_path,
        random_seed=42,
        initial_train_size=20,
        emb_batch_size=64,
        model_name="facebook/esm2_t33_650M_UR50D"):

        # Parameters
        self.data_path = data_path
        self.random_seed = random_seed
        self.initial_train_size = initial_train_size
        self.emb_batch_size = emb_batch_size
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        self.df = None
        self.seq_to_score = {}
        self.training_sequences = []
        self.test_sequences = []

        # Seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

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
        if not self.classification:
            for col in score_col:
                df[col] = df[col].astype(float)
        else:
            self.label_enc = {}
            self.positive_label_freq = {}
            for col in score_col:
                positive_candidates = ["CPP", "Non-toxic"]
                column_values = df[col].astype(str)

                # Detect positive class
                positive_class = None
                for cls in positive_candidates:
                    if cls in column_values.values:
                        positive_class = cls
                        break

                if positive_class is None:
                    raise ValueError(
                        f"No positive class found in column '{col}'. "
                        f"Expected one of {positive_candidates}"
                    )

                positive_count = (column_values == positive_class).sum()
                total_count = len(column_values)
                positive_ratio = positive_count / total_count
                self.positive_label_freq[col] = positive_ratio
                    
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_enc[col] = le

        # Keep only sequence + score columns
        df = df[["sequence"] + score_col].copy()

        self.df = df

        self.y_all = df[score_col[0]]

        # Create a mapping from sequence → list of scores if multiple columns
        if len(score_col) == 1:
            self.seq_to_score = dict(zip(df[seq_col], df[score_col[0]]))
        else:
            self.seq_to_score = dict(zip(df[seq_col], df[score_col].values.tolist()))

        print("Finished preparing data!")

    def load_model(self, model_name="ESM"):

        self.model_str_name = model_name.lower() 
        if model_name.lower() == "esm":
            print(f"Loading base pretrained ESM model: {self.model_name}")
            model_source = self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=False)
            self.model = EsmModel.from_pretrained(model_source)

            self.model = self.model.to(self.device)
            self.model.eval()

            print("ESM model loaded. Using device:", self.device)


    def compute_embeddings(self, 
                           sequences, 
                           baseline=False, 
                           batch_size=None):
        if batch_size is None:
            batch_size = self.emb_batch_size

        foundation_embeddings = []
        X_physchem = []
        n = len(sequences)

        for i in range(0, n, batch_size):
            batch = sequences[i:i+batch_size]
            labels, seqs = zip(*batch)

            if not baseline:

                if self.model_str_name == "esm":
                    encoded = self.tokenizer(
                        list(seqs),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024  
                    ).to(self.device)

                    input_ids = encoded["input_ids"]
                    attention_mask = encoded["attention_mask"]

                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                        hidden_states = outputs.hidden_states

                    last_layer = hidden_states[-1:] 
                    stacked = torch.stack(last_layer, dim=0) 
                    mean_layers = stacked.mean(dim=0)

                    for j, seq in enumerate(seqs):
                        mask = attention_mask[j].bool().to(mean_layers.device)
                        seq_emb = mean_layers[j, mask].mean(dim=0)
                        foundation_embeddings.append(seq_emb.cpu().numpy())

            else:
                for j, seq in enumerate(seqs):
                    feats = self.compute_generalizable_features(seq)
                    X_physchem.append(feats.flatten())

        if baseline:
            all_embeddings = np.array(pd.DataFrame(X_physchem))
        else:
            all_embeddings = np.vstack(foundation_embeddings)

        return all_embeddings


    def initialize_training_set(self):

        size = max(0, self.initial_train_size)
        initial_training_df = self.df.sample(n=size, random_state=self.random_seed)
        self.training_sequences = initial_training_df["sequence"].tolist()
        self.test_df = self.df.drop(initial_training_df.index).reset_index(drop=True).copy()
        self.test_sequences = self.test_df["sequence"].tolist()

    def compute_generalizable_features(self, seq):
        
        # Global features
        global_desc = GlobalDescriptor([seq])
        global_desc.calculate_all()
        global_feats = global_desc.descriptor

        # Amino acid composition
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY' 
        seq_len = len(seq)
        aa_counts = [seq.count(aa)/seq_len if seq_len > 0 else 0 for aa in amino_acids]

        # Additional features inspired by HeliQuest
        additional_features = self.additional_features_fun(seq)


        combined_feats = np.concatenate([global_feats.flatten(), np.array(aa_counts), np.array(additional_features)]).reshape(1, -1)

        return combined_feats
    
    def additional_features_fun(self, seq):
        """
        Function for calculating amino acid type composition, hydrophobic moment and discrimation factor, similar to HeliQuest:
        Gautier R., Douguet D., Antonny B. and Drin G. HELIQUEST: a web server to screen sequences with specific α-helical properties. Bioinformatics. 2008 Sep 15;24(18):2101-2.

        Parameters
        ----------
        seq : str
        Amino acid sequence.

        Returns
        -------
        np.ndarray
        Feature array of shape (1, n_features).
        """
        def assign_hydrophobicity(sequence, scale='Fauchere-Pliska'):  
            """
            Assigns a hydrophobicity value to each amino acid in the sequence

            Author:
            Joao Rodrigues
            j.p.g.l.m.rodrigues@gmail.com
            https://github.com/JoaoRodrigues/hydrophobic_moment/tree/main
            """

            scales = {'Fauchere-Pliska': {'A':  0.31, 'R': -1.01, 'N': -0.60,
                                'D': -0.77, 'C':  1.54, 'Q': -0.22,
                                'E': -0.64, 'G':  0.00, 'H':  0.13,
                                'I':  1.80, 'L':  1.70, 'K': -0.99,
                                'M':  1.23, 'F':  1.79, 'P':  0.72,
                                'S': -0.04, 'T':  0.26, 'W':  2.25,
                                'Y':  0.96, 'V':  1.22},

            'Eisenberg': {'A':  0.25, 'R': -1.80, 'N': -0.64,
                        'D': -0.72, 'C':  0.04, 'Q': -0.69,
                        'E': -0.62, 'G':  0.16, 'H': -0.40,
                        'I':  0.73, 'L':  0.53, 'K': -1.10,
                        'M':  0.26, 'F':  0.61, 'P': -0.07,
                        'S': -0.26, 'T': -0.18, 'W':  0.37,
                        'Y':  0.02, 'V':  0.54},
            }

            hscale = scales.get(scale, None)
            if not hscale:
                raise KeyError('{} is not a supported scale. '.format(scale))

            hvalues = []
            for aa in sequence:
                sc_hydrophobicity = hscale.get(aa, None)
                if sc_hydrophobicity is None:
                    raise KeyError('Amino acid not defined in scale: {}'.format(aa))
                hvalues.append(sc_hydrophobicity)

            return hvalues

        def calculate_moment(array, angle=100):
            """Calculates the hydrophobic dipole moment from an array of hydrophobicity
            values. Formula defined by Eisenberg, 1982 (Nature). Returns the average
            moment (normalized by sequence length)

            uH = sqrt(sum(Hi cos(i*d))**2 + sum(Hi sin(i*d))**2),
            where i is the amino acid index and d (delta) is an angular value in
            degrees (100 for alpha-helix, 180 for beta-sheet).

            Author:
            Joao Rodrigues
            j.p.g.l.m.rodrigues@gmail.com
            https://github.com/JoaoRodrigues/hydrophobic_moment/tree/main
            """

            sum_cos, sum_sin = 0.0, 0.0
            for i, hv in enumerate(array):
                rad_inc = ((i*angle)*math.pi)/180.0
                sum_cos += hv * math.cos(rad_inc)
                sum_sin += hv * math.sin(rad_inc)
            return math.sqrt(sum_cos**2 + sum_sin**2) / len(array)


        def calculate_charge(sequence):
            """
            Calculates the charge of the peptide sequence at pH 7.4

            Author:
            Joao Rodrigues
            j.p.g.l.m.rodrigues@gmail.com
            https://github.com/JoaoRodrigues/hydrophobic_moment/tree/main
            """
            charge_dict = {'E': -1, 'D': -1, 'K': 1, 'R': 1}
            sc_charges = [charge_dict.get(aa, 0) for aa in sequence]
            return sum(sc_charges)


        def calculate_discrimination(mean_uH, total_charge):
            """
            Returns a discrimination factor according to Rob Keller (IJMS, 2011)
            A sequence with d>0.68 can be considered a potential lipid-binding region.

            Author:
            Joao Rodrigues
            j.p.g.l.m.rodrigues@gmail.com
            https://github.com/JoaoRodrigues/hydrophobic_moment/tree/main
            """
            d = 0.944*mean_uH + 0.33*total_charge
            return d


        def calculate_composition(sequence):
            """
            Returns a dictionary with percentages per classes

            Author:
            Joao Rodrigues
            j.p.g.l.m.rodrigues@gmail.com
            https://github.com/JoaoRodrigues/hydrophobic_moment/tree/main
            """

            # Residue character table
            polar_aa = set(('S', 'T', 'N', 'H', 'Q', 'G'))
            speci_aa = set(('P', 'C'))
            apolar_aa = set(('A', 'L', 'V', 'I', 'M'))
            charged_aa = set(('E', 'D', 'K', 'R'))
            aromatic_aa = set(('W', 'Y', 'F'))

            n_p, n_s, n_a, n_ar, n_c = 0, 0, 0, 0, 0
            tot = 0
            for aa in sequence:
                tot += 1
                if aa in polar_aa:
                    n_p += 1
                elif aa in speci_aa:
                    n_s += 1
                elif aa in apolar_aa:
                    n_a += 1
                elif aa in charged_aa:
                    n_c += 1
                elif aa in aromatic_aa:
                    n_ar += 1

            comp_dict = {'polar': n_p, 'special': n_s,
                        'apolar': n_a, 'charged': n_c, 'aromatic': n_ar}
            n_tot_pol = comp_dict['polar'] + comp_dict['charged']
            n_tot_apol = comp_dict['apolar'] + comp_dict['aromatic'] + comp_dict['special'] 
            n_charged = comp_dict['charged']  
            n_aromatic = comp_dict['aromatic']  

            return torch.cat([
                    torch.tensor([n_tot_pol/tot]), # polar
                    torch.tensor([n_tot_apol/tot]), # apolar          
                    torch.tensor([n_charged/tot]), # charged
                    torch.tensor([n_aromatic/tot]) # aromatic
                ])
        
        # HeliQuest inspired features
        z = calculate_charge(seq)
        seq_h = assign_hydrophobicity(seq)
        av_uH = calculate_moment(seq_h)
        d = calculate_discrimination(av_uH, z)
        aa_type_comp = calculate_composition(seq)

        additional_features = torch.cat([
            torch.tensor([av_uH]),
            torch.tensor([d]),
            aa_type_comp 
        ])
        return additional_features


    def optimize_all_models(self, 
                            folds=10, 
                            random_seed=42, 
                            n_trials=100, 
                            baseline=False, 
                            optuna_print=True):

        # Suppress convergence and feature name warnings
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        

        ## Seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        label_seq = [(f"train_{i}", seq) for i, seq in enumerate(self.training_sequences)]
        X_train_temp = self.compute_embeddings(sequences=label_seq, baseline=baseline)

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
        n_samples = X_train_temp.shape[0]
        folds = min(folds, n_samples)
        self.folds = folds
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_seed)

        # Split data
        self.baseline = baseline

        seq_array = np.array(self.training_sequences)

        folds_per_label = {}

        for label_idx in range(n_targets):
            folds_list = []
            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(seq_array)):
                train_df_temp = seq_array[train_idx]
                val_df_temp = seq_array[valid_idx]

                label_seq_train = [(f"train_{i}", seq) for i, seq in enumerate(train_df_temp)]
                X_train_fold = self.compute_embeddings(sequences=label_seq_train, baseline=baseline)

                label_seq_val = [(f"val_{i}", seq) for i, seq in enumerate(val_df_temp)]
                X_val_fold = self.compute_embeddings(sequences=label_seq_val, baseline=baseline)

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
        
        if not self.classification:
            model_configs = {
                #"Ridge": {"class": Ridge, "params": {"alpha": ("float_log", 1e-6, 1e6)}}
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
        else:
            model_configs = {
                "ExtraTrees": {
                    "class": ExtraTreesClassifier,
                    "params": {
                        "n_estimators": ("int", 50, 300),
                        "max_depth": ("int", 2, 15),
                        "min_samples_split": ("int", 2, 10),
                        "min_samples_leaf": ("int", 1, 8),
                        "max_features": ("categorical", ["sqrt", "log2", None])
                }}
            }

        # Iterate over targets and models
        for label_idx in range(n_targets):

            print(f"\nOptimizing models for target {self.target_names[label_idx]} {label_idx + 1}/{n_targets}...")
            for name, cfg in model_configs.items():
                print(f"\nOptimizing {name}...")
                self.current_name = name

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

                        for fold_idx, dataset_dict in enumerate(folds_per_label[f"label_{label_idx}"]):
                            model_fold = clone(model)
                            train_df = dataset_dict["train"]
                            val_df = dataset_dict["validation"]

                            X_train_fold = np.vstack(train_df["sequence"])
                            X_val_fold = np.vstack(val_df["sequence"])

                            if not self.classification:
                                y_train_fold = np.vstack(train_df["score"]).ravel()
                                y_val_fold = np.vstack(val_df["score"]).ravel()

                                model_fold.fit(X_train_fold, y_train_fold)
                                y_pred = model_fold.predict(X_val_fold)

                                mse = mean_squared_error(y_val_fold, y_pred)
                                fold_scores.append(mse)

                            else:
                                y_train_fold = train_df["score"].astype(int)
                                y_val_fold = val_df["score"].astype(int)

                                model_fold.fit(X_train_fold, y_train_fold)

                                y_proba = model_fold.predict_proba(X_val_fold)
                                label_enc = self.label_enc[self.target_names[label_idx]]
                                loss = log_loss(y_val_fold, y_proba, labels=label_enc.transform(label_enc.classes_))

                                fold_scores.append(loss)

                        return np.mean(fold_scores)
                    except:
                        raise optuna.TrialPruned()
                
                optuna.logging.set_verbosity(optuna.logging.ERROR) 
                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_seed))
                study.optimize(objective, n_trials=n_trials, show_progress_bar=optuna_print)
                
                best_params = study.best_params
                
                best_model = ModelClass(**best_params)

        return best_model, y_train, y_test
    
    def pred(self, train_seqs, y_train, test_seqs, label_idx, best_model, score_col):

        label_seq = [(f"train_{i}", seq) for i, seq in enumerate(train_seqs)]
        X_train = self.compute_embeddings(sequences=label_seq, baseline=self.baseline)

        label_seq_test = [(f"test_{i}", seq) for i, seq in enumerate(test_seqs)]
        X_test = self.compute_embeddings(sequences=label_seq_test, baseline=self.baseline)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = clone(best_model)
        model.fit(X_train, y_train[:, label_idx])

        if not self.classification:
            pred_score_test = model.predict(X_test)
            return pred_score_test
        else:
            proba = model.predict_proba(X_test)
            le = self.label_enc[score_col[0]]
            classes = le.classes_

            # Possible positive class names
            positive_candidates = ["CPP", "Non-toxic"]

            # Find which one exists in the model
            positive_class = None
            for cls in positive_candidates:
                if cls in classes:
                    positive_class = cls
                    break

            if positive_class is None:
                raise ValueError(f"None of {positive_candidates} found in model.classes_: {classes}")

            pos_index = list(classes).index(positive_class)

            positive_proba = proba[:, pos_index]
            return positive_proba

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
                        baseline=False, 
                        classification=False,
                        optuna_print=True):
        """
        Run an iterative acquisition simulation to evaluate model-driven
        selection performance versus a random baseline.

        This function simulates an active-learning–style workflow where, at each
        round, a model selects new sequences from a test pool based on predicted
        scores.
        """
        os.makedirs("./simulation_results", exist_ok=True)
        
        if self.model_name.split("/")[0].lower() == "facebook":
            self.load_model()
        else:
            raise ValueError(f"Foundation model needs to be an ESM2 model from Huggingface...")

        self.classification = classification
        target_candidates = ["CPP", "Non-toxic"]

        n_seeds = n_seeds
        start_seed = random_seed
        seeds = [start_seed + i for i in range(n_seeds)]

        rows = []

        for data_idx, data_path in enumerate(data_paths):

            ds_name = dataset_names[data_idx]

            self.data_path = data_path

            for seed in seeds:
                self.random_seed=seed
                self.prep_data(score_col=score_col)
                self.initialize_training_set()

                k = int(np.ceil(len(self.y_all) * 0.1))
                all_sequences = list(self.seq_to_score.keys())

                top10pct_highest_idx = np.argsort(self.y_all)[-k:]
                top10pct_highest_seqs = {all_sequences[i] for i in top10pct_highest_idx}

                all_selected_seqs = set(self.training_sequences)

                # Simulation
                
                round_idx = -1
                while len(self.training_sequences) < int(max_num_samp):
                    round_idx += 1

                    best_model, y_train, y_test = self.optimize_all_models(
                        folds=folds, 
                        random_seed=seed, 
                        n_trials=n_trials, 
                        baseline=baseline, 
                        optuna_print=optuna_print
                    )

                    pred_scores = self.pred(
                        train_seqs=self.training_sequences,
                        y_train=y_train,
                        test_seqs=self.test_sequences,
                        label_idx=0,
                        best_model=best_model,
                        score_col=score_col
                    )

                    if not self.classification:

                        if round_idx == 0:
                        
                            ### Selection performance normalized between the overall mean value and the highest 
                            ### possible mean value based on the selection at each round
                            mean_all = np.mean(self.y_all)
                            highest_idx = np.argsort(self.y_all)[-new_samp_per_step:][::-1]
                            highest_mean = np.mean(self.y_all[np.array(highest_idx)])
                            selected_mean = np.mean(y_train[:, 0])
                            norm_selected_mean = (selected_mean - mean_all) / (highest_mean - mean_all)
                            rows.append([ds_name, seed, len(self.training_sequences), "Norm. Target Mean (Round)", norm_selected_mean, baseline])


                        # Select top predicted
                        pred_idx = np.argsort(pred_scores)[-new_samp_per_step:][::-1]

                        ### Selection performance normalized between the overall mean value and the highest 
                        ### possible mean value based on the selection at each round
                        mean_all = np.mean(y_test[:, 0])
                        highest_idx = np.argsort(y_test[:, 0])[-new_samp_per_step:][::-1]
                        highest_mean = np.mean(y_test[:, 0][np.array(highest_idx)])
                        selected_mean = np.mean(y_test[:, 0][np.array(pred_idx)])
                        norm_selected_mean = (selected_mean - mean_all) / (highest_mean - mean_all)
                        rows.append([ds_name, seed, len(self.training_sequences)+new_samp_per_step, "Norm. Target Mean (Round)", norm_selected_mean, baseline])

                        ### Cumulative performance of entire training data set normalized from overall mean value to
                        ### highest possible mean based on current number of samples in training data
                        current_mean = np.mean(y_train[:, 0])
                        highest_idx = np.argsort(self.y_all)[-len(y_train[:, 0]):][::-1]
                        highest_mean = np.mean(self.y_all[np.array(highest_idx)])
                        mean_all = np.mean(self.y_all)
                        norm_current_mean = (current_mean - mean_all) / (highest_mean - mean_all)
                        rows.append([ds_name, seed, len(self.training_sequences), "Norm. Target Mean (Cumulative)", norm_current_mean, baseline])

                        ### Percentage of actual top 50 samples that have already been taken 
                        overlap = len(top10pct_highest_seqs & all_selected_seqs)
                        pct_top10pct = overlap / k * 100
                        rows.append([ds_name, seed, len(self.training_sequences), "Top 10% peptides selected (%)", pct_top10pct, baseline])

                        ### Difference between mean score of the top 10% peptides that are avilable at each round and the mean of all data
                        ### Normalized between max mean and global mean
                        k_avail = int(np.ceil(len(y_test[:,0]) * 0.1))

                        top10_idx = np.argsort(y_test[:,0])[-k_avail:]
                        top10_mean = np.mean(y_test[:,0][top10_idx])

                        k_global = int(np.ceil(len(self.y_all) * 0.1))
                        global_top10_mean = np.mean(np.sort(self.y_all)[-k_global:])

                        top10_gap_norm = (top10_mean - mean_all) / (global_top10_mean - mean_all)

                        rows.append([
                            ds_name,
                            seed,
                            len(self.training_sequences),
                            "Remaining Top10 Quality",
                            top10_gap_norm,
                            baseline
                        ])

                    else:
                        
                        if round_idx == 0:
                        
                            ### Enrichment factor per selection round
                            le = self.label_enc[score_col[0]]
                            total = len(y_train[:, 0])

                            for label in target_candidates:
                                if label in le.classes_:
                                    encoded_value = le.transform([label])[0]
                                    portion = np.sum(y_train[:, 0] == encoded_value) / total

                            ef_per_round = (portion / self.positive_label_freq[score_col[0]]) - 1
                            rows.append([ds_name, seed, len(self.training_sequences), "Enrichment Factor (Round)", ef_per_round, baseline])

                        # Select top predicted
                        pred_idx = np.argsort(pred_scores)[-new_samp_per_step:][::-1]
                        y_selection = y_test[:, 0][np.array(pred_idx)]

                        ### Enrichment factor per selection round
                        le = self.label_enc[score_col[0]]
                        total = len(y_selection)

                        for label in target_candidates:
                            if label in le.classes_:
                                encoded_value = le.transform([label])[0]
                                portion = np.sum(y_selection == encoded_value) / total
                                positive_label_freq = np.sum(y_test[:, 0] == encoded_value) / len(y_test[:, 0])

                        ef_per_round = (portion / positive_label_freq) - 1
                        rows.append([ds_name, seed, len(self.training_sequences)+new_samp_per_step, "Enrichment Factor (Round)", ef_per_round, baseline])

                        ### Enrichment Factor (Cumulative)
                        le = self.label_enc[score_col[0]]
                        total = len(y_train[:, 0])

                        for label in target_candidates:
                            if label in le.classes_:
                                encoded_value = le.transform([label])[0]
                                portion = np.sum(y_train[:, 0] == encoded_value) / total

                        ef_cumulative = (portion / self.positive_label_freq[score_col[0]]) - 1
                        rows.append([ds_name, seed, len(self.training_sequences), "Enrichment Factor (Cumulative)", ef_cumulative, baseline])


                    # Update new training set
                    selected_seqs = [self.test_sequences[i] for i in pred_idx]
                    self.training_sequences.extend(selected_seqs)

                    # remove from test set
                    mask = np.ones(len(self.test_sequences), dtype=bool)
                    mask[pred_idx] = False
                    self.test_sequences = [s for i, s in enumerate(self.test_sequences) if mask[i]]
                    self.test_df = self.test_df[self.test_df["sequence"].isin(self.test_sequences)].reset_index(drop=True)

                    all_selected_seqs.update(selected_seqs)

                ### Cumulative performance of entire training data set normalized from overall mean value to
                ### highest possible mean based on current number of samples in training data
                y_train = []
                for s in self.training_sequences:
                    y_train.append(self.seq_to_score[s])
                y_train = np.array(y_train)
                if y_train.ndim == 1:
                    y_train = y_train.reshape(-1, 1)
                    
                if not self.classification:
                    current_mean = np.mean(y_train[:, 0])
                    highest_idx = np.argsort(self.y_all)[-len(y_train[:, 0]):][::-1]
                    highest_mean = np.mean(self.y_all[np.array(highest_idx)])
                    mean_all = np.mean(self.y_all)
                    norm_current_mean = (current_mean - mean_all) / (highest_mean - mean_all)
                    rows.append([ds_name, seed, len(self.training_sequences), "Norm. Target Mean (Cumulative)", norm_current_mean, baseline])
                    
                    ### Percentage of actual top 50 samples that have already been taken 
                    overlap = len(top10pct_highest_seqs & all_selected_seqs)
                    pct_top10pct = overlap / k * 100
                    rows.append([ds_name, seed, len(self.training_sequences), "Top 10% peptides selected (%)", pct_top10pct, baseline])

                    ### Difference between mean score of the top 10% peptides that are avilable at each round and the mean of all data
                    ### Normalized between max mean and global mean
                    k_avail = int(np.ceil(len(y_test[:,0]) * 0.1))

                    top10_idx = np.argsort(y_test[:,0])[-k_avail:]
                    top10_mean = np.mean(y_test[:,0][top10_idx])

                    k_global = int(np.ceil(len(self.y_all) * 0.1))
                    global_top10_mean = np.mean(np.sort(self.y_all)[-k_global:])

                    top10_gap_norm = (top10_mean - mean_all) / (global_top10_mean - mean_all)

                    rows.append([
                        ds_name,
                        seed,
                        len(self.training_sequences),
                        "Remaining Top10 Quality",
                        top10_gap_norm,
                        baseline
                    ])
                else:
                    ### Enrichment Factor (Cumulative)
                    le = self.label_enc[score_col[0]]
                    total = len(y_train[:, 0])

                    for label in target_candidates:
                        if label in le.classes_:
                            encoded_value = le.transform([label])[0]
                            portion = np.sum(y_train[:, 0] == encoded_value) / total

                    ef_cumulative = (portion / self.positive_label_freq[score_col[0]]) - 1
                    rows.append([ds_name, seed, len(self.training_sequences), "Enrichment Factor (Cumulative)", ef_cumulative, baseline])
                    
        df_all = pd.DataFrame(
            rows,
            columns=["Dataset", "Seed", "Num_samples", "Metric", "Value", "Baseline"]
        )

        def save_or_append(df, filepath):
            filepath = Path(filepath)
            
            if filepath.exists():
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                df.to_csv(filepath, mode="w", header=True, index=False)
        
        save_or_append(df_all, "./simulation_results/df_all.csv")

        
    def visualize(self, dataset_names):

        os.makedirs("./simulation_results/figures", exist_ok=True)
        df = pd.read_csv("./scripts/simulation_results/df_all.csv", sep=None, engine='python')
        df.loc[df["Dataset"] == "AMP_Ecoli", "Dataset"] = "Short AMPs"

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

        data_size_dict = {"A0A247D711_LISMN": 1653,
                          "DN7A_SACS2": 1008,
                          "ENVZ_ECOLI": 1121,
                          "FKBP3_HUMAN": 1237,
                          "MAFG_MOUSE_sub": 1429,
                          "POLG_PESV_sub": 5130,
                          "SBI_STAAM": 1025,
                          "SDA_BACSU": 2770,
                          "SOX30_HUMAN": 1010,
                          "YNZC_BACSU_sub": 2300,
                          "Short AMPs": 1212}
        data_sizes = np.linspace(20, 200, 10)
        y_pct_random = {}
        for (data_name, data_size) in data_size_dict.items():
            y_pct_random[data_name] = [i / data_size * 100 for i in data_sizes]

        metrics_dict = {"per_round": ["Norm. Target Mean (Round)", "Enrichment Factor (Round)"],
                   "cumulative": ["Norm. Target Mean (Cumulative)", "Enrichment Factor (Cumulative)"],
                   "selection": ["Top 10% peptides selected (%)"]}
        palette = ['#e6194b', 
                   '#3cb44b', 
                   '#ffe119', 
                   '#4363d8', 
                   '#f58231', 
                   '#911eb4', 
                   '#46f0f0', 
                   '#f032e6', 
                   '#bcf60c', 
                   '#fabebe', 
                   '#008080', 
                   '#e6beff', 
                   '#9a6324', 
                   '#fffac8', 
                   '#800000', 
                   '#aaffc3', 
                   '#808000', 
                   '#ffd8b1', 
                   '#000075', 
                   '#808080', 
                   '#ffffff', 
                   '#000000']
        color_gpr = "#7eb9db"
        color_desc = "#e56e8c"
        color_et = "#41431B"
        color_et_desc = "#AEB784"
        dataset_color = {"False": color_gpr,
                         "True": color_desc,
                         "ET": color_et,
                         "ET_desc": color_et_desc}

        letters = list(string.ascii_lowercase)
        
        metric_idx = -1
        for metrics in metrics_dict.values():
            metric_idx += 1

            # ---- Create figure ----
            fig = plt.figure(figsize=(14, 16))
            gs = GridSpec(5, 3, figure=fig, hspace=0.65, wspace=0.02)
            if metric_idx == 2:
                fig = plt.figure(figsize=(14, 14))
                gs = GridSpec(4, 3, figure=fig, hspace=0.65, wspace=0.02)

            axes = []
            for metric in metrics:
                print(metric)

                df_metric = df[df["Metric"] == metric]
                if metric == "Norm. Target Mean (Round)":
                    metric = "Target mean (Round)"
                elif metric == "Norm. Target Mean (Cumulative)":
                    metric = "Target mean (Cumulative)"

                # Compute mean/std across seeds
                summary = (
                    df_metric
                    .groupby(["Dataset", "Num_samples", "Baseline"])["Value"]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                # ---- Compute global y-limits across ALL datasets ----
                y_min = np.inf
                y_max = -np.inf

                for dataset in dataset_names:
                    df_tmp = summary[summary["Dataset"] == dataset]
                    if df_tmp.empty:
                        continue

                    lower = (df_tmp["mean"] - df_tmp["std"]).min()
                    upper = (df_tmp["mean"] + df_tmp["std"]).max()

                    y_min = min(y_min, lower)
                    y_max = max(y_max, upper)

                # Add small padding
                padding = 0.05 * (y_max - y_min)
                y_min -= padding
                y_max += padding
                y_min = math.floor(y_min * 10) / 10
                y_max = math.ceil(y_max * 10) / 10

                for i, dataset in enumerate(dataset_names):
                    if i >= 11 and metric_idx == 2:
                        continue

                    row = i // 3
                    col = i % 3

                    if row == 3 and col == 2:
                        row = 4
                        col = 0
                    elif row == 4 and col == 0:
                        row = 4
                        col = 1

                    ax = fig.add_subplot(gs[row, col])
                    axes.append(ax)

                    ax.text(
                        0.02, 1.12,
                        letters[i],
                        transform=ax.transAxes,
                        fontsize=11,
                        fontweight="bold",
                        fontfamily="DejaVu Serif",
                        va="top",
                        ha="left"
                    )

                    df_plot = summary[summary["Dataset"] == dataset]

                    if df_plot.empty:
                        ax.axis("off")
                        continue
                    
                    if dataset not in ["CellPPD", "ToxinPred3"]:
                        for baseline in [False, True]:
                            df_temp = df_plot[df_plot["Baseline"] == baseline]
                            x = df_temp["Num_samples"].values
                            y_mean = df_temp["mean"].values
                            y_std = df_temp["std"].values

                            color = dataset_color[str(baseline)]

                            # Mean line
                            ax.plot(x, y_mean, linewidth=2.2, color=color, label="GPR + Descriptors (Baseline)" if baseline else "GPR + ESM2 (SCARSE)")

                            # Std shading
                            ax.fill_between(
                                x,
                                y_mean - y_std,
                                y_mean + y_std,
                                alpha=0.25,
                                color=color
                            )
                    else:
                        for baseline in [False, True]:
                            df_temp = df_plot[df_plot["Baseline"] == baseline]
                            x = df_temp["Num_samples"].values
                            y_mean = df_temp["mean"].values
                            y_std = df_temp["std"].values

                            color = dataset_color["ET" if baseline else "ET_desc"]

                            # Mean line
                            ax.plot(x, y_mean, linewidth=2.2, color=color, label="ET + Descriptors (Baseline)" if baseline else "ET + ESM2 (SCARSE)")

                            # Std shading
                            ax.fill_between(
                                x,
                                y_mean - y_std,
                                y_mean + y_std,
                                alpha=0.25,
                                color=color
                            )

                    if col == 1 and row == 0 and metric != "Top 10% peptides selected (%)":
                        color_map = {
                            "GPR + ESM2 (SCARSE)": "#7eb9db",
                            "GPR + Descriptors (Baseline)": "#e56e8c",
                            "ET + ESM2 (SCARSE)": "#41431B",
                            "ET + Descriptors (Baseline)": "#AEB784"
                        }
                        legend_handles = [Patch(facecolor=color_map[k], label=k) for k in color_map]

                        ax.legend(
                            handles=legend_handles,
                            loc="upper center",
                            ncol=4,
                            bbox_to_anchor=(0.5, 1.5),
                            frameon=False
                        )
                    elif col == 1 and row == 0 and metric == "Top 10% peptides selected (%)":
                        handles, labels = ax.get_legend_handles_labels() 
                        ax.legend( handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.5), frameon=False)

                    # Random baseline
                    if metric_idx == 2 and dataset in y_pct_random:
                        ax.plot(
                            data_sizes,
                            y_pct_random[dataset],
                            color="red",
                            linewidth=1.8,
                            linestyle="--",
                            label="Random"
                        )

                    ax.set_title(dataset)

                    # ---- Set identical y-limits ----
                    if metric_idx == 2:
                        ax.set_ylim(0, 40)
                        yticks = np.linspace(0, 100, 11)
                        ax.set_yticks(yticks)
                    elif row != 4:
                        ax.set_ylim(-0.6, 1.0)
                        yticks = np.linspace(-0.6, 1.0, 9)
                        ax.set_yticks(yticks)
                    else:
                        if metric_idx == 0:
                            ax.set_ylim(-0.5, 2.5)
                            yticks = np.linspace(-0.5, 2.5, 7)
                            ax.set_yticks(yticks)
                        else:
                            ax.set_ylim(-0.6, 1.2)
                            yticks = np.linspace(-0.6, 1.2, 10)
                            ax.set_yticks(yticks)

                    # ---- Only left column shows y-axis ----
                    if col == 0:
                        ax.set_ylabel(metric)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])

                    ax.set_xlabel("n peptides screened")
                    ax.set_xticks(np.arange(20, 201, 20))

                    for spine in ax.spines.values():
                        spine.set_visible(True)

                    ax.grid(True, axis="y")
                    ax.grid(False, axis="x")

            # Remove unused panels if < 15 datasets
            total_plots = len(dataset_names)
            for j in range(total_plots, 15):
                if j >= 11 and metric_idx == 2:
                    continue
                row = j // 3
                col = j % 3
                ax = fig.add_subplot(gs[row, col])
                ax.axis("off")


            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            plt.show()

if __name__ == "__main__":
    pass
