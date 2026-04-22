import os
import random
import warnings
import numpy as np
import gc
from modlamp.descriptors import GlobalDescriptor
import pandas as pd
import torch
import optuna
import math
from Levenshtein import distance as levenshtein_distance 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    DotProduct
)
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, log_loss
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import spearmanr
from transformers import AutoTokenizer, EsmModel


class ModelOptimization:
    
    def __init__(
        self,
        data_path,
        random_seed=42,
        initial_train_size=20,
        initial_test_size=500,
        emb_batch_size=64,
        classification=False,
        model_name="facebook/esm2_t33_650M_UR50D"):

        # Parameters
        self.data_path = data_path
        self.random_seed = random_seed
        self.initial_train_size = initial_train_size
        self.initial_test_size = initial_test_size
        self.emb_batch_size = emb_batch_size
        self.model_name = model_name
        self.classification = classification
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.esmfold_tokenizer = None

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
            for col in score_col:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_enc[col] = le

        # Keep only sequence + score columns
        df = df[["sequence"] + score_col].copy()

        self.df = df

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
                           baseline=True, 
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


    def initialize_training_set(self):

        remaining_size = max(0, self.initial_train_size)

        if self.levenshtein_split and self.classification:
            selected_rows = []
            for label_idx in range(len(self.target_names)):
                target_col = self.target_names[label_idx]

                class_counts = self.df_remaining[target_col].value_counts(normalize=True)
                class_sizes = self.allocate_counts(
                    remaining_size,
                    class_counts
                )

                # select FARTEST samples per class
                for cls, n_cls in class_sizes.items():
                    
                    df_cls = self.df_remaining[self.df_remaining[target_col] == cls].copy()
                    seq_length = [len(i) for i in df_cls["sequence"]]
                    df_cls["sequence_length"] = seq_length
                    length_counts = df_cls["sequence_length"].value_counts()
                    length_proportions = length_counts / len(df_cls["sequence_length"])
                    train_counts = self.allocate_counts(class_sizes[cls], length_proportions)

                    # Per class we select the farthest samples per sequence length from the test data
                    for seq_len in np.unique(seq_length):
                        
                        subset = df_cls[df_cls["sequence_length"] == seq_len]
                        if len(subset) == 0:
                            continue
                        
                        test_n = min(train_counts.get(seq_len, 0), len(subset))
                        #print(test_n)

                        col_name = f"lev_distance_{target_col}_{cls}_seq_length_{seq_len}"

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
        
        elif self.levenshtein_split and not self.classification:
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

        elif self.classification:
            
            # To make sure we have at least 2 samples per label
            two_per_cls_df = []

            # Sample 2 per class
            for label_idx in range(len(self.target_names)):
                target_col = self.target_names[label_idx]

                class_counts = self.df_remaining[target_col].value_counts(normalize=True)
                for cls in class_counts.keys():
                    df_cls = self.df_remaining[self.df_remaining[target_col] == cls].copy()

                    n = min(2, len(df_cls))  # safety
                    if n < 2:
                        raise ValueError("Not enough samples per class!")
                    
                    sampled_cls_df = df_cls.sample(n=n, random_state=self.random_seed)

                    two_per_cls_df.append(sampled_cls_df)

            # Combine per-class samples
            two_per_cls_df = pd.concat(two_per_cls_df, ignore_index=True)

            remaining_needed = remaining_size - len(two_per_cls_df)

            remaining_df = (
                self.df_remaining
                .drop(two_per_cls_df.index)
                .sample(n=remaining_needed, random_state=self.random_seed)
            )
            rest_training_df = pd.concat([two_per_cls_df, remaining_df], ignore_index=True).reset_index(drop=True)

            if len(rest_training_df) != remaining_size:
                raise ValueError(f"Number of sequences for training was not achieved {len(test_indices)}")

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

            if levenshtein_split and self.classification:
                rng = np.random.default_rng(self.random_seed)
                test_rows = []
                self.df_remaining = self.df.copy()

                # Loop over all target columns for multi-output
                for label_idx in range(len(self.target_names)):
                    target_col = self.target_names[label_idx]

                    # Compute class proportions
                    class_counts = self.df[target_col].value_counts(normalize=True)
                    class_sizes = self.allocate_counts(
                        test_size,
                        class_counts
                    )

                    # Per-class Levenshtein sampling
                    test_indices = []
                    for cls, cls_test_size in class_sizes.items():
                        if cls_test_size <= 0:
                            continue
                        
                        df_cls = self.df_remaining[self.df_remaining[target_col] == cls].copy()

                        seq_length = [len(i) for i in df_cls["sequence"]]
                        df_cls["sequence_length"] = seq_length
                        length_counts = df_cls["sequence_length"].value_counts()
                        length_proportions = length_counts / len(df_cls)
                        test_counts = self.allocate_counts(class_sizes[cls], length_proportions)

                        # Calc levensthein distance per sequence length group and select most similar sequences per group
                        for seq_len in np.unique(seq_length):
                            subset = df_cls[df_cls["sequence_length"] == seq_len].copy()
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

                            self.df_remaining.loc[subset.index, f"lev_distance_{target_col}_{cls}_seq_length_{seq_len}"] = subset["lev_distance"]

                if len(test_indices) != test_size:
                    raise ValueError(f"Number of sequences for testing was not achieved {len(test_indices)}")
                self.test_df = self.df.loc[test_indices]
                self.df_remaining = self.df_remaining.drop(index=test_indices).reset_index(drop=True)
                self.test_df = self.test_df.reset_index(drop=True)
                self.test_sequences = self.test_df["sequence"].tolist()

            elif levenshtein_split and not self.classification:
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
    
    def compute_roc_auc(self, model, X_test, y_test):

        # Get predicted probabilities
        proba = model.predict_proba(X_test)

        # Number of unique classes
        n_classes = len(model.classes_)

        if n_classes == 2:
            # Binary classification
            # Use probability of the positive class (class with higher label by default)
            positive_class_index = 1
            roc_auc = roc_auc_score(y_test, proba[:, positive_class_index])

        else:
            # Multiclass classification
            roc_auc = roc_auc_score(
                y_test,
                proba,
                multi_class="ovr",  # One-vs-Rest (most common choice)
                average="weighted"  # Handles class imbalance
            )

        return roc_auc

    def optimize_all_models(self, 
                            folds=10, 
                            random_seed=42, 
                            n_trials=100, 
                            baseline=True, 
                            optuna_print=True):
        
        # Suppress convergence and feature name warnings
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        

        ## Seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Load foundation model
        if self.model_name.split("/")[0].lower() == "facebook":
            self.load_model()
        else:
            raise ValueError(f"Foundation model needs to be an ESM2 model from Huggingface...")

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
        n_samples = X_train_temp.shape[0]
        folds = min(folds, n_samples)
        
        if self.classification:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_seed)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=random_seed)

        seq_to_emb = {}
        for idx, seq in enumerate(self.training_sequences):
            seq_to_emb[seq] = X_train_temp[idx]

        seq_array = np.array(self.training_sequences)

        folds_per_label = {}

        for label_idx in range(n_targets):
            folds_list = []
            
            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(seq_array) if not self.classification else cv.split(seq_array, y_train[:, label_idx])):
            
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
        label_seq_test = [(f"test_{i}", seq) for i, seq in enumerate(self.test_sequences)]
        X_test = self.compute_embeddings(sequences=label_seq_test, baseline=baseline)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if not self.classification:
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

                                #if hasattr(model_fold, "predict_proba"):
                                y_proba = model_fold.predict_proba(X_val_fold)
                                label_enc = self.label_enc[self.target_names[label_idx]]
                                loss = log_loss(y_val_fold, y_proba, labels=label_enc.transform(label_enc.classes_))
                                #else:
                                #    y_pred = model_fold.predict(X_val_fold)
                                #    loss = 1.0 - accuracy_score(y_val_fold, y_pred)

                                fold_scores.append(loss)

                        return np.mean(fold_scores)
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

                if not self.classification:

                    predictions_per_model[self.current_name] = [
                        {"sequence": seq, "prediction": pred, "true_value": y_true}
                        for seq, pred, y_true in zip(self.test_sequences, pred_score_test, y_test_col)
                    ]

                    mse_test = mean_squared_error(y_test_col, pred_score_test)
                    rmse_test = np.sqrt(mse_test)
                    mae_test = mean_absolute_error(y_test_col, pred_score_test)
                    r2_test = r2_score(y_test_col, pred_score_test)
                    rho, p = spearmanr(y_test_col, pred_score_test)

                    # If one selects top 20 predictions, what % of them are in the top 20 of true labels
                    k = 20
                    top_indices_true = np.argpartition(y_test_col, -k)[-k:]
                    top_indices_pred = np.argpartition(pred_score_test, -k)[-k:]
                    overlap = len(set(top_indices_pred) & set(top_indices_true))
                    top20 = overlap / k * 100
                    
                    label_results.append({
                        "Model": self.current_name,
                        "Best Params": best_params,
                        "CV MSE": mean_mse,
                        "Test MSE": mse_test,
                        "Test RMSE": rmse_test,
                        "Test MAE": mae_test,
                        "Test R2": r2_test,
                        "Test Spearman Correlation": rho,
                        "Top 20 Accuracy": top20
                    })
                else:
                    decoded_true = self.label_enc[self.target_names[label_idx]].inverse_transform(y_test_col)
                    decoded_preds = self.label_enc[self.target_names[label_idx]].inverse_transform(pred_score_test)
                    predictions_per_model[self.current_name] = [
                        {"sequence": seq, "prediction": pred, "true_value": y_true}
                        for seq, pred, y_true in zip(self.test_sequences, decoded_preds, decoded_true)
                    ]

                    acc = accuracy_score(y_test_col, pred_score_test)
                    bacc = balanced_accuracy_score(y_test_col, pred_score_test)
                    labels = self.label_enc[self.target_names[label_idx]].transform(self.label_enc[self.target_names[label_idx]].classes_)
                    f1 = f1_score(y_test_col, pred_score_test, average="weighted", labels=labels)
                    cm = confusion_matrix(y_test_col, pred_score_test, labels=labels)
                    mcc = matthews_corrcoef(y_test_col, pred_score_test)
                    roc_auc = self.compute_roc_auc(best_model, X_test, y_test_col)

                    label_results.append({
                        "Model": self.current_name,
                        "Best Params": best_params,
                        "CV Loss": mean_mse,
                        "Test Accuracy": acc,
                        "Test Balanced Accuracy": bacc,
                        "Test F1 (weighted)": f1,
                        "Test MCC": mcc,
                        "Test ROC AUC": roc_auc,
                        "Confusion Matrix": cm
                    })

                del study
                gc.collect()

            if not self.classification:
                results_df = pd.DataFrame(label_results).sort_values("CV MSE", ascending=True)
            else:
                results_df = pd.DataFrame(label_results).sort_values("CV Loss", ascending=True)
            results_df_per_label[self.target_names[label_idx]] = results_df
            all_predictions_per_label[self.target_names[label_idx]] = predictions_per_model

        return results_df_per_label, all_predictions_per_label
    

if __name__ == "__main__":
    pass