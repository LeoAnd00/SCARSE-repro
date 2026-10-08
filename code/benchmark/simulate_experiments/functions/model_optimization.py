"""
Sequence representations and downstream regression model optimization.

This module provides :class:`ModelOptimization`, which turns peptide/protein
sequences into fixed-length feature vectors and optimizes a downstream
regression model on top of them.

Representations
---------------
Exactly one representation is used per run, selected by ``representation``:

``esm``
    Mean-pooled last-hidden-state embeddings from a pretrained ESM2 model.
``physchem``
    Physicochemical baseline: modlAMP global descriptors, amino acid
    composition, and HeliQuest-inspired features (hydrophobic moment,
    discrimination factor, residue-class composition).
``morgan``
    Morgan fingerprint baseline: the sequence is built as a peptide molecule
    with RDKit and hashed into a binary circular-fingerprint bit vector.
``ngram``
    N-gram baseline: character n-grams over the sequence. Two schemes, set
    with ``ngram_vectorizer``:

    ``tfidf`` (default)
        A learned vocabulary with IDF weighting. The vocabulary and IDF
        weights are fit on the *training* sequences only and applied
        unchanged to the test sequences, so no test information leaks into
        the representation. N-grams that appear only in the test set have no
        column and are dropped.
    ``hashing``
        Every n-gram is hashed into one of ``ngram_features`` buckets. There
        is no vocabulary, so the representation is defined for *every*
        sequence and nothing is ever dropped; the cost is hash collisions
        between distinct n-grams. Stateless, so it cannot leak.

    Note that under either scheme an n-gram never seen during training
    carries no learnable signal: the downstream model has no fitted
    coefficient for it. Hashing changes what happens to such an n-gram from
    "dropped" to "added to a bucket shared with training n-grams", which
    keeps the feature space total and identical across seeds and datasets.

The three baselines (``physchem``, ``morgan``, ``ngram``) need neither a GPU
nor the torch/transformers stack: those imports are deferred until an ESM
model is actually requested.

This module is regression-only.
"""

import gc
import math
import random
import warnings

import numpy as np
import pandas as pd
import optuna
from Levenshtein import distance as levenshtein_distance
from modlamp.descriptors import GlobalDescriptor
from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
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


#: Representations that can be benchmarked. "esm" is the foundation model,
#: the rest are the baselines.
REPRESENTATIONS = ("esm", "physchem", "morgan", "ngram")

#: Baselines, i.e. every representation that is not a foundation model.
BASELINE_REPRESENTATIONS = ("physchem", "morgan", "ngram")

#: Ways of turning character n-grams into a fixed-width feature vector.
NGRAM_VECTORIZERS = ("tfidf", "hashing")

#: The ESM2 checkpoints up to 650M parameters, smallest first, with the
#: embedding width each one produces. Used by the foundation-model benchmark
#: so the scripts and the analysis notebooks agree on one list.
#: The 3B and 15B checkpoints are deliberately excluded.
ESM2_VARIANTS = {
    "facebook/esm2_t6_8M_UR50D":    {"params": "8M",   "layers": 6,  "embedding_dim": 320},
    "facebook/esm2_t12_35M_UR50D":  {"params": "35M",  "layers": 12, "embedding_dim": 480},
    "facebook/esm2_t30_150M_UR50D": {"params": "150M", "layers": 30, "embedding_dim": 640},
    "facebook/esm2_t33_650M_UR50D": {"params": "650M", "layers": 33, "embedding_dim": 1280},
}


class ModelOptimization:
    """Build sequence representations and optimize a regression model on them.

    Parameters
    ----------
    data_path : str
        CSV/TSV file with a sequence column and one or more score columns.
    random_seed : int
        Seed for splits, Optuna sampling and model initialization.
    initial_train_size : int
        Number of training sequences to draw.
    initial_test_size : int
        Number of test sequences to draw.
    emb_batch_size : int
        Batch size used when embedding with an ESM model.
    representation : str
        One of :data:`REPRESENTATIONS`.
    model_name : str
        HuggingFace identifier of the ESM2 model. Ignored unless
        ``representation="esm"``.
    morgan_radius : int
        Morgan fingerprint radius (``morgan`` only).
    morgan_bits : int
        Morgan fingerprint length in bits (``morgan`` only).
    ngram_range : tuple[int, int]
        Minimum and maximum character n-gram length (``ngram`` only).
    ngram_vectorizer : str
        ``"tfidf"`` for a train-fit vocabulary, ``"hashing"`` for a
        vocabulary-free hashed representation that covers every n-gram
        (``ngram`` only).
    ngram_features : int
        Number of hash buckets when ``ngram_vectorizer="hashing"``.
    """

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
        if ngram_vectorizer not in NGRAM_VECTORIZERS:
            raise ValueError(
                f"ngram_vectorizer must be one of {NGRAM_VECTORIZERS}, got {ngram_vectorizer!r}"
            )

        # Parameters
        self.data_path = data_path
        self.random_seed = random_seed
        self.initial_train_size = initial_train_size
        self.initial_test_size = initial_test_size
        self.emb_batch_size = emb_batch_size
        self.representation = representation
        self.model_name = model_name
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.ngram_range = tuple(ngram_range)
        self.ngram_vectorizer_kind = ngram_vectorizer
        self.ngram_features = ngram_features

        self.model = None
        self.tokenizer = None
        self.device = None
        self._morgan_generator = None
        self.ngram_vectorizer = None

        self.df = None
        self.seq_to_score = {}
        self.training_sequences = []
        self.test_sequences = []
        self.levenshtein_split = False

        # Seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prep_data(self, seq_col="sequence", score_col=["score"]):
        """Load the dataset and build the sequence -> score lookup."""

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

        # Keep only sequence + score columns
        df = df[["sequence"] + score_col].copy()

        self.df = df

        # Create a mapping from sequence -> score, or list of scores if multiple columns
        if len(score_col) == 1:
            self.seq_to_score = dict(zip(df["sequence"], df[score_col[0]]))
        else:
            self.seq_to_score = dict(zip(df["sequence"], df[score_col].values.tolist()))

        print("Finished preparing data!")

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def load_model(self):
        """Load the ESM2 model. Only needed when ``representation="esm"``."""

        # torch/transformers are imported here so the baselines can run in a
        # plain CPU environment without the deep learning stack installed.
        import torch
        from transformers import AutoTokenizer, EsmModel

        torch.manual_seed(self.random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading base pretrained ESM model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = EsmModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        print("ESM model loaded. Using device:", self.device)

    def compute_embeddings(self, sequences, batch_size=None, fit=False):
        """Turn ``sequences`` into a dense 2D feature matrix.

        Parameters
        ----------
        sequences : list[tuple[str, str]]
            ``(label, sequence)`` pairs. Labels are ignored.
        batch_size : int, optional
            ESM batch size. Defaults to ``emb_batch_size``.
        fit : bool, default=False
            Whether this call may fit representation state on the given
            sequences. Only the ``ngram`` representation has such state, and
            it must be fit on the training sequences and only on those.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(sequences), n_features)``.
        """

        seqs = [seq for _, seq in sequences]

        if self.representation == "esm":
            return self._embed_esm(seqs, batch_size=batch_size)
        if self.representation == "physchem":
            return self._embed_physchem(seqs)
        if self.representation == "morgan":
            return self._embed_morgan(seqs)
        if self.representation == "ngram":
            return self._embed_ngram(seqs, fit=fit)

        raise ValueError(f"Unknown representation: {self.representation!r}")

    def _embed_esm(self, seqs, batch_size=None):
        """Mean-pooled last hidden state of a pretrained ESM2 model."""

        import torch

        if self.model is None:
            raise RuntimeError("ESM model not loaded. Call load_model() first.")

        if batch_size is None:
            batch_size = self.emb_batch_size

        embeddings = []

        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]

            encoded = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states

            last_layer = hidden_states[-1:]
            stacked = torch.stack(last_layer, dim=0)
            mean_layers = stacked.mean(dim=0)

            for j in range(len(batch)):
                mask = attention_mask[j].bool().to(mean_layers.device)
                seq_emb = mean_layers[j, mask].mean(dim=0)
                embeddings.append(seq_emb.cpu().numpy())

        return np.vstack(embeddings)

    def _embed_physchem(self, seqs):
        """Physicochemical baseline (modlAMP + composition + HeliQuest-like)."""

        feats = [self.compute_generalizable_features(seq).flatten() for seq in seqs]
        return np.asarray(pd.DataFrame(feats), dtype=float)

    def _get_morgan_generator(self):
        from rdkit.Chem import rdFingerprintGenerator

        if self._morgan_generator is None:
            self._morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.morgan_radius,
                fpSize=self.morgan_bits
            )
        return self._morgan_generator

    def _embed_morgan(self, seqs):
        """Morgan fingerprint baseline.

        The sequence is built as a linear peptide molecule and hashed into a
        binary circular fingerprint. Sequences RDKit cannot parse (e.g.
        non-standard residues) fall back to an all-zero vector and are
        reported, rather than aborting the run.
        """

        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")

        generator = self._get_morgan_generator()

        fingerprints = []
        unparsed = []

        for seq in seqs:
            mol = Chem.MolFromSequence(seq)
            if mol is None:
                unparsed.append(seq)
                fingerprints.append(np.zeros(self.morgan_bits, dtype=np.uint8))
                continue
            fingerprints.append(generator.GetFingerprintAsNumPy(mol))

        if unparsed:
            print(
                f"[morgan] WARNING: RDKit could not build a molecule for "
                f"{len(unparsed)}/{len(seqs)} sequences; using zero vectors. "
                f"First example: {unparsed[0]}"
            )

        return np.asarray(fingerprints, dtype=float)

    def _embed_ngram(self, seqs, fit=False):
        """Character n-gram baseline.

        With ``ngram_vectorizer="tfidf"`` the vectorizer is fit on the
        training sequences only; test sequences are transformed with that
        fixed vocabulary, so n-grams seen only in the test set are ignored
        instead of leaking into the feature space.

        With ``ngram_vectorizer="hashing"`` there is no vocabulary at all:
        every n-gram of every sequence is hashed into a fixed number of
        buckets, so the representation is well defined for any sequence and
        no n-gram is ever dropped. Being stateless, it cannot leak, and
        ``fit`` is ignored.
        """

        if self.ngram_vectorizer_kind == "hashing":
            if self.ngram_vectorizer is None:
                self.ngram_vectorizer = HashingVectorizer(
                    analyzer="char",
                    ngram_range=self.ngram_range,
                    n_features=self.ngram_features,
                    alternate_sign=False,
                    norm="l2",
                    lowercase=False
                )
                print(
                    f"[ngram] Hashing n-grams (range {self.ngram_range}) into "
                    f"{self.ngram_features} buckets; every n-gram is covered."
                )
            matrix = self.ngram_vectorizer.transform(seqs)
            return np.asarray(matrix.todense(), dtype=float)

        if fit or self.ngram_vectorizer is None:
            if not fit:
                raise RuntimeError(
                    "N-gram vectorizer has not been fit yet. Compute the training "
                    "representation with fit=True before the test representation."
                )
            self.ngram_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.ngram_range,
                lowercase=False
            )
            matrix = self.ngram_vectorizer.fit_transform(seqs)
            print(
                f"[ngram] Fit on {len(seqs)} training sequences: "
                f"{len(self.ngram_vectorizer.vocabulary_)} n-grams "
                f"(range {self.ngram_range})"
            )
        else:
            matrix = self.ngram_vectorizer.transform(seqs)
            covered = np.asarray(matrix.sum(axis=1)).ravel() > 0
            if not covered.all():
                print(
                    f"[ngram] WARNING: {int((~covered).sum())}/{len(seqs)} sequences "
                    "share no n-gram with the training vocabulary and are all-zero "
                    "vectors. Consider ngram_vectorizer='hashing' or a smaller "
                    "ngram_range."
                )

        return np.asarray(matrix.todense(), dtype=float)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
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

    def initialize_test_set(self, df_path=None, seq_col="sequence", score_col="score",
                            levenshtein_split=False):

        if df_path:
            df = pd.read_csv(df_path, sep=None, engine='python')
            required_cols = {seq_col, score_col}
            if not required_cols.issubset(df.columns):
                raise ValueError(
                    f"Input file must contain columns: {required_cols}. Found: {df.columns.tolist()}"
                )

            df["sequence"] = df[seq_col].astype(str)
            df["score"] = df[score_col].astype(float)
            df = df[["sequence", "score"]].copy()

            self.seq_to_score = dict(zip(df["sequence"], df["score"]))
            self.test_sequences = df["sequence"].tolist()
            return

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
            # Calc levenshtein distance per sequence length group and select most similar sequences per group
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

    def initialize_training_set(self):

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
                raise ValueError(
                    f"Number of sequences for training was not achieved {len(rest_training_df)}"
                )
        else:
            rest_training_df = self.df_remaining.sample(
                n=remaining_size, random_state=self.random_seed
            ).reset_index(drop=True)

        self.training_sequences = rest_training_df["sequence"].tolist()

    # ------------------------------------------------------------------
    # Physicochemical features
    # ------------------------------------------------------------------
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

        combined_feats = np.concatenate([
            global_feats.flatten(),
            np.array(aa_counts),
            np.array(additional_features)
        ]).reshape(1, -1)

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
            Returns per-class residue fractions

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

            return np.array([
                n_tot_pol / tot,    # polar
                n_tot_apol / tot,   # apolar
                n_charged / tot,    # charged
                n_aromatic / tot,   # aromatic
            ], dtype=float)

        # HeliQuest inspired features
        z = calculate_charge(seq)
        seq_h = assign_hydrophobicity(seq)
        av_uH = calculate_moment(seq_h)
        d = calculate_discrimination(av_uH, z)
        aa_type_comp = calculate_composition(seq)

        additional_features = np.concatenate([
            np.array([av_uH], dtype=float),
            np.array([d], dtype=float),
            aa_type_comp
        ])
        return additional_features

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def optimize_all_models(self, folds=10, random_seed=42, n_trials=100, optuna_print=True):
        """Optimize the downstream regression model for every target column."""

        # Suppress convergence and feature name warnings
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        ## Seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load the foundation model only when it is the representation being used.
        if self.representation == "esm":
            if self.model_name.split("/")[0].lower() != "facebook":
                raise ValueError("Foundation model needs to be an ESM2 model from Huggingface...")
            self.load_model()

        # Training representation. fit=True lets the n-gram vectorizer learn its
        # vocabulary here, on training sequences only.
        label_seq = [(f"train_{i}", seq) for i, seq in enumerate(self.training_sequences)]
        X_train_temp = self.compute_embeddings(sequences=label_seq, fit=True)

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

        # Data to be used for evaluating on test data. fit=False keeps the
        # n-gram vocabulary fixed to what the training sequences produced.
        X_train = X_train_temp
        label_seq_test = [(f"test_{i}", seq) for i, seq in enumerate(self.test_sequences)]
        X_test = self.compute_embeddings(sequences=label_seq_test, fit=False)

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

                            y_train_fold = np.vstack(train_df["score"]).ravel()
                            y_val_fold = np.vstack(val_df["score"]).ravel()

                            model_fold.fit(X_train_fold, y_train_fold)
                            y_pred = model_fold.predict(X_val_fold)

                            mse = mean_squared_error(y_val_fold, y_pred)
                            fold_scores.append(mse)

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
                predictions_per_model[self.current_name] = [
                    {"sequence": seq, "prediction": pred, "true_value": y_true}
                    for seq, pred, y_true in zip(self.test_sequences, pred_score_test, y_test_col)
                ]

                mse_test = mean_squared_error(y_test_col, pred_score_test)
                rmse_test = np.sqrt(mse_test)
                mae_test = mean_absolute_error(y_test_col, pred_score_test)
                r2_test = r2_score(y_test_col, pred_score_test)
                rho, p = spearmanr(y_test_col, pred_score_test)

                # If one selects top k predictions, what % of them are in the top k of true labels
                k = min(20, len(y_test_col))
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

                del study
                gc.collect()

            results_df = pd.DataFrame(label_results).sort_values("CV MSE", ascending=True)
            results_df_per_label[self.target_names[label_idx]] = results_df
            all_predictions_per_label[self.target_names[label_idx]] = predictions_per_model

        return results_df_per_label, all_predictions_per_label


if __name__ == "__main__":
    pass