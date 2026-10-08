"""
Sequence representations shared by the benchmark, the active-learning workflow
simulation and the selection analysis.

Exactly one representation is used per run:

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
    Character n-grams over the sequence, TF-IDF weighted. The vocabulary and
    IDF weights are fit on *training* sequences only and applied unchanged to
    anything else, so no held-out information leaks into the features.

The three baselines need neither a GPU nor the torch/transformers stack: those
imports are deferred until an ESM model is actually requested.

Keep one copy of this file next to each project's ``model_optimization.py`` so
the three pipelines compute identical features.
"""

import math

import numpy as np
import pandas as pd
from modlamp.descriptors import GlobalDescriptor
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

#: Representations that can be used. "esm" is the foundation model, the rest
#: are the baselines.
REPRESENTATIONS = ("esm", "physchem", "morgan", "ngram")

#: Baselines, i.e. every representation that is not a foundation model.
BASELINE_REPRESENTATIONS = ("physchem", "morgan", "ngram")

#: Ways of turning character n-grams into a fixed-width feature vector.
NGRAM_VECTORIZERS = ("tfidf", "hashing")

#: The ESM2 checkpoints up to 650M parameters, smallest first.
ESM2_VARIANTS = {
    "facebook/esm2_t6_8M_UR50D":    {"params": "8M",   "layers": 6,  "embedding_dim": 320},
    "facebook/esm2_t12_35M_UR50D":  {"params": "35M",  "layers": 12, "embedding_dim": 480},
    "facebook/esm2_t30_150M_UR50D": {"params": "150M", "layers": 30, "embedding_dim": 640},
    "facebook/esm2_t33_650M_UR50D": {"params": "650M", "layers": 33, "embedding_dim": 1280},
}


class SequenceRepresenter:
    """Turn sequences into a fixed-width feature matrix.

    Parameters
    ----------
    representation : str
        One of :data:`REPRESENTATIONS`.
    model_name : str
        HuggingFace identifier of the ESM2 model. Ignored unless
        ``representation="esm"``.
    emb_batch_size : int
        Batch size used when embedding with an ESM model.
    random_seed : int
        Seed applied to torch when an ESM model is loaded.
    morgan_radius, morgan_bits : int
        Morgan fingerprint settings (``morgan`` only).
    ngram_range : tuple[int, int]
        Character n-gram range (``ngram`` only).
    ngram_vectorizer : str
        ``"tfidf"`` for a train-fit vocabulary, ``"hashing"`` for a
        vocabulary-free hashed representation covering every n-gram.
    ngram_features : int
        Number of hash buckets when ``ngram_vectorizer="hashing"``.
    """

    def __init__(
        self,
        representation="esm",
        model_name="facebook/esm2_t33_650M_UR50D",
        emb_batch_size=64,
        random_seed=42,
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

        self.representation = representation
        self.model_name = model_name
        self.emb_batch_size = emb_batch_size
        self.random_seed = random_seed
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

    # ------------------------------------------------------------------
    @property
    def needs_model(self):
        """Whether this representation requires loading a foundation model."""
        return self.representation == "esm"

    def load_model(self):
        """Load the ESM2 model. A no-op for the baselines."""
        if not self.needs_model:
            return

        # torch/transformers are imported here so the baselines can run in a
        # plain CPU environment without the deep learning stack installed.
        import torch
        from transformers import AutoTokenizer, EsmModel

        if self.model_name.split("/")[0].lower() != "facebook":
            raise ValueError("Foundation model needs to be an ESM2 model from Huggingface...")

        torch.manual_seed(self.random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading base pretrained ESM model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = EsmModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        print("ESM model loaded. Using device:", self.device)

    # ------------------------------------------------------------------
    def transform(self, sequences, fit=False, batch_size=None):
        """Represent ``sequences`` as a dense 2D array.

        Parameters
        ----------
        sequences : list[str] | list[tuple[str, str]]
            Sequences, or ``(label, sequence)`` pairs as used elsewhere in the
            codebase. Labels are ignored.
        fit : bool
            Whether this call may fit representation state on these sequences.
            Only the TF-IDF n-gram representation has such state, and it must
            be fit on training sequences and only on those. In the
            active-learning loop the training set grows each round, so this is
            passed True once per round for the current training set.
        batch_size : int, optional
            ESM batch size. Defaults to ``emb_batch_size``.
        """
        seqs = [s[1] if isinstance(s, (tuple, list)) else s for s in sequences]

        if self.representation == "esm":
            return self._embed_esm(seqs, batch_size=batch_size)
        if self.representation == "physchem":
            return self._embed_physchem(seqs)
        if self.representation == "morgan":
            return self._embed_morgan(seqs)
        if self.representation == "ngram":
            return self._embed_ngram(seqs, fit=fit)

        raise ValueError(f"Unknown representation: {self.representation!r}")

    # ------------------------------------------------------------------
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

        Sequences RDKit cannot parse (e.g. non-standard residues) fall back to
        an all-zero vector and are reported, rather than aborting the run.
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
        """Character n-gram baseline."""
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
            matrix = self.ngram_vectorizer.transform(seqs)
            return np.asarray(matrix.todense(), dtype=float)

        if fit:
            self.ngram_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.ngram_range,
                lowercase=False
            )
            matrix = self.ngram_vectorizer.fit_transform(seqs)
        else:
            if self.ngram_vectorizer is None:
                raise RuntimeError(
                    "N-gram vectorizer has not been fit yet. Represent the training "
                    "sequences with fit=True before anything else."
                )
            matrix = self.ngram_vectorizer.transform(seqs)

        return np.asarray(matrix.todense(), dtype=float)

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
        """Amino acid type composition, hydrophobic moment and discrimination
        factor, similar to HeliQuest:

        Gautier R., Douguet D., Antonny B. and Drin G. HELIQUEST: a web server
        to screen sequences with specific alpha-helical properties.
        Bioinformatics. 2008 Sep 15;24(18):2101-2.
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

        return np.concatenate([
            np.array([av_uH], dtype=float),
            np.array([d], dtype=float),
            aa_type_comp
        ])


if __name__ == "__main__":
    pass
