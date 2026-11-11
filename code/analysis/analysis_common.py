"""Shared helpers for the reviewer-response analysis notebooks.

These notebooks live in ``code/analysis`` and read the same processed data and
the same simulation outputs the rest of the pipeline uses. Nothing here fits or
runs a model; it only summarises data and already-computed results, so it needs
no GPU and only the scientific Python stack.

Paths are resolved relative to the repository root, found by walking up from this
file, so the notebooks work regardless of the directory Jupyter is started in.
"""

import os
import glob

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Repository layout
# ----------------------------------------------------------------------
def repo_root():
    """Absolute path to the repository root (the folder containing ``code``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "data")):
            return d
        d = os.path.dirname(d)
    # Fallback: two levels up from code/analysis.
    return os.path.dirname(os.path.dirname(here))


ROOT = repo_root()
DATA_DIR = os.path.join(ROOT, "data")
BENCH_OUT = os.path.join(ROOT, "code", "benchmark", "simulate_experiments", "simulation_output")
AL_RESULTS = os.path.join(ROOT, "code", "workflow_simulation", "simulation_results")
FIG_DIR = os.path.join(ROOT, "code", "analysis", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Dataset catalogue
# ----------------------------------------------------------------------
#: Peptide datasets (AMP potency + HemoPI2), name -> processed CSV (relative to DATA_DIR).
PEPTIDE_FILES = {
    "Ecoli_AMPs":       "ecoli_amps/processed/EC_all_40.csv",
    "Saureus_AMPs":     "saureus_amps/processed/SA_all_40.csv",
    "Paeruginosa_AMPs": "paeruginosa_amps/processed/PA_all_40.csv",
    "HemoPI2":          "hemopi2/processed/hemopi2_all.csv",
}

#: Substitution datasets, name -> processed CSV stem (relative to DATA_DIR).
SUBSTITUTION_FILES = {
    "A0A247D711_LISMN": "proteingym_dms/processed/substitutions/A0A247D711_LISMN_Stadelmann_2021.csv",
    "DN7A_SACS2":       "proteingym_dms/processed/substitutions/DN7A_SACS2_Tsuboyama_2023_1JIC.csv",
    "ENVZ_ECOLI":       "proteingym_dms/processed/substitutions/ENVZ_ECOLI_Ghose_2023.csv",
    "FKBP3_HUMAN":      "proteingym_dms/processed/substitutions/FKBP3_HUMAN_Tsuboyama_2023_2KFV.csv",
    "MAFG_MOUSE_sub":   "proteingym_dms/processed/substitutions/MAFG_MOUSE_Tsuboyama_2023_1K1V.csv",
    "POLG_PESV_sub":    "proteingym_dms/processed/substitutions/POLG_PESV_Tsuboyama_2023_2MXD.csv",
    "SBI_STAAM":        "proteingym_dms/processed/substitutions/SBI_STAAM_Tsuboyama_2023_2JVG.csv",
    "SDA_BACSU":        "proteingym_dms/processed/substitutions/SDA_BACSU_Tsuboyama_2023_1PV0.csv",
    "SOX30_HUMAN":      "proteingym_dms/processed/substitutions/SOX30_HUMAN_Tsuboyama_2023_7JJK.csv",
    "YNZC_BACSU_sub":   "proteingym_dms/processed/substitutions/YNZC_BACSU_Tsuboyama_2023_2JVD.csv",
}

#: AMP potency datasets, where a HIGHER score (pMIC) is more desirable.
AMP_DATASETS = ["Ecoli_AMPs", "Saureus_AMPs", "Paeruginosa_AMPs"]

#: Datasets optimised as a MINIMISE objective (lower score is better).
MINIMIZE_DATASETS = ["HemoPI2"]

#: The 20 standard amino acids.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Kyte-Doolittle hydropathy scale (mean over residues = GRAVY).
KD_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

#: Net charge contribution at ~pH 7 (simple Lys/Arg = +1, Asp/Glu = -1, His = +0.1).
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def direction_sign(dataset):
    """+1 for maximise datasets, -1 for minimise (HemoPI2), so that after
    multiplying the score higher always means more desirable. Mirrors the
    simulation's prep_data negation."""
    return -1.0 if dataset in MINIMIZE_DATASETS else 1.0


def load_dataset(dataset, files=None, seq_col="sequence", score_col="score"):
    """Load one processed dataset as a DataFrame with 'sequence' and 'score'.

    Returns None (with a message) if the file is missing, so a notebook can skip
    datasets that have not been downloaded rather than aborting.
    """
    if files is None:
        files = {**PEPTIDE_FILES, **SUBSTITUTION_FILES}
    rel = files.get(dataset)
    if rel is None:
        print(f"[skip] no file mapping for {dataset}")
        return None
    path = os.path.join(DATA_DIR, rel)
    if not os.path.exists(path):
        print(f"[skip] missing data file for {dataset}: {path}")
        return None
    df = pd.read_csv(path, sep=None, engine="python")
    if seq_col not in df.columns or score_col not in df.columns:
        print(f"[skip] {dataset}: need columns {seq_col!r},{score_col!r}; found {list(df.columns)}")
        return None
    out = pd.DataFrame({
        "sequence": df[seq_col].astype(str),
        "score": pd.to_numeric(df[score_col], errors="coerce"),
    }).dropna()
    return out


# ----------------------------------------------------------------------
# Peptide physicochemical properties (lightweight, dependency-free)
# ----------------------------------------------------------------------
def seq_length(seq):
    return len(str(seq))


def net_charge(seq):
    return float(sum(CHARGE.get(aa, 0.0) for aa in str(seq).upper()))


def gravy(seq):
    s = str(seq).upper()
    vals = [KD_HYDRO.get(aa) for aa in s if aa in KD_HYDRO]
    return float(np.mean(vals)) if vals else np.nan


def aromaticity(seq):
    s = str(seq).upper()
    if not s:
        return np.nan
    return float(sum(s.count(aa) for aa in "FWY") / len(s))


def property_table(df):
    """Attach length, net charge, GRAVY hydrophobicity and aromaticity columns."""
    out = df.copy()
    out["length"] = out["sequence"].map(seq_length)
    out["net_charge"] = out["sequence"].map(net_charge)
    out["gravy"] = out["sequence"].map(gravy)
    out["aromaticity"] = out["sequence"].map(aromaticity)
    return out


# ----------------------------------------------------------------------
# Benchmark results (Test R2 per representation)
# ----------------------------------------------------------------------
def load_benchmark_summary(group):
    """Load one benchmark family's simulation_summary.csv.

    group is 'peptides', 'substitutions' or 'indels'. Returns None if missing.
    """
    path = os.path.join(BENCH_OUT, group, "simulation_summary.csv")
    if not os.path.exists(path):
        print(f"[skip] missing benchmark summary: {path}")
        return None
    return pd.read_csv(path)


def load_al_results():
    """Load the active-learning df_all.csv (long format). None if missing."""
    path = os.path.join(AL_RESULTS, "df_all.csv")
    if not os.path.exists(path):
        print(f"[skip] missing AL results: {path}")
        return None
    return pd.read_csv(path)


def savefig(fig, name):
    """Save a figure as PNG + PDF into code/analysis/figures."""
    png = os.path.join(FIG_DIR, name + ".png")
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("saved", png)
    print("saved", pdf)


# ----------------------------------------------------------------------
# Display formatting
# ----------------------------------------------------------------------
def pretty_dataset(name):
    """Publication display name for a dataset.

    The three antimicrobial-peptide species read as italic Latin binomials
    (E. coli, S. aureus, P. aeruginosa); every other name only has its
    underscores turned to spaces. Accepts both the friendly ids used in this
    analysis layer (e.g. ``Ecoli_AMPs``) and the CSV-basename ids stored in the
    benchmark summaries (e.g. ``EC_all_40``, ``hemopi2_all``). Idempotent, so it
    is safe to apply whether the input is a raw id or already a display string.
    """
    s = str(name)
    _AMP = {
        "Ecoli_AMPs":       r"$\it{E.\ coli}$ AMPs",
        "Saureus_AMPs":     r"$\it{S.\ aureus}$ AMPs",
        "Paeruginosa_AMPs": r"$\it{P.\ aeruginosa}$ AMPs",
        # benchmark CSV basenames
        "EC_all_40":        r"$\it{E.\ coli}$ AMPs",
        "SA_all_40":        r"$\it{S.\ aureus}$ AMPs",
        "PA_all_40":        r"$\it{P.\ aeruginosa}$ AMPs",
        "hemopi2_all":      "HemoPI2",
    }
    if s in _AMP:
        return _AMP[s]
    disp = s.replace("_", " ")
    _SPECIES = {
        "E. coli AMPs":       r"$\it{E.\ coli}$ AMPs",
        "E.coli AMPs":        r"$\it{E.\ coli}$ AMPs",
        "S. aureus AMPs":     r"$\it{S.\ aureus}$ AMPs",
        "S.aureus AMPs":      r"$\it{S.\ aureus}$ AMPs",
        "P. aeruginosa AMPs": r"$\it{P.\ aeruginosa}$ AMPs",
        "P.aeruginosa AMPs":  r"$\it{P.\ aeruginosa}$ AMPs",
    }
    return _SPECIES.get(disp, disp)


def pretty_family(fam):
    """Display label for a dataset family/type used in legends and titles."""
    return {"AMP": "AMP", "HemoPI2": "HemoPI2",
            "Substitution": "Substitution", "Indel": "Indel"}.get(str(fam), str(fam))
