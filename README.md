# SCARSE: Small-sample Cross-validation-Anchored Regression Solution for low resource peptide Engineering - Reproducability code

<p align="center">
  <img src="figures/model_architecture.png" alt="Workflow Diagram" width="500"/>
</p>

## Abstract

Reliable estimation of downstream performance in low-data peptide machine
learning is critical for guiding early-stage AI-driven peptide engineering, yet
it is often unclear how to assess whether a model will be effective in iterative
discovery settings. Here we ask whether a cross-validation score computed on a
handful of labelled peptides can serve as a simple and robust proxy for how well
an active-learning workflow built on that model will eventually perform. To
support this we introduce SCARSE, a machine learning framework combining ESM-2
protein language model embeddings with Gaussian process regression, designed for
low-resource peptide property prediction (20–500 training samples). We benchmark
SCARSE across 24 peptide and small-protein datasets covering substitution
variants, indel variants, antimicrobial peptide potency and haemolytic activity,
against three hand-engineered baselines — physicochemical descriptors, Morgan
fingerprints and character n-grams — and across four ESM-2 checkpoints from 8M
to 650M parameters, to establish both which representation and which foundation
model size a low-data peptide campaign should use. We then simulate active
learning workflows under two candidate selection strategies, purely greedy
top-k acquisition and a mixed strategy that deliberately spends part of each
batch on the peptides the model rates worst, and relate the end-point
performance of each workflow back to the cross-validation R² and Spearman ρ
measured early in the campaign. SCARSE is released as a pip package and is
available via HuggingFace Spaces to facilitate integration into peptide
engineering workflows.

---

## Contents

1. [Overview](#overview) — the four benchmarks and the order to run them
2. [Repository layout](#repository-layout)
3. [Setup](#setup) — environment, cluster, data
4. [Stage 0 — cache the ESM-2 checkpoints](#stage-0--cache-the-esm-2-checkpoints)
5. [Stage 1 — process the data](#stage-1--process-the-data)
6. [Sequence representations](#sequence-representations)
7. [Benchmark B — foundation models](#benchmark-b--which-esm-2-checkpoint-is-best)
8. [Benchmark A — representations](#benchmark-a--does-esm-2-beat-the-three-baselines)
9. [Benchmark C — active learning](#benchmark-c--active-learning-workflow-simulations)
10. [Benchmark D — CV score vs end-point performance](#benchmark-d--cv-score-vs-end-point-performance)
11. [Dependencies between stages](#dependencies-between-stages)
12. [Compute](#compute)
13. [Citation](#citation)

---

## Overview

Everything runs in five stages. Each depends only on the ones before it, so they
can be run days apart:

```
0. Cache the ESM-2 models  -> hf_cache/                       (once, first)
1. Process the data        -> data/*/processed/*.csv
2. Benchmarks              -> simulation_output/...           (SLURM)
3. Aggregate               -> simulation_summary[_all].csv
4. Statistics & figures
```

Stage 2 holds four benchmarks. They all build on the processed data from stage 1,
and **B feeds A**, so run them in this order:

| # | Benchmark | Question | Where |
| --- | --- | --- | --- |
| 1 | B. Foundation models | Which ESM-2 checkpoint up to 650M is best? | `code/benchmark/` |
| 2 | A. Representations | Does ESM-2 beat the three baselines? | `code/benchmark/` |
| — | C. Active learning | How good is the acquired set, round by round? | `code/workflow_simulation/` |
| — | D. Selection analysis | Does the CV score predict active-learning end-point performance? | `code/selection_analysis/` |

B comes first because A's `esm` arm *is* B's 650M arm — identical datasets,
seeds, training sizes and `optimize_models.py` call — so A reads those results
instead of repeating 1 380 tasks. C and D are independent of both, except that
D's stage 4 reads C's output.

The four sequence representations are the same everywhere: `esm`, `physchem`,
`morgan`, `ngram`. Every script takes the representation as its first argument.

### 24 datasets in three families

| Family | Datasets | Target | Split |
| --- | --- | --- | --- |
| Substitutions | 10 ProteinGym DMS assays | DMS score | Random |
| Indels | 10 ProteinGym DMS assays | DMS score | Random |
| Peptides | E.coli / S.aureus / P.aeruginosa AMPs, HemoPI2 | pMIC, −log10 HC50 | Levenshtein |

Every target is optimised toward higher values (more active), with one exception:
HemoPI2's score is −log10(HC50), where a lower value is less haemolytic and thus
more desirable, so HemoPI2 is optimised as a minimize objective in the
active-learning workflow (see Benchmark C).

---

## Repository layout

```
project-root/
├── README.md                                   this file
├── requirements.txt
├── env.sif                                     Apptainer image the jobs run in
│
├── data/
│   ├── proteingym_dms/raw/{DMS_ProteinGym_substitutions,DMS_ProteinGym_indels}/
│   ├── ecoli_amps/raw/{EC_X_train_40,EC_X_val_40,EC_X_test_40}.csv
│   ├── saureus_amps/raw/{SA_X_train_40,SA_X_val_40,SA_X_test_40}.csv
│   ├── paeruginosa_amps/raw/{PA_X_train_40,PA_X_val_40,PA_X_test_40}.csv
│   └── hemopi2/raw/{cross_val_dataset,independent_dataset}.csv
│
└── code/
    ├── setup/
    │   ├── cache_esm_models.py                    stage 0
    │   └── cache_esm_models.sh
    │
    ├── benchmark/
    │   ├── process_data/
    │   │   ├── process_data.ipynb                 stage 1
    │   │   └── visualize_processed_data.ipynb
    │   │
    │   ├── simulate_experiments/                  benchmarks A and B
    │   │   ├── optimize_models.py
    │   │   ├── aggregate_simulation_results.py
    │   │   ├── functions/
    │   │   │   ├── __init__.py
    │   │   │   └── model_optimization.py
    │   │   ├── scripts/
    │   │   │   ├── substitutions.sh                  A
    │   │   │   ├── indels.sh                         A
    │   │   │   ├── peptides.sh                       A
    │   │   │   ├── run_aggregate.sh                  A
    │   │   │   ├── foundations_substitutions.sh      B
    │   │   │   ├── foundations_indels.sh             B
    │   │   │   ├── foundations_peptides.sh           B
    │   │   │   └── run_aggregate_foundations.sh      B
    │   │   └── simulation_output/                  created by the runs
    │   │       ├── {substitutions,indels,peptides}/                A
    │   │       └── foundations/{substitutions,indels,peptides}/    B
    │   │
    │   ├── simulation_output/                     optional copy of the above
    │   │   ├── {substitutions,indels,peptides}/
    │   │   └── foundations/{substitutions,indels,peptides}/
    │   │
    │   ├── statistics/
    │   │   ├── model_comparison.py                vendored Polaris code
    │   │   ├── model_comparison_README.md
    │   │   ├── substitutions.ipynb                   A
    │   │   ├── indels.ipynb                          A
    │   │   ├── peptides.ipynb                        A
    │   │   ├── foundation_models.ipynb               B
    │   │   └── results/                           created by the notebooks
    │   │
    │   ├── visualizations_3.ipynb                    A
    │   └── visualizations_foundations.ipynb          B
    │
    ├── workflow_simulation/                       benchmark C
    │   ├── simulation.py
    │   ├── simulation.ipynb
    │   ├── seed_variance_analysis.ipynb           appendix seed-variance figure
    │   ├── rebuild_df_all.py                      rebuild df_all.csv from runs/
    │   ├── replace_hemopi2.py                     swap in the HemoPI2 minimize rerun
    │   ├── functions/
    │   │   ├── __init__.py
    │   │   ├── representations.py                 shared, identical copy
    │   │   └── model_optimization.py
    │   ├── scripts/
    │   │   ├── simulation_per_seed.sh             the main array (all datasets)
    │   │   ├── rerun_hemopi2_minimize.sh          HemoPI2 only, minimize
    │   │   ├── replace_hemopi2.sh                 merge the HemoPI2 rerun into runs/
    │   │   └── rebuild_df_all.sh                  runs/ -> df_all.csv
    │   └── simulation_results/                    created by the runs
    │       ├── df_all.csv                         combined file (legacy / rebuilt)
    │       └── runs/                              one CSV per (dataset, rep, strategy, seed)
    │
    └── selection_analysis/                        benchmark D
        ├── optimize_models.py
        ├── aggregate_simulation_results.py
        ├── selecting_workflow_eval.ipynb
        ├── functions/
        │   ├── __init__.py
        │   ├── representations.py                 shared, identical copy
        │   └── model_optimization.py
        ├── scripts/
        │   ├── benchmark_script_substitutions.sh
        │   ├── benchmark_script_peptides.sh
        │   └── run_aggregate.sh
        ├── simulation_output/                     created by the runs
        └── results/                               created by the notebook
```

Three things to note about the layout:

* **`functions/representations.py` is one file kept in two places** —
  `workflow_simulation/functions/` and `selection_analysis/functions/`. The
  copies must stay identical or the pipelines stop computing the same features.
  (`simulate_experiments` keeps its own copy of the same logic inside
  `model_optimization.py`.)
* **`functions/__init__.py` must exist** in each `functions/` folder, since the
  code imports `functions.model_optimization`.
* **Where the notebooks look for results.** The SLURM jobs write to
  `simulate_experiments/simulation_output/`. The statistics and visualization
  notebooks find that folder themselves — `find_results_root()` at the top of
  each takes the first of these that exists, and prints which one it used:

  ```
  simulate_experiments/simulation_output/     where the jobs write
  simulation_output/                          a copy under code/benchmark/
  simulation_outputs/                         same, plural spelling
  ```

  So no copying is needed, and results copied to either spelling under
  `code/benchmark/` still work — for example when the outputs are moved off the
  cluster without the rest of the tree. The jobs' own folder is checked first,
  so a stale copy can never shadow fresh results. If none exists, the error
  lists all three absolute paths it tried.

---

## Setup

### Environment

Build the environment from *requirements.txt* and wrap it in the Apptainer image
the jobs run in, `env.sif` at the project root. Beyond the usual scientific
stack (`pandas`, `numpy`, `scikit-learn`, `scipy`, `optuna`, `matplotlib`,
`seaborn`) the pipelines need:

| Package | Used by |
| --- | --- |
| `torch`, `transformers` | the `esm` representation |
| `modlamp` | the `physchem` baseline |
| `rdkit` | the `morgan` baseline |
| `Levenshtein` | similarity-based splits |
| `pingouin`, `statsmodels`, `scikit-posthocs` | the statistics notebooks |

Only `esm` needs torch and transformers; the three baselines import them
lazily, so baseline jobs run on a plain CPU node.

### Cluster

**The one thing to edit** is `PROJECT_ROOT`, near the top of every `.sh` file:

```bash
PROJECT_ROOT="/proj/berzelius-2026-62/users/${USER}/reproducibility_code"
```

Everything else — the container, the data directory, the HF cache, the output
directories — derives from it, so there are no other paths to change. Each
script `cd`s to its own module root, derived from `PROJECT_ROOT`, so relative
paths do not depend on where you submit from. Each also pins BLAS threads to
`$SLURM_CPUS_PER_TASK`; without that, numpy grabs every core on the node and the
array tasks fight each other.

**Submit each script from its own `scripts` folder, and create the log directory
first.** `#SBATCH -o logs/...` resolves relative to the directory you run
`sbatch` from, and SLURM opens those files *before* the script body runs, so no
`mkdir` inside a script can cover it — submit from a folder without `logs/` and
the job dies before printing anything:

```bash
mkdir -p code/benchmark/simulate_experiments/scripts/logs
mkdir -p code/workflow_simulation/scripts/logs
mkdir -p code/selection_analysis/scripts/logs
mkdir -p code/setup/logs
```

### Data

Create the following folder structure:

```
project-root/
├── data/
    ├── proteingym_dms/
    │    └── raw/
    │       ├── DMS_ProteinGym_indels/
    │       ├── DMS_ProteinGym_substitutions/
    │       ├── DMS_indels.csv
    │       └── DMS_substitutions.csv
    ├── ecoli_amps/
    │   └── raw/
    │       ├── EC_X_train_40.csv
    │       ├── EC_X_val_40.csv
    │       └── EC_X_test_40.csv
    ├── saureus_amps/
    │   └── raw/
    │       ├── SA_X_train_40.csv
    │       ├── SA_X_val_40.csv
    │       └── SA_X_test_40.csv
    ├── paeruginosa_amps/
    │   └── raw/
    │       ├── PA_X_train_40.csv
    │       ├── PA_X_val_40.csv
    │       └── PA_X_test_40.csv
    └── hemopi2/
        └── raw/
            ├── independent_dataset.csv
            └── cross_val_dataset.csv
```

Download each dataset into its `raw/` folder:

* Substitution and indel DMS assay datasets: https://proteingym.org/download
* E.coli AMPs: https://doi.org/10.1016/j.isci.2024.110718
* S.aureus AMPs: https://doi.org/10.1016/j.isci.2024.110718
* P.aeruginosa AMPs: https://doi.org/10.1016/j.isci.2024.110718
* HemoPI2: https://doi.org/10.1038/s42003-025-07615-w

---

## Stage 0 — cache the ESM-2 checkpoints

Every job that uses `--representation esm` loads an ESM-2 checkpoint. If
thousands of array tasks start at once with an empty cache they all try to
download the same files simultaneously: you get rate limited, and concurrent
writes into one cache directory can leave truncated files that only fail hours
into a run.

So download them once, up front:

```bash
cd code/setup
# edit PROJECT_ROOT at the top of cache_esm_models.sh, then:
mkdir -p logs && sbatch cache_esm_models.sh
# if the compute nodes have no outbound network, run it on a login node instead:
#   bash cache_esm_models.sh
```

This fetches all four checkpoints (~3.2 GB) and then **loads each one with the
network disabled**, which proves the run scripts will work offline rather than
only that some files landed on disk.

Every run script sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so once
this has succeeded no job will ever contact huggingface.co. A job that reports a
missing model is telling you the cache is incomplete — re-run this step rather
than removing the offline flags.

Useful variants:

```bash
python cache_esm_models.py --only 650M     # just the one the main benchmarks use
python cache_esm_models.py --verify_only   # check an existing cache
```

If you are only running the three baselines, you can skip this step entirely —
`physchem`, `morgan` and `ngram` never load a model.

---

## Stage 1 — process the data

```bash
cd code/benchmark/process_data
# run process_data.ipynb             -> writes data/*/processed/*.csv
# run visualize_processed_data.ipynb -> sanity-check figures
```

`process_data.ipynb` creates the `processed/` folders itself. Cleaned files use a
common schema: `sequence`, `score`, and — where applicable — `sequence_length`
and `classes`. Everything downstream reads only these files:

```
data/proteingym_dms/processed/substitutions/*.csv
data/proteingym_dms/processed/indels/*.csv
data/ecoli_amps/processed/EC_all_40.csv
data/saureus_amps/processed/SA_all_40.csv
data/paeruginosa_amps/processed/PA_all_40.csv
data/hemopi2/processed/hemopi2_all.csv
```

The datasets that ship as ready-made splits are fused back together into a
single file, with a `split` column recording where each row came from:

| Dataset | Fused file | Rows |
| --- | --- | --- |
| E.coli AMPs | `ecoli_amps/processed/EC_all_40.csv` | 3818 |
| S.aureus AMPs | `saureus_amps/processed/SA_all_40.csv` | 2644 |
| P.aeruginosa AMPs | `paeruginosa_amps/processed/PA_all_40.csv` | 2458 |
| HemoPI2 | `hemopi2/processed/hemopi2_all.csv` | 1926 |

### Sequence length filter

Every dataset is capped at `MAX_SEQUENCE_LENGTH = 40` residues, set in the
*E.coli AMPs* cell and repeated in the HemoPI2 cell. The filter runs before the
median-based class split, so classes reflect the sequences that are kept. All
four datasets are already within the cap, so it currently drops nothing — it is
there to keep any re-downloaded or extended raw data within range.

### Score conventions

Scores are stored so that a **higher value means a more potent peptide**:

* AMP datasets — `score` = pMIC = `-log10(MIC[uM])`, so a higher score is a lower
  inhibitory concentration. Set `SCORE_AS_PMIC = False` in the *E.coli AMPs* cell
  to keep the raw `log10(MIC)` orientation instead.
* HemoPI2 — `score` = `-log10(HC50[uM])`, so a higher score is a lower
  haemolytic concentration, i.e. a MORE haemolytic peptide. Labels mark
  HC50 ≤ 100 µM as haemolytic.

The stored orientation is only about how the data is written. Whether the
active-learning workflow selects high or low scores is a separate choice made at
run time with `--direction` (see Benchmark C). HemoPI2 is run with
`--direction minimize`, because for haemolysis a lower −log10(HC50), i.e. a less
haemolytic peptide, is the desirable outcome.

### Strategic similarity-based splits

The last section of *process_data.ipynb* builds an alternative split for each
dataset from its fused file. Within each sequence length a reference sequence is
drawn at random; the closest sequences become the test set and the furthest of
the remainder become the training set, so the length profile of the full dataset
is preserved while the test set is a tight similarity cluster and the training
set a diverse one.

Splits are plotted but not written to disk unless `SAVE_SPLITS = True`.

---

## Sequence representations

Every benchmark compares the same four representations, selected with
`--representation` and passed to each `.sh` script as its first argument.

| `--representation` | Features | Notes |
| --- | --- | --- |
| `esm` | 1280 | mean-pooled ESM-2 650M embeddings |
| `physchem` | 36 | modlAMP global descriptors + amino acid composition + HeliQuest-inspired features |
| `morgan` | 2048 | RDKit peptide molecule → circular fingerprint |
| `ngram` | varies | TF-IDF character n-grams, vocabulary fit on training data only |

The n-gram vocabulary is fit on the **training** sequences alone and applied
unchanged to everything else. In the active-learning loop the training set grows
each round, so it is refit per round. Fitting it on the full dataset would leak
held-out composition into the features and flatter the baseline.
`--ngram_vectorizer hashing` offers a vocabulary-free alternative that covers
every n-gram at the cost of hash collisions.

`<family>` in the commands below is one of `substitutions`, `indels` or
`peptides`.

---

## Benchmark B — which ESM-2 checkpoint is best?

**Run this first.** Benchmark A reuses its 650M results as its `esm` arm, so
doing B first saves 1 380 tasks.

### Stages 2 + 3

```bash
cd code/benchmark/simulate_experiments/scripts

for m in esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D; do
    sbatch foundations_substitutions.sh $m   # array 0-699
    sbatch foundations_indels.sh        $m   # array 0-399
    sbatch foundations_peptides.sh      $m   # array 0-279
done

# once a family's four arrays have finished
sbatch run_aggregate_foundations.sh substitutions
sbatch run_aggregate_foundations.sh indels
sbatch run_aggregate_foundations.sh peptides
```

Queue them smallest checkpoint first — 8M and 35M are far cheaper than 650M, so
you get most of the answer early. Results land in
`simulation_output/foundations/<family>/<dataset>/<checkpoint>/seed_*/train_size_*/`,
and one aggregation run covers all four checkpoints in that family.

### Stage 4

```bash
cd code/benchmark/statistics
# 1. run foundation_models.ipynb
#      -> results/foundations_within_dataset.csv            level 1
#      -> results/foundations_cell_means.csv
#      -> results/foundations_across_datasets.csv           level 2
#      -> results/foundations_across_datasets_pairwise.csv
#      -> results/foundations_mean_table.csv                headline answer

cd ..
# 2. run visualizations_foundations.ipynb   (independent of step 1)
#      -> foundation_curves_<family>_{r2,spearman}.png/.pdf
#      -> foundation_summary.png/.pdf
```

The statistics work on the metric values themselves at two levels, with the
same test at each: a repeated-measures ANOVA followed by a Tukey HSD for every
pair of checkpoints.

1. **Within a dataset** — the random seed is the repeated measure, per dataset
   and training size.
2. **Across datasets** — each (family, dataset, training size) cell is the
   repeated measure, on the mean over seeds. This answers the headline question:
   whether one checkpoint is better *on average across datasets*, rather than on
   a single one. It is reported over all datasets and per family.

The figures show Test R² and Test Spearman ρ. Per family there are learning
curves per dataset (error bars ±1 SD across seeds). The summary figure has the
mean per checkpoint and family (bars) above the mean across datasets against
training size (lines), with error bars ±1 s.e.m. across datasets. Indels use a
different training-size grid (20–80) from substitutions and peptides (20–500),
so they are drawn as their own dashed line rather than averaged into one line
whose dataset composition would change from point to point.

---

## Benchmark A — does ESM-2 beat the three baselines?

### Stages 2 + 3

```bash
cd code/benchmark/simulate_experiments/scripts

# Only the three baselines: esm comes from benchmark B.
for rep in physchem morgan ngram; do
    sbatch substitutions.sh $rep     # array 0-699
    sbatch indels.sh        $rep     # array 0-399
    sbatch peptides.sh      $rep     # array 0-279
done

# when those finish (4 140 tasks total)
sbatch run_aggregate.sh substitutions
sbatch run_aggregate.sh indels
sbatch run_aggregate.sh peptides
```

The `esm` arm is not submitted here. It is the identical computation to
benchmark B's 650M arm — same datasets, same seeds 42–51, same training sizes,
the same `optimize_models.py` call with
`--representation esm --foundation_model facebook/esm2_t33_650M_UR50D` — and
`save_results` even files both at the same relative path,
`<dataset>/esm2_t33_650M_UR50D/`. So the aggregation reads those runs out of
`simulation_output/foundations/<family>/` (`--reuse_esm_from`, on by default) and
prints how many it reused. Only the 650M checkpoint is taken, so 8M/35M/150M
never enter the representation comparison, and a run present in both trees is
counted once. If benchmark B has not been run for that family, the aggregation
stops with an explicit message rather than writing a summary with no `esm` rows.

To make benchmark A stand on its own instead — four representations, no
dependency on B:

```bash
for rep in esm physchem morgan ngram; do
    sbatch substitutions.sh $rep
    sbatch indels.sh        $rep
    sbatch peptides.sh      $rep
done

REUSE_ESM=0 sbatch run_aggregate.sh substitutions   # and indels, peptides
```

To queue the representations in sequence rather than all at once, each script's
header has a ready-made dependency loop.

### Stage 4

Order matters: the visualizations read the tables the statistics notebooks write.

```bash
cd code/benchmark/statistics
# 1. run substitutions.ipynb   -> results/substitutions_results*.csv
# 2. run indels.ipynb          -> results/indels_results*.csv
# 3. run peptides.ipynb        -> results/peptides_results*.csv

cd ..
# 4. run visualizations_3.ipynb   (needs the three results/*.csv above)
```

Statistics are a repeated-measures Tukey HSD with the random seed as the
repeated measure. With four representations there are six pairwise comparisons
per dataset and training size, and every one is recorded.

---

## Benchmark C — active learning workflow simulations

Each run writes its own CSV to `simulation_results/runs/`, one file per
(dataset, representation, strategy, seed). There is no separate aggregation job:
`load_results()` stitches the per-run files together at read time (plus a legacy
`df_all.csv` if one is present), drops any torn rows, and lets a re-run's file
replace its earlier one. To produce a single combined `df_all.csv` for a fresh
checkout, run `scripts/rebuild_df_all.sh` after the array finishes.

Two candidate selection strategies are simulated per round of 20 peptides,
passed as the second argument:

| Strategy | Batch composition | Idea |
| --- | --- | --- |
| `greedy` | 20 highest-predicted | Exploit the model's current ranking |
| `mixed` | 14 highest- + 6 lowest-predicted | Spend 30% of the budget on the region the model likes least, which labels the part of the space the model is most likely to be wrong about |

### Stage 2

```bash
cd code/workflow_simulation/scripts

for rep in esm physchem morgan ngram; do
    for acq in greedy mixed; do
        sbatch simulation_per_seed.sh $rep $acq   # array 0-139
    done
done
# -> simulation_results/runs/*.csv               (1 120 tasks total)
```

`simulation_per_seed.sh` carries a per-dataset optimisation direction: every
dataset uses `--direction maximize` except HemoPI2, which uses
`--direction minimize` (a lower −log10(HC50) is less haemolytic and thus more
desirable). The direction is applied once inside `prep_data` by negating the
target, so selection, the "true top 10%" set and every normalised mean stay
consistent; the reported metrics are unchanged in meaning.

A run is one (dataset, seed, strategy): the two strategies start from the same
seeded initial 20 peptides and diverge from the first acquisition onward, since
each one's training set, refit model and remaining pool depend on what it
acquired. The 30% is `--explore_frac`, set once as `EXPLORE_FRAC` in
*simulation_per_seed.sh*; the figure legends derive their labels from it, so
changing it does not leave stale text behind. It does not apply to `greedy`,
which always takes the top 20.

Every row is tagged with a `Representation` and an `Acquisition` column. Because
each run owns its own file in `runs/`, re-running a configuration overwrites that
run's file rather than appending a duplicate, so re-runs are safe and no rows are
silently double-counted.

### Rebuilding a combined df_all.csv

For reproducibility a single `df_all.csv` holding every run can be rebuilt from
the per-run files:

```bash
cd code/workflow_simulation/scripts
mkdir -p logs
sbatch rebuild_df_all.sh          # runs/ -> ../simulation_results/df_all.csv
# CHECK=1 sbatch rebuild_df_all.sh   preview only
```

It reads only `runs/` (it never folds an existing `df_all.csv` back into itself)
and reuses the pipeline's own `load_results()`, so the rebuilt file is identical
to what the notebooks consume. The notebooks themselves keep reading both
`df_all.csv` and `runs/`, so this step is for producing the standalone file, not
required for the notebooks to work.

### HemoPI2 minimize rerun (patching an existing results set)

If a results set was produced before HemoPI2 was switched to `minimize`, it can
be corrected without rerunning every dataset:

```bash
cd code/workflow_simulation/scripts
mkdir -p logs
sbatch rerun_hemopi2_minimize.sh    # HemoPI2 only, all reps x both strategies, minimize
sbatch replace_hemopi2.sh           # remove old HemoPI2 run files, copy in the new ones
sbatch rebuild_df_all.sh            # regenerate the combined df_all.csv
```

`replace_hemopi2.sh` also strips any old HemoPI2 rows from a legacy `df_all.csv`,
so `load_results()` returns only the corrected (minimize) HemoPI2. Every other
dataset is left untouched. Set `DRY_RUN=1` before `replace_hemopi2.sh` to preview.

### Stage 4

```bash
cd code/workflow_simulation
# run simulation.ipynb
#   -> simulation_results/figures/workflow_*.png          both strategies,
#                                                         colour = representation,
#                                                         style = strategy
#   -> simulation_results/figures/workflow_*_greedy.png   one strategy each
#   -> simulation_results/figures/workflow_*_mixed.png
#   + a paired greedy-vs-mixed end-point comparison printed in the last cell
```

A second notebook produces the appendix seed-variance figure:

```bash
cd code/workflow_simulation
# run seed_variance_analysis.ipynb
#   -> simulation_results/figures/seed_variance/seed_variance_summary.png/.pdf
```

It reads the SCARSE greedy results via `load_results()` and asks why some
datasets vary more across the 10 random seeds than others. The single figure has
five panels: (a) end-point std per dataset, (b) per-seed trajectory fans,
(c) how the seed-to-seed spread grows over the rounds, and (d, e) the end-point
performance of each seed against two properties of its initial 20-peptide seed
set (the label spread and the best label present). Each seed feature is oriented
by the dataset's optimisation direction, so "a better initial seed" means the
same thing for every dataset, including HemoPI2 (where a lower −log10(HC50) is
better). Two exploratory cells additionally relate the outcome to the sequence
and physicochemical-descriptor similarity within each initial seed.

---

## Benchmark D — CV score vs end-point performance

### Stages 2 + 3

```bash
cd code/selection_analysis/scripts

for rep in esm physchem morgan ngram; do
    sbatch benchmark_script_substitutions.sh $rep   # array 0-899
    sbatch benchmark_script_peptides.sh      $rep   # array 0-359
done

sbatch run_aggregate.sh   # -> simulation_output/simulation_summary.csv
```

### Stage 4

**This one needs benchmark C to have finished**, because it relates CV scores to
active-learning end points.

```bash
cd code/selection_analysis
# run selecting_workflow_eval.ipynb
#   reads ../workflow_simulation/simulation_results/ via load_results()   (C)
#         (the runs/ files plus any legacy df_all.csv)
#   reads ./simulation_output/simulation_summary.csv                      (D)
#   -> results/cv_vs_endpoint_correlations.csv
#   -> results/cv_metric_comparison.csv
#   -> results/cv_vs_endpoint_correlation_summary.png
```

Run for both CV R² and CV Spearman ρ, and for all four representations. The CV
score of a representation is related to the end-point performance of that same
representation, so the join is on dataset *and* representation. The CV score
does not depend on how candidates were acquired, so each CV row is compared
against both strategies' end points and every correlation is reported per
strategy. The notebook ends by naming which CV metric is the better predictor,
for each strategy.

---

## Dependencies between stages

* **B runs before A**: A's aggregation reads B's 650M results as its `esm` arm.
  With `REUSE_ESM=0` and `<family>.sh esm` submitted, A is independent again.
* C and D's stages 2 + 3 are independent of everything else.
* D's stage 4 needs C's results (read from `runs/` plus any `df_all.csv` via
  `load_results()`).
* Within A and B, the statistics notebooks must run before the visualizations.

## Compute

| Benchmark | Tasks per submission | Submissions | Total |
| --- | --- | --- | --- |
| A. Representations | 700 + 400 + 280 = 1 380 | 3 baselines (esm reused from B) | 4 140 |
| B. Foundation models | 700 + 400 + 280 = 1 380 | 4 checkpoints | 5 520 |
| C. Active learning | 140 | 4 representations × 2 strategies | 1 120 |
| D. Selection analysis | 900 + 360 = 1 260 | 4 representations | 5 040 |

Array sizes are all kept under 1 000, the usual SLURM `MaxArraySize`. That is
why the representation and the checkpoint are passed as arguments rather than
folded into the array index, and why D is split into two scripts.

## Citation

If using this work, please cite: <br>
Andrekson L, Rydbergh R, Mercado R, Wenzel M. AI-guided discovery for low-resource peptide engineering using evolutionary scale modeling. bioRxiv. 2026. https://doi.org/10.64898/2026.06.25.734678.