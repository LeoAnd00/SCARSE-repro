# Code for reproducing results

## Preperations
1. Create the following folder structure for the data:
```
project-root/
├── data/
    ├── proteingym_dms/
    │    └── raw/
    │       ├── DMS_ProteinGym_indels/           per-assay CSVs
    │       ├── DMS_ProteinGym_substitutions/    per-assay CSVs
    │       ├── DMS_indels.csv                   assay reference table
    │       └── DMS_substitutions.csv            assay reference table
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
2. Download all required data and place the files in the raw folder for each dataset. 
    * Substitution and indel DMS assay datasets: https://proteingym.org/download
    * E.coli AMPs: https://doi.org/10.1016/j.isci.2024.110718
    * S.aureus AMPs: https://doi.org/10.1016/j.isci.2024.110718
    * P.aeruginosa AMPs: https://doi.org/10.1016/j.isci.2024.110718
    * HemoPI2: https://doi.org/10.1038/s42003-025-07615-w
3. Set up environment using *requirements.txt*.
4. Go to */code/benchmark/process_data* and run *process_data.ipynb*.
5. Then run *visualize_processed_data.ipynb* in the same folder.

## Processed output
*process_data.ipynb* writes a `processed/` folder next to each `raw/` folder. Cleaned
files use a common schema: `sequence`, `score`, and — where applicable —
`sequence_length` and `classes`.

The datasets that ship as ready-made splits are fused back together and saved as a
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
median-based class split, so classes reflect the sequences that are kept. All four
datasets are already within the cap, so it currently drops nothing — it is there to
keep any re-downloaded or extended raw data within range.

### Score conventions
All scores are oriented so that **higher = more active**:
* AMP datasets — `score` = pMIC = `-log10(MIC[uM])`, i.e. `-NEW-CONCENTRATION`.
  Set `SCORE_AS_PMIC = False` in the *E.coli AMPs* cell to keep the raw
  `log10(MIC)` orientation instead.
* HemoPI2 — `score` = `-log10(HC50[uM])`; labels mark HC50 <= 100 uM as haemolytic.

## Strategic similarity-based splits
The last section of *process_data.ipynb* builds an alternative split for each dataset
from its fused file. Within each sequence length a reference sequence is drawn at
random; the closest sequences become the test set and the furthest of the remainder
become the training set, so the length profile of the full dataset is preserved while
the test set is a tight similarity cluster and the training set a diverse one.

Splits are plotted but not written to disk unless `SAVE_SPLITS = True`.