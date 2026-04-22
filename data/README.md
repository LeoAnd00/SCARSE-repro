# Code for reproducing results

## Preperations
1. Create the following folder structure for the data:
```
project-root/
├── data/
    ├── cellppd/
    │   └── raw/
    │       ├── cellppd_negative.csv/
    │       └── cellppd_positive.csv/
    ├── mbc/
    │    └── raw/
    │       └──  EC.csv/
    ├── proteingym_dms/
    │    └── raw/
    │       ├── DMS_ProteinGym_indels/
    │       └── DMS_ProteinGym_substitutions/
    └── toxinpred3/
        └── raw/
           ├── train_neg.csv/
           └── train_pos.csv/
```
2. Download all required data and place the files in the raw folder for each dataset. 
    * Substitution and indel DMS assay datasets: https://proteingym.org/download
    * Short AMPs: https://journals.asm.org/doi/10.1128/msystems.00345-23
    * CellPPD: https://webs.iiitd.edu.in/raghava/cellppd/algo.php
    * ToxinPred3: https://webs.iiitd.edu.in/raghava/toxinpred3/ 
3. Set up environment using *requirements.txt*.
4. Go to */code/benchmark/process_data* and run *process_data.ipynb*.

