# SCARSE: Small-sample Classification And Regression Solution for low-resource peptide Engineering - Reproducability code
<p align="center">
  <img src="figures/model_architecture.png" alt="Workflow Diagram" width="500"/>
</p>

## Abstract
Reliable estimation of downstream performance in low-data peptide machine learning is critical for guiding early-stage AI-driven peptide engineering, yet it is often unclear how to assess whether a model will be effective in iterative discovery settings. Here, we show that cross validation R² score can serve as a simple and robust proxy for predicting active learning workflow performance, enabling early-stage evaluation of model suitability for sequential peptide optimization. To support this, we introduce SCARSE, a machine learning framework combining ESM-2 protein language model embeddings with Gaussian process regression and extremely randomized trees classification, designed for low-resource peptide property prediction (20–500 training samples). We benchmark SCARSE across 23 peptide and small-protein datasets covering substitution and indel variants, antimicrobial peptides, cell-penetrating peptides, and toxic/non-toxic peptides. The protein language model approach significantly outperforms a hand-engineered descriptor baseline on substitution and indel tasks, while the two approaches achieve comparable performance on shorter peptide non-mutant datasets where simpler descriptors capture enough of the signal. In simulated active learning workflows, SCARSE consistently outperforms baseline and random sampling strategies, and notably we demonstrate that CV R² computed from as few as 50 labeled peptides can be sufficient to estimate final active learning endpoint performance, providing a practical, data-efficient criterion for deciding whether a given dataset combined with SCARSE is suitable for iterative peptide discovery. SCARSE is released as a pip package and is available via HuggingFace Spaces to facilitate integration into peptide engineering workflows.

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

* Before running all the code below, update all the paths in the .sh files so that the path to the datasets, environment and the output path are correct for your setup.
* Make sure you've downloaded ESM-2 650M from HuggingFace and specify the path to it in the .sh files. It will automatically be downloaded the first time one runs the scripts, but make sure to only run one job the first time, so that errors does not occur from multiple jobs trying to download the same model at the same time.

## Benchmark
1. Go to */code/benchmark/simulate_experiments/scripts* and run *sbatch* for each script.
2. Go to */code/benchmark/simulate_experiments* and run *run_aggregate.sh* and *run_aggragate_classification.sh* for each dataset (manually change the dataset name each time running the scripts).
3. Go to */code/benchmark/simulate_experiments/statistics* and run all notebooks.
4. Go to */code/benchmark/simulate_experiments* and run *visualizations_3.ipynb*.

## Active learning workflow simulations
1. Go to */code/workflow_simulation/scripts* and run *simulation_per_seed.sh* and *baseline_simulation_per_seed.sh*.
2. Go to */code/workflow_simulation* and run *simulation.ipynb*.

## Investigation into CV $R^2$ score correlation with end-point active learning simulation performance
1. Go to */code/selection_analysis* and run *benchmark_script.sh*.
2. Run *run_aggregate.sh*.
3. Run *selecting_workflow_eval.ipynb*.

