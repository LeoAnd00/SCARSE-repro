#!/usr/bin/env bash
#SBATCH -A C3SE2026-1-32
#SBATCH -p vera
#SBATCH -t 01:00:00
#SBATCH -C ZEN4
#SBATCH -n 1

ml purge

# Set variables
OUT_DIR="/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/reproducibility_code/code/selection_analysis/simulation_output" # Root folder with per-seed CSVs
SUMMARY_CSV="${OUT_DIR}/simulation_summary.csv"   # Output aggregated summary CSV

# Run the Python aggregation script
apptainer exec /cephyr/users/leoan/Alvis/Desktop/mimer_naiss2023-6-290/Leo_Andrekson/reproducibility_code/env.sif python aggregate_simulation_results.py \
    "$OUT_DIR" "$SUMMARY_CSV"
