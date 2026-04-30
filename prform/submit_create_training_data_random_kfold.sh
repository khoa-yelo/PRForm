#!/bin/bash
#SBATCH --job-name=prform_random_kfold_csvs
#SBATCH --output=logs/random_kfold_csvs_%j.out
#SBATCH --error=logs/random_kfold_csvs_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000151
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4

# Generates per-fold training_data.csv for both species- and sequence-level
# random K-fold splits (companion to genus-blocked k-fold). Run before the
# h5 array jobs (submit_create_datasets_{species,sequence}_kfold.sh).

mkdir -p logs

SCRIPT=/projects/m000151/khoa/repos/PRForm/scripts/create_training_data_kfold_random.py

echo "=== species-level random k-fold ==="
python "${SCRIPT}" --split-by species

echo
echo "=== sequence-level random k-fold ==="
python "${SCRIPT}" --split-by sequence

echo
echo "=== Done ==="
