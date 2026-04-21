#!/bin/bash
#SBATCH --job-name=prform_datasets_genus_kfold
#SBATCH --output=logs/create_datasets_genus_kfold_%A_%a.out
#SBATCH --error=logs/create_datasets_genus_kfold_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000151
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4

mkdir -p logs

FOLD_DIR=../training_data/genus_block_kfold/fold_${SLURM_ARRAY_TASK_ID}

echo "=== Creating dataset for fold ${SLURM_ARRAY_TASK_ID}: ${FOLD_DIR} ==="
python ./utils/create_datasets.py \
    --csv ${FOLD_DIR}/training_data.csv \
    --format h5 \
    --outdir ${FOLD_DIR}

echo "=== Done fold ${SLURM_ARRAY_TASK_ID} ==="
