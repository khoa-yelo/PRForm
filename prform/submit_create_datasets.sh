#!/bin/bash
#SBATCH --job-name=prform_datasets
#SBATCH --output=logs/create_datasets_%j.out
#SBATCH --error=logs/create_datasets_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4

mkdir -p logs

# echo "=== Creating dataset: training_data_random ==="
# python ./utils/create_datasets.py \
#     --csv ../training_data/training_data_random.csv \
#     --format h5 \
#     --outdir ../training_data/training_data_random

echo "=== Creating dataset: training_data_genus_block ==="
python ./utils/create_datasets.py \
    --csv ../training_data/training_data_genus_block.csv \
    --format h5 \
    --outdir ../training_data/training_data_genus_block

# echo "=== Creating dataset: training_data_species_block ==="
# python ./utils/create_datasets.py \
#     --csv ../training_data/training_data_species_block.csv \
#     --format h5 \
#     --outdir ../training_data/training_data_species_block

# echo "=== All datasets created ==="
