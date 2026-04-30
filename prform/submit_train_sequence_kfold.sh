#!/bin/bash
#SBATCH --job-name=sequence_kfold_train
#SBATCH --output=logs/train_sequence_kfold_%A_%a.out
#SBATCH --error=logs/train_sequence_kfold_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000151
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
mamba activate py312_PRForm

# check cuda is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "CUDA is not available"
    exit 1
fi
# check pytorch cuda is available
if ! python -c "import torch; print(torch.cuda.is_available())" > /dev/null 2>&1; then
    echo "PyTorch CUDA is not available"
    exit 1
fi

FOLD_DIR=/projects/m000151/khoa/repos/PRForm/training_data/sequence_random_kfold/fold_${SLURM_ARRAY_TASK_ID}

echo "=== Training fold ${SLURM_ARRAY_TASK_ID}: ${FOLD_DIR} ==="

python train.py \
    --train_data ${FOLD_DIR}/train.h5 \
    --val_data ${FOLD_DIR}/val.h5 \
    --output_dir ${FOLD_DIR}/outputs \
    --batch_size 36 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --flank 5000 \
    --mid_channels 32 \
    --dropout 0.4 \
    --warmup_epochs 5 \
    --augment \
    --pos_weight 2.0 \
    --seed 42

echo "=== Done fold ${SLURM_ARRAY_TASK_ID} ==="
