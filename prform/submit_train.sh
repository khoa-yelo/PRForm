#!/bin/bash
#SBATCH --job-name=prform_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4

mkdir -p logs

# eval "$(micromamba shell hook --shell bash)"
# micromamba activate py312_PRForm

# check cuda is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "CUDA is not available"
    exit 1
fi
# check pytoch cuda is available
if ! python -c "import torch; print(torch.cuda.is_available())" > /dev/null 2>&1; then
    echo "PyTorch CUDA is not available"
    exit 1
fi
python train.py \
    --train_data /farmshare/user_data/khoang99/repos/PRForm/training_data/training_data_random/train.h5 \
    --val_data /farmshare/user_data/khoang99/repos/PRForm/training_data/training_data_random/val.h5 \
    --output_dir /farmshare/user_data/khoang99/repos/PRForm/training_data/training_data_random/outputs \
    --batch_size 36 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --flank 5000 \
    --mid_channels 32 \
    --dropout 0.4 \
    --warmup_epochs 5 \
    --augment \
    --pos_weight 5.0 \
    --seed 42 