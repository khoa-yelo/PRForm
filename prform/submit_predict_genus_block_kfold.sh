#!/bin/bash
#SBATCH --job-name=genus_kfold_predict
#SBATCH --output=logs/predict_genus_kfold_%A_%a.out
#SBATCH --error=logs/predict_genus_kfold_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000151
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
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

FOLD_DIR=/projects/m000151/khoa/repos/PRForm/training_data/genus_block_kfold/fold_${SLURM_ARRAY_TASK_ID}
CKPT=${FOLD_DIR}/outputs/model_best.pth
FLANK=5000
MID_CHANNELS=32
DROPOUT=0.4
BATCH_SIZE=36
NEG_RATIO=1000

for SPLIT in test train val; do
    echo "=== Predicting fold ${SLURM_ARRAY_TASK_ID} split: ${SPLIT} ==="
    python predict.py \
        --checkpoint   "${CKPT}" \
        --data         "${FOLD_DIR}/${SPLIT}.h5" \
        --output_dir   "${FOLD_DIR}/outputs/predictions_${SPLIT}" \
        --flank        ${FLANK} \
        --mid_channels ${MID_CHANNELS} \
        --dropout      ${DROPOUT} \
        --batch_size   ${BATCH_SIZE} \
        --neg_ratio    ${NEG_RATIO}
done

echo "=== Done predictions fold ${SLURM_ARRAY_TASK_ID} ==="
