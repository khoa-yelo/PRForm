#!/bin/bash
#SBATCH --job-name=prform_predict
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000151
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4

mkdir -p logs

# eval "$(micromamba shell hook --shell bash)"
# micromamba activate py312_PRForm

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

BASE=/projects/m000151/khoa/repos/PRForm/training_data
CKPT=model_best.pth
FLANK=5000
MID_CHANNELS=32
DROPOUT=0.4
BATCH_SIZE=36
NEG_RATIO=1000

for DATASET in training_data_random training_data_genus_block training_data_species_block; do
    echo "=== Predicting on test set: ${DATASET} ==="
    python predict.py \
        --checkpoint  "${BASE}/${DATASET}/outputs/${CKPT}" \
        --data        "${BASE}/${DATASET}/test.h5" \
        --output_dir  "${BASE}/${DATASET}/outputs/predictions_test" \
        --flank       ${FLANK} \
        --mid_channels ${MID_CHANNELS} \
        --dropout     ${DROPOUT} \
        --batch_size  ${BATCH_SIZE} \
        --neg_ratio   ${NEG_RATIO}
done

echo "=== All predictions complete ==="
