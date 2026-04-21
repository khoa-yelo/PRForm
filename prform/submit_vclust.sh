#!/bin/bash
#SBATCH --job-name=vclust_ani90
#SBATCH --output=logs/vclust_ani90_%j.out
#SBATCH --error=logs/vclust_ani90_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000151
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16

mkdir -p logs

INPUT=/projects/m000151/khoa/repos/PRForm/training_data/mmseqs_clusters/contigs.fasta
OUTDIR=/projects/m000151/khoa/repos/PRForm/training_data/vclust_ani80
THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p "$OUTDIR"

echo "=== vclust prefilter ==="
vclust prefilter \
    -i "$INPUT" \
    -o "$OUTDIR/fltr.txt" \
    -t "$THREADS"

echo "=== vclust align ==="
vclust align \
    -i "$INPUT" \
    -o "$OUTDIR/ani.tsv" \
    --filter "$OUTDIR/fltr.txt" \
    -t "$THREADS"

echo "=== vclust cluster (ANI >= 0.9) ==="
vclust cluster \
    -i "$OUTDIR/ani.tsv" \
    -o "$OUTDIR/clusters.tsv" \
    --ids "$OUTDIR/ani.ids.tsv" \
    --metric ani \
    --ani 0.90

echo "=== Done. Results in $OUTDIR ==="
