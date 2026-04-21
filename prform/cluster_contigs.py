"""
Cluster DNA contigs with MMseqs2 easy-linclust and attach cluster IDs
back onto the training-data CSV.

Usage:
    python cluster_contigs.py \
        --csv /projects/m000151/khoa/repos/PRForm/training_data/training_data_template.csv \
        --out-dir /projects/m000151/khoa/repos/PRForm/training_data/mmseqs_clusters \
        --min-seq-id 0.8 --coverage 0.8

Produces:
    <out-dir>/contigs.fasta            deduplicated input FASTA
    <out-dir>/mmseqs/*                 raw MMseqs2 outputs
    <out-dir>/cluster_map.tsv          record_id -> mmseqs_cluster_id
    <out-dir>/training_data_clustered.csv  input CSV + mmseqs_cluster_id column
"""

import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd


def write_fasta(df: pd.DataFrame, fasta_path: str) -> int:
    seen = set()
    n = 0
    with open(fasta_path, "w") as f:
        for rid, seq in zip(df["record_id"], df["sequence"]):
            if rid in seen or not isinstance(seq, str) or len(seq) == 0:
                continue
            seen.add(rid)
            f.write(f">{rid}\n{seq}\n")
            n += 1
    return n


def run_mmseqs(fasta: str, work_dir: str, min_id: float, cov: float, threads: int) -> str:
    os.makedirs(work_dir, exist_ok=True)
    prefix = os.path.join(work_dir, "clusterRes")
    tmp = os.path.join(work_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    cmd = [
        "mmseqs", "easy-linclust", fasta, prefix, tmp,
        "--min-seq-id", str(min_id),
        "-c", str(cov),
        "--cov-mode", "1",
        "--threads", str(threads),
        "--dbtype", "2",
    ]
    print("[mmseqs]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return prefix + "_cluster.tsv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-seq-id", type=float, default=0.8,
                    help="MMseqs2 --min-seq-id (0.7-0.8 typical for DNA)")
    ap.add_argument("--coverage", type=float, default=0.8,
                    help="MMseqs2 -c (alignment coverage)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--id-col", default="record_id")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    for col in (args.id_col, args.seq_col):
        if col not in df.columns:
            sys.exit(f"missing column: {col}")

    fasta = os.path.join(args.out_dir, "contigs.fasta")
    n_unique = write_fasta(
        df.rename(columns={args.id_col: "record_id", args.seq_col: "sequence"}),
        fasta,
    )
    print(f"[fasta] {n_unique} unique contigs -> {fasta}", flush=True)

    cluster_tsv = run_mmseqs(
        fasta,
        os.path.join(args.out_dir, "mmseqs"),
        args.min_seq_id,
        args.coverage,
        args.threads,
    )

    cmap = pd.read_csv(cluster_tsv, sep="\t", header=None,
                       names=["mmseqs_cluster_id", "record_id"])
    cmap.to_csv(os.path.join(args.out_dir, "cluster_map.tsv"),
                sep="\t", index=False)

    n_clusters = cmap["mmseqs_cluster_id"].nunique()
    print(f"[cluster] {n_unique} contigs -> {n_clusters} clusters "
          f"(reduction {n_unique / max(n_clusters, 1):.1f}x)", flush=True)

    merged = df.merge(cmap, left_on=args.id_col, right_on="record_id",
                      how="left", suffixes=("", "_cmap"))
    if "record_id_cmap" in merged.columns:
        merged = merged.drop(columns=["record_id_cmap"])
    n_missing = merged["mmseqs_cluster_id"].isna().sum()
    if n_missing:
        print(f"[warn] {n_missing} rows without a cluster assignment", flush=True)

    out_csv = os.path.join(args.out_dir, "training_data_clustered.csv")
    merged.to_csv(out_csv, index=False)
    print(f"[done] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
