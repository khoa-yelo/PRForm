#!/usr/bin/env python

"""
create_datasets.py  –  Build training data for Programmed Ribosomal Frameshift (PRF) prediction

This script:
  1. Reads a CSV file with viral genome sequences and annotated PRF positions.
  2. For each record, one-hot encodes the sequence (reverse-complementing if on minus strand).
  3. Pads to a multiple of block_len, adds flanking context on both sides.
  4. Slices into non-overlapping blocks of (block_len + 2*flank) for X, with Y covering
     the inner block_len only (matching SpliceAI blocking).
  5. Generates a binary label track per block: 1 at PRF site(s), 0 elsewhere.
  6. Splits into train/val/test based on the 'split' column and writes output files.

Input CSV columns:
  accession_id   – id of the viral genome
  record_id      – id of the record in the corresponding accession
  cluster_id     – a record_id representative id
  prf_position   – position of PRF event (1-based); can be comma-separated for multiple sites
  strand         – +/- strand
  sequence       – raw DNA/RNA sequence
  split          – train / val / test
  species_taxid  – (optional) NCBI taxonomy ID for the species
  genus_taxid    – (optional) NCBI taxonomy ID for the genus
  species_name   – (optional) species name string
  genus_name     – (optional) genus name string
  sample_weight  – (optional) per-sample training weight (float, default 1.0)

Output per split:
  Pickle (.pkl) – list of dicts with keys:
    'accession_id', 'record_id', 'cluster_id', 'block_idx', 'species_taxid', 'genus_taxid', 'species_name', 'genus_name',
    'sample_weight' (float),
    'sequence' (one-hot, shape [block_len + 2*flank, 4]),
    'y' (binary label, shape [block_len, 1])

  HDF5 (.h5) – datasets per split:
    X     – (N, block_len + 2*flank, 4) uint8
    Y     – (N, block_len, 1) uint8
    metadata (group) –
      accession_id   – (N,) variable-length string
      record_id      – (N,) variable-length string
      cluster_id     – (N,) variable-length string
      species_taxid  – (N,) variable-length string
      genus_taxid    – (N,) variable-length string
      species_name   – (N,) variable-length string
      genus_name     – (N,) variable-length string
      block_idx      – (N,) int32
      sample_weight  – (N,) float32

Usage:
  python create_datasets.py \
    --csv       prf_data.csv \
    --outdir    data/prf \
    --block_len 15000 \
    --flank     5000 \
    --format    h5

Requirements:
  Python 3.7+, numpy, pandas, biopython, tqdm
  h5py (only if --format h5)
"""

import os
import argparse
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm
import pickle
import logging


def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot encode a nucleotide sequence.
    A=0, C=1, G=2, T/U=3. Ambiguous bases get all-zero rows.

    Parameters
    ----------
    seq : str
        Nucleotide sequence (uppercase recommended).

    Returns
    -------
    np.ndarray of shape (len(seq), 4), dtype uint8
    """
    nt_map = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    L = len(seq)
    oh = np.zeros((L, 4), dtype=np.uint8)
    for i, base in enumerate(seq):
        idx = nt_map.get(base.upper())
        if idx is not None:
            oh[i, idx] = 1
    return oh


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA/RNA sequence."""
    return str(Seq(seq).reverse_complement())


def process_record_blocks(row, block_len: int, flank: int):
    """
    Process a single CSV row into one or more fixed-size training blocks.

    Follows SpliceAI blocking logic:
      1. One-hot encode the full sequence (reverse-complement if minus strand).
      2. Pad to the next multiple of block_len.
      3. Add flank on each side.
      4. Slice into non-overlapping windows of (block_len + 2*flank) for X,
         with Y covering only the inner block_len.
      5. All blocks are yielded (even those without a PRF site).

    Parameters
    ----------
    row : pd.Series
        A row from the input CSV.
    block_len : int
        Core block length for labels (analogous to SpliceAI's 5000).
    flank : int
        Extra flanking context added on each side (included in X, not in Y).

    Yields
    ------
    dict with keys: accession_id, record_id, cluster_id,
         species_taxid, genus_taxid, species_name, genus_name,
         block_idx, sequence (np.ndarray shape [block_len+2*flank, 4]),
         y (np.ndarray shape [block_len, 1])
    """
    raw_seq = str(row["sequence"]).strip()
    strand = str(row["strand"]).strip()

    # Parse PRF position(s) – support comma-separated for multiple sites
    prf_raw = str(row["prf_position"]).strip()
    prf_positions = []
    for p in prf_raw.split(","):
        p = p.strip()
        if p and p.lower() != "nan":
            prf_positions.append(int(float(p)))  # 1-based

    L = len(raw_seq)

    # --- Handle strand ---
    if strand in ("-", "-1"):
        raw_seq = reverse_complement(raw_seq)
        # Mirror PRF positions: if original pos is p (1-based), on rev-comp it
        # maps to L - p + 1 (still 1-based)
        prf_positions = [L - p + 1 for p in prf_positions]

    # Convert PRF positions to 0-based
    prf_positions_0b = [p - 1 for p in prf_positions]

    # --- One-hot encode ---
    oh = one_hot_encode(raw_seq)  # (L, 4)
    L = oh.shape[0]

    # --- Pad to next multiple of block_len ---
    pad = (-L) % block_len
    if pad:
        oh = np.pad(oh, ((0, pad), (0, 0)), constant_values=0)

    # --- Add flanking context on both sides ---
    oh = np.pad(oh, ((flank, flank), (0, 0)), constant_values=0)
    total = oh.shape[0]

    # --- Slice into non-overlapping blocks ---
    window = block_len + 2 * flank
    idx = 0
    blk_idx = 0
    while idx + window <= total:
        block_x = oh[idx : idx + window]  # (block_len + 2*flank, 4)

        # Build Y for inner block_len region
        y = np.zeros((block_len, 1), dtype=np.uint8)
        for pos in prf_positions_0b:
            # pos is 0-based index in the original (pre-pad) sequence
            # In the padded+flanked array, the core region of this block
            # starts at idx+flank and covers block_len positions.
            # The core-relative position is: pos - (idx+flank - flank) = pos - idx
            rel = pos - idx
            if 0 <= rel < block_len:
                y[rel, 0] = 1

        yield {
            "accession_id": row["accession_id"],
            "record_id": row["record_id"],
            "cluster_id": row.get("cluster_id", ""),
            "species_taxid": str(row.get("species_taxid", "")),
            "genus_taxid": str(row.get("genus_taxid", "")),
            "species_name": str(row.get("species_name", "")),
            "genus_name": str(row.get("genus_name", "")),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "block_idx": blk_idx,
            "sequence": block_x,  # shape (block_len + 2*flank, 4)
            "y": y,               # shape (block_len, 1)
        }

        idx += block_len
        blk_idx += 1


def save_pickle(records, outdir):
    """Save each split as a pickle file."""
    for split_name, recs in records.items():
        out_path = os.path.join(outdir, f"{split_name}.pkl")
        logging.info(f"Saving {len(recs)} {split_name} blocks to {out_path}")
        with open(out_path, "wb") as f:
            pickle.dump(recs, f)


def save_h5(records, outdir):
    """
    Save each split as an HDF5 file.

    Layout per file:
      X              – (N, block_len + 2*flank, 4) uint8
      Y              – (N, block_len, 1)            uint8
      metadata/
        accession_id – (N,) variable-length string
        record_id    – (N,) variable-length string
        cluster_id   – (N,) variable-length string
        species_taxid – (N,) variable-length string
        genus_taxid   – (N,) variable-length string
        species_name – (N,) variable-length string
        genus_name   – (N,) variable-length string
        block_idx    – (N,) int32
    """
    import h5py

    for split_name, recs in records.items():
        if len(recs) == 0:
            logging.info(f"Skipping empty {split_name} split for HDF5")
            continue

        out_path = os.path.join(outdir, f"{split_name}.h5")
        logging.info(f"Saving {len(recs)} {split_name} blocks to {out_path}")

        N = len(recs)
        x_shape = recs[0]["sequence"].shape  # (block_len + 2*flank, 4)
        y_shape = recs[0]["y"].shape          # (block_len, 1)

        with h5py.File(out_path, "w") as hf:
            X_ds = hf.create_dataset("X", shape=(N, *x_shape), dtype=np.uint8)
            Y_ds = hf.create_dataset("Y", shape=(N, *y_shape), dtype=np.uint8)

            vlen_str = h5py.special_dtype(vlen=str)
            meta = hf.create_group("metadata")
            acc_ds = meta.create_dataset("accession_id", shape=(N,), dtype=vlen_str)
            rec_ds = meta.create_dataset("record_id", shape=(N,), dtype=vlen_str)
            clu_ds = meta.create_dataset("cluster_id", shape=(N,), dtype=vlen_str)
            sptx_ds = meta.create_dataset("species_taxid", shape=(N,), dtype=vlen_str)
            gntx_ds = meta.create_dataset("genus_taxid", shape=(N,), dtype=vlen_str)
            spnm_ds = meta.create_dataset("species_name", shape=(N,), dtype=vlen_str)
            gnnm_ds = meta.create_dataset("genus_name", shape=(N,), dtype=vlen_str)
            blk_ds = meta.create_dataset("block_idx", shape=(N,), dtype=np.int32)
            sw_ds = meta.create_dataset("sample_weight", shape=(N,), dtype=np.float32)

            for i, rec in enumerate(recs):
                X_ds[i] = rec["sequence"]
                Y_ds[i] = rec["y"]
                acc_ds[i] = str(rec["accession_id"])
                rec_ds[i] = str(rec["record_id"])
                clu_ds[i] = str(rec["cluster_id"])
                sptx_ds[i] = str(rec.get("species_taxid", ""))
                gntx_ds[i] = str(rec.get("genus_taxid", ""))
                spnm_ds[i] = str(rec.get("species_name", ""))
                gnnm_ds[i] = str(rec.get("genus_name", ""))
                blk_ds[i] = rec["block_idx"]
                sw_ds[i] = float(rec.get("sample_weight", 1.0))


def main():
    p = argparse.ArgumentParser(
        description="Create PRF prediction datasets from annotated CSV"
    )
    p.add_argument("--csv", required=True, help="Path to input CSV file")
    p.add_argument("--outdir", required=True, help="Output directory for pickle files")
    p.add_argument(
        "--block_len",
        type=int,
        default=15000,
        help="Core block length for labels (sequences are padded to multiples of this). Default: 15000",
    )
    p.add_argument(
        "--flank",
        type=int,
        default=5000,
        help="Flanking context on each side (included in X, not in Y). Default: 5000",
    )
    p.add_argument(
        "--format",
        choices=["pkl", "h5"],
        default="pkl",
        help="Output format: 'pkl' for pickle, 'h5' for HDF5. Default: pkl",
    )
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load CSV ---
    logging.info(f"Reading CSV from {args.csv}")
    df = pd.read_csv(args.csv)
    logging.info(f"Loaded {len(df)} records")

    required_cols = {"accession_id", "record_id", "prf_position", "strand", "sequence", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # --- Process each record into blocks ---
    records = {"train": [], "val": [], "test": []}
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        split = str(row["split"]).strip().lower()
        if split not in records:
            logging.warning(f"Unknown split '{split}' for record {row['record_id']}, skipping")
            skipped += 1
            continue

        try:
            for block in process_record_blocks(row, block_len=args.block_len, flank=args.flank):
                records[split].append(block)
        except Exception as e:
            logging.warning(f"Error processing record {row['record_id']}: {e}")
            skipped += 1
            continue

    # --- Summary ---
    for split_name, recs in records.items():
        n_pos = sum(int(r["y"].sum()) for r in recs)
        n_tx = len(set(r["record_id"] for r in recs))
        logging.info(
            f"{split_name:5s}: {len(recs):6d} blocks from {n_tx:5d} records, "
            f"{n_pos:6d} total PRF sites"
        )
    if skipped:
        logging.info(f"Skipped {skipped} records due to errors or unknown split")

    # --- Save ---
    if args.format == "pkl":
        save_pickle(records, args.outdir)
    elif args.format == "h5":
        save_h5(records, args.outdir)

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
