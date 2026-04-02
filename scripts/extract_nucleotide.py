"""
extract_nucleotide.py

Reads model_training.csv (or any CSV with 'fna_path' and 'record_id' columns),
extracts the full nucleotide sequence from the corresponding FASTA file for each
record_id, and adds it as a new 'sequence' column.

Usage:
    python scripts/extract_nucleotide.py                          # defaults
    python scripts/extract_nucleotide.py --input my.csv --output out.csv
"""

import argparse
import os
import pandas as pd
from Bio import SeqIO


def load_fasta_sequences(fna_path):
    """Parse a FASTA file and return a dict of {record_id: sequence_string}."""
    seq_dict = {}
    try:
        for record in SeqIO.parse(fna_path, "fasta"):
            seq_dict[record.id] = str(record.seq)
    except Exception as e:
        print(f"  Warning: could not parse {fna_path}: {e}")
    return seq_dict


def extract_nucleotides(df, base_dir=None):
    """
    Given a DataFrame with 'fna_path' and 'record_id' columns, look up
    each record's nucleotide sequence from the FASTA file and return
    the DataFrame with a new 'sequence' column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'fna_path' and 'record_id' columns.
    base_dir : str, optional
        Directory to resolve relative fna_path entries from.
        Defaults to the directory containing this script.
        The fna_path values in model_training.csv are relative to
        the scripts/ directory (e.g. '../viral_data_all/...').

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'sequence' column.
    """
    # Cache parsed FASTA files so each file is only read once
    fasta_cache = {}
    sequences = []
    missing_count = 0

    unique_paths = df["fna_path"].nunique()
    print(f"Extracting sequences for {len(df)} rows from {unique_paths} unique FASTA files...")

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    for i, (_, row) in enumerate(df.iterrows()):
        fna_path = row["fna_path"]
        record_id = row["record_id"]

        # Resolve relative path from base_dir
        # e.g. base_dir="scripts", fna_path="../viral_data_all/..." 
        #  -> "scripts/../viral_data_all/..." -> "viral_data_all/..."
        if pd.isna(fna_path):
            sequences.append(None)
            missing_count += 1
        else:
            full_path = os.path.normpath(os.path.join(base_dir, str(fna_path)))

            # Load and cache the FASTA file
            if full_path not in fasta_cache:
                fasta_cache[full_path] = load_fasta_sequences(full_path)

            seq = fasta_cache[full_path].get(record_id)
            if seq is None:
                missing_count += 1
            sequences.append(seq)

        if (i + 1) % 10000 == 0:
            print(f"  [{i + 1}/{len(df)}] processed...")

    df = df.copy()
    df["sequence"] = sequences

    print(f"Done. {len(df) - missing_count}/{len(df)} sequences found.")
    if missing_count > 0:
        print(f"  {missing_count} record_ids were not found in their FASTA files.")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract nucleotide sequences from FASTA files referenced in a CSV."
    )
    parser.add_argument(
        "--input", "-i",
        default="model_training.csv",
        help="Input CSV with 'fna_path' and 'record_id' columns (default: model_training.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: <input_stem>_with_seq.csv)",
    )
    parser.add_argument(
        "--base-dir", "-d",
        default=None,
        help="Directory that fna_path values are relative to (default: directory of this script)",
    )
    args = parser.parse_args()

    # Default output name
    if args.output is None:
        stem, ext = os.path.splitext(args.input)
        args.output = f"{stem}_with_seq{ext}"

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input)

    if "fna_path" not in df.columns or "record_id" not in df.columns:
        raise ValueError("Input CSV must have 'fna_path' and 'record_id' columns.")

    df = extract_nucleotides(df, base_dir=args.base_dir)

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
