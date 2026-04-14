#!/usr/bin/env python3
"""
six_frame_translate.py
──────────────────────
Translate a nucleotide FASTA file in all six reading frames.

Output: one multi-FASTA with headers indicating sequence ID, frame, and strand.

Usage:
    python six_frame_translate.py -i input.fasta -o translated.fasta [--table 1] [--min-length 10]
"""

import argparse
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq


def six_frame_translate(seq: Seq, table: int = 1):
    """Yield (frame_label, protein_seq) for all 6 reading frames.

    Frames:
        +1, +2, +3  →  forward strand, offset 0/1/2
        -1, -2, -3  →  reverse complement, offset 0/1/2
    """
    rc = seq.reverse_complement()
    for strand_label, strand_seq in [("+", seq), ("-", rc)]:
        for offset in range(3):
            frame_num = offset + 1
            # Trim to a length divisible by 3
            trimmed = strand_seq[offset:]
            remainder = len(trimmed) % 3
            if remainder:
                trimmed = trimmed[:-remainder]
            protein = trimmed.translate(table=table)
            yield f"{strand_label}{frame_num}", str(protein)


def main():
    parser = argparse.ArgumentParser(
        description="Six-frame translation of nucleotide FASTA sequences."
    )
    parser.add_argument("-i", "--input", required=True, help="Input nucleotide FASTA file.")
    parser.add_argument("-o", "--output", required=True, help="Output protein FASTA file.")
    parser.add_argument(
        "--table",
        type=int,
        default=1,
        help="NCBI translation table number (default: 1 = standard).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Minimum amino acid length to include in output (default: 0, keep all).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Error: input file '{args.input}' not found.")

    n_seqs = 0
    n_written = 0

    with open(args.output, "w") as out_fh:
        for record in SeqIO.parse(args.input, "fasta"):
            n_seqs += 1
            for frame_label, protein in six_frame_translate(record.seq, table=args.table):
                if len(protein) < args.min_length:
                    continue
                header = f">{record.id}_frame{frame_label} {record.description}"
                out_fh.write(f"{header}\n{protein}\n")
                n_written += 1

    print(f"Processed {n_seqs} sequence(s); wrote {n_written} translated frame(s) to {args.output}")


if __name__ == "__main__":
    main()
