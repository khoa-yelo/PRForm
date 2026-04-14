#!/usr/bin/env python3
"""
orf_finder.py
─────────────
Find all open reading frames (ORFs) in nucleotide FASTA sequences.

Searches all six reading frames for stretches from a start codon (ATG by default)
to the nearest in-frame stop codon (TAA, TAG, TGA).  Also supports an
"any-start" mode that reports every stop-to-stop region (useful for viral /
phage genomes with non-canonical starts).

Outputs:
    • FASTA of ORF nucleotide sequences   (-o / --output-nt)
    • FASTA of translated ORF proteins     (--output-aa)
    • BED6 coordinate file                 (--output-bed)

Usage:
    python orf_finder.py -i genome.fasta -o orfs_nt.fasta --output-aa orfs_aa.fasta \
        --output-bed orfs.bed --min-length 100 --table 1
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq


STOP_CODONS = {"TAA", "TAG", "TGA"}


@dataclass
class ORF:
    seq_id: str
    start: int          # 0-based on the *original* forward strand
    end: int            # 0-based, exclusive
    strand: str         # '+' or '-'
    frame: int          # 1, 2, or 3
    nt_seq: str
    aa_seq: str

    @property
    def length_nt(self) -> int:
        return len(self.nt_seq)

    @property
    def length_aa(self) -> int:
        return len(self.aa_seq)

    @property
    def label(self) -> str:
        return (
            f"{self.seq_id}_ORF_{self.strand}{self.frame}"
            f"_{self.start + 1}_{self.end}"
        )


def find_orfs_in_strand(
    seq_id: str,
    seq_str: str,
    seq_len: int,
    strand: str,
    min_len: int,
    table: int,
    require_start_codon: bool,
    start_codons: set[str],
):
    """Find ORFs in a single strand sequence string.

    Parameters
    ----------
    seq_str : str
        The nucleotide sequence for this strand (already reverse-complemented
        if strand == '-').
    seq_len : int
        Length of the *original* forward sequence (used for coordinate mapping).
    """
    orfs: list[ORF] = []
    for offset in range(3):
        frame_num = offset + 1
        i = offset
        while i + 2 < len(seq_str):
            codon = seq_str[i : i + 3]

            # ── Decide where to begin an ORF ──
            if require_start_codon:
                if codon.upper() not in start_codons:
                    i += 3
                    continue
                orf_start = i
            else:
                # "any-start" mode: begin at current position
                orf_start = i

            # ── Walk forward to the first in-frame stop ──
            j = orf_start
            while j + 2 < len(seq_str):
                c = seq_str[j : j + 3].upper()
                if c in STOP_CODONS:
                    break
                j += 3

            orf_end = min(j + 3, len(seq_str))  # include the stop codon
            nt = seq_str[orf_start:orf_end]

            # Trim to multiple of 3
            trim = len(nt) % 3
            if trim:
                nt = nt[:-trim]
                orf_end -= trim

            if len(nt) >= min_len:
                aa = str(Seq(nt).translate(table=table))
                # Remove trailing '*'
                if aa.endswith("*"):
                    aa = aa[:-1]

                # Map coordinates back to the forward strand
                if strand == "+":
                    fwd_start = orf_start
                    fwd_end = orf_end
                else:
                    fwd_start = seq_len - orf_end
                    fwd_end = seq_len - orf_start

                orfs.append(
                    ORF(
                        seq_id=seq_id,
                        start=fwd_start,
                        end=fwd_end,
                        strand=strand,
                        frame=frame_num,
                        nt_seq=nt,
                        aa_seq=aa,
                    )
                )

            # Advance past this ORF (past the stop codon)
            i = orf_end if j + 3 <= len(seq_str) else orf_end
            if i <= orf_start:
                i = orf_start + 3  # safety: always advance

    return orfs


def find_all_orfs(
    record,
    min_len: int = 75,
    table: int = 1,
    require_start_codon: bool = True,
    start_codons: set[str] | None = None,
) -> list[ORF]:
    if start_codons is None:
        start_codons = {"ATG"}

    fwd = str(record.seq).upper()
    rev = str(record.seq.reverse_complement()).upper()
    seq_len = len(fwd)

    common_kw = dict(
        seq_id=record.id,
        seq_len=seq_len,
        min_len=min_len,
        table=table,
        require_start_codon=require_start_codon,
        start_codons=start_codons,
    )

    orfs = find_orfs_in_strand(seq_str=fwd, strand="+", **common_kw)
    orfs += find_orfs_in_strand(seq_str=rev, strand="-", **common_kw)
    # Sort by start position
    orfs.sort(key=lambda o: (o.start, o.strand))
    return orfs


def main():
    parser = argparse.ArgumentParser(description="Find ORFs in nucleotide FASTA sequences.")
    parser.add_argument("-i", "--input", required=True, help="Input nucleotide FASTA.")
    parser.add_argument("-o", "--output-nt", required=True, help="Output ORF nucleotide FASTA.")
    parser.add_argument("--output-aa", default=None, help="Output ORF protein FASTA.")
    parser.add_argument("--output-bed", default=None, help="Output BED6 coordinate file.")
    parser.add_argument(
        "--min-length",
        type=int,
        default=75,
        help="Minimum ORF nucleotide length (default: 75, i.e. 25 aa).",
    )
    parser.add_argument("--table", type=int, default=1, help="NCBI translation table (default: 1).")
    parser.add_argument(
        "--no-start-codon",
        action="store_true",
        help="Report all stop-to-stop regions (do not require ATG start).",
    )
    parser.add_argument(
        "--start-codons",
        default="ATG",
        help="Comma-separated list of start codons (default: ATG). "
        "Common alternatives: ATG,GTG,TTG",
    )

    args = parser.parse_args()

    if not Path(args.input).is_file():
        sys.exit(f"Error: input file '{args.input}' not found.")

    start_codons = {c.strip().upper() for c in args.start_codons.split(",")}

    out_nt = open(args.output_nt, "w")
    out_aa = open(args.output_aa, "w") if args.output_aa else None
    out_bed = open(args.output_bed, "w") if args.output_bed else None

    total_orfs = 0

    for record in SeqIO.parse(args.input, "fasta"):
        orfs = find_all_orfs(
            record,
            min_len=args.min_length,
            table=args.table,
            require_start_codon=not args.no_start_codon,
            start_codons=start_codons,
        )
        for orf in orfs:
            total_orfs += 1
            out_nt.write(f">{orf.label} len={orf.length_nt}nt\n{orf.nt_seq}\n")
            if out_aa:
                out_aa.write(f">{orf.label} len={orf.length_aa}aa\n{orf.aa_seq}\n")
            if out_bed:
                out_bed.write(
                    f"{orf.seq_id}\t{orf.start}\t{orf.end}\t{orf.label}\t"
                    f"{orf.length_nt}\t{orf.strand}\n"
                )

    out_nt.close()
    if out_aa:
        out_aa.close()
    if out_bed:
        out_bed.close()

    print(f"Found {total_orfs} ORF(s). Written to {args.output_nt}")
    if args.output_aa:
        print(f"  Protein sequences: {args.output_aa}")
    if args.output_bed:
        print(f"  BED coordinates:   {args.output_bed}")


if __name__ == "__main__":
    main()
