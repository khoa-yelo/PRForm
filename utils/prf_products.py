#!/usr/bin/env python3
"""
prf_products.py
───────────────
Programmed Ribosomal Frameshift (PRF) product nomination.

Given:
  • a nucleotide sequence (FASTA)
  • annotated gene/ORF coordinates (GFF3, BED, or simple TSV)
  • a PRF slippery-site position (1-based coordinate on the + strand)

This script generates the potential frameshifted fusion protein products for
each shift case: -1, +1, -2, +2.

Background
──────────
In a -1 PRF, the ribosome slips back 1 nt at the slippery site and then
continues translating in the new reading frame. The resulting protein is a
fusion of the 0-frame N-terminal portion (up to the slip site) with the new
reading frame C-terminal portion (from the slip site onward).

Similarly for +1 (ribosome advances 1 nt), -2 (back 2 nt), +2 (forward 2 nt).

Outputs
───────
  • FASTA of predicted fusion protein products
  • TSV summary table

Usage:
    python prf_products.py \
        -i genome.fasta \
        --gff annotations.gff3 \
        --prf-site 13468 \
        -o prf_products.fasta \
        --table 1

    # Or with simple TSV of ORFs (columns: name, start, end, strand):
    python prf_products.py \
        -i genome.fasta \
        --orfs orfs.tsv \
        --prf-site 13468 \
        -o prf_products.fasta
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq


@dataclass
class Gene:
    name: str
    start: int   # 1-based inclusive
    end: int     # 1-based inclusive
    strand: str  # '+' or '-'


SHIFT_OFFSETS = {
    "-1": -1,
    "+1": +1,
    "-2": -2,
    "+2": +2,
}


def parse_gff_genes(path: str) -> list[Gene]:
    """Parse CDS / gene features from a GFF3 file."""
    genes = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            feat_type = parts[2]
            if feat_type not in ("CDS", "gene", "ORF"):
                continue
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            # Extract Name or ID from attributes
            attrs = parts[8]
            name = "unknown"
            for attr in attrs.split(";"):
                if attr.startswith("Name=") or attr.startswith("ID="):
                    name = attr.split("=", 1)[1]
                    break
            genes.append(Gene(name=name, start=start, end=end, strand=strand))
    return genes


def parse_bed_genes(path: str) -> list[Gene]:
    """Parse genes from a BED file (0-based half-open)."""
    genes = []
    with open(path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            name = row[3] if len(row) > 3 else f"orf_{row[1]}_{row[2]}"
            strand = row[5] if len(row) > 5 else "+"
            # BED is 0-based half-open → convert to 1-based inclusive
            genes.append(Gene(name=name, start=int(row[1]) + 1, end=int(row[2]), strand=strand))
    return genes


def parse_tsv_genes(path: str) -> list[Gene]:
    """Parse genes from a simple TSV: name  start  end  strand."""
    genes = []
    with open(path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            genes.append(Gene(name=row[0], start=int(row[1]), end=int(row[2]), strand=row[3].strip()))
    return genes


def translate_to_first_stop(nt_seq: str, table: int) -> str:
    """Translate nt_seq codon-by-codon, stopping at the first stop codon."""
    protein = []
    for i in range(0, len(nt_seq) - 2, 3):
        codon = nt_seq[i : i + 3]
        aa = str(Seq(codon).translate(table=table))
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def generate_prf_products(
    genome_seq: str,
    genes: list[Gene],
    prf_site: int,
    table: int = 1,
):
    """
    For each gene that spans the PRF site and each frameshift offset,
    produce the predicted fusion protein.

    Parameters
    ----------
    genome_seq : str  (1-indexed via slicing with -1 offset)
    genes : list of Gene (1-based inclusive coords, + strand only for now)
    prf_site : int  (1-based position on + strand where the slip occurs)
    table : int  (NCBI translation table)

    Yields (shift_label, gene_name, n_term, c_term, fusion_protein, details_dict)
    """
    seq = genome_seq.upper()
    seq_len = len(seq)

    for gene in genes:
        if gene.strand == "-":
            # For - strand genes containing the PRF site, we would need to
            # map the site to the reverse-complement frame. For simplicity
            # we skip unless the site falls within the gene span.
            if not (gene.start <= prf_site <= gene.end):
                continue

        if gene.strand == "+":
            if not (gene.start <= prf_site <= gene.end):
                continue

        # ── 0-frame N-terminal portion (gene start → PRF site) ──
        if gene.strand == "+":
            gene_start_0 = gene.start - 1  # 0-based
            slip_0 = prf_site - 1          # 0-based

            # Nucleotides from gene start to the slip site (inclusive of slip codon)
            # We align to the gene's reading frame
            n_term_nt = seq[gene_start_0:slip_0]
            # Trim to codon boundary
            trim = len(n_term_nt) % 3
            if trim:
                n_term_nt = n_term_nt[:-trim]
            n_term_aa = str(Seq(n_term_nt).translate(table=table)).rstrip("*")
        else:
            # Minus strand: reverse-complement the gene region
            rc_full = str(Seq(seq).reverse_complement())
            # Map positions to RC coordinates
            rc_gene_start = seq_len - gene.end
            rc_gene_end = seq_len - gene.start + 1
            rc_slip = seq_len - prf_site

            n_term_nt = rc_full[rc_gene_start:rc_slip]
            trim = len(n_term_nt) % 3
            if trim:
                n_term_nt = n_term_nt[:-trim]
            n_term_aa = str(Seq(n_term_nt).translate(table=table)).rstrip("*")
            slip_0 = prf_site - 1  # still needed below for + strand calc

        for shift_label, offset in SHIFT_OFFSETS.items():
            if gene.strand == "+":
                # New reading position after the slip
                new_pos = slip_0 + offset  # 0-based
                if new_pos < 0 or new_pos >= seq_len:
                    continue

                # Translate from new_pos to the first in-frame stop codon
                downstream_nt = seq[new_pos:]
                c_term_aa = translate_to_first_stop(downstream_nt, table)
            else:
                rc_full = str(Seq(seq).reverse_complement())
                new_pos = (seq_len - prf_site) + offset
                if new_pos < 0 or new_pos >= seq_len:
                    continue
                downstream_nt = rc_full[new_pos:]
                c_term_aa = translate_to_first_stop(downstream_nt, table)

            fusion = n_term_aa + c_term_aa

            details = {
                "gene": gene.name,
                "gene_start": gene.start,
                "gene_end": gene.end,
                "strand": gene.strand,
                "prf_site": prf_site,
                "shift": shift_label,
                "n_term_len_aa": len(n_term_aa),
                "c_term_len_aa": len(c_term_aa),
                "fusion_len_aa": len(fusion),
            }

            yield shift_label, gene.name, n_term_aa, c_term_aa, fusion, details


def main():
    parser = argparse.ArgumentParser(
        description="Generate PRF (programmed ribosomal frameshift) fusion protein products."
    )
    parser.add_argument("-i", "--input", required=True, help="Input genome FASTA (single sequence).")
    parser.add_argument(
        "--prf-site",
        type=int,
        required=True,
        help="1-based position of the slippery site on the + strand.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output FASTA of fusion proteins.")
    parser.add_argument("--output-tsv", default=None, help="Output TSV summary table.")
    parser.add_argument("--table", type=int, default=1, help="NCBI translation table (default: 1).")

    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument("--gff", help="GFF3 file with CDS/gene annotations.")
    gene_group.add_argument("--bed", help="BED file with ORF coordinates.")
    gene_group.add_argument(
        "--orfs",
        help="Simple TSV file: name<TAB>start<TAB>end<TAB>strand  (1-based inclusive).",
    )
    parser.add_argument(
        "--shifts",
        default="-1,+1,-2,+2",
        help="Comma-separated list of frame shifts to compute (default: -1,+1,-2,+2).",
    )

    args = parser.parse_args()

    # ── Load genome ──
    records = list(SeqIO.parse(args.input, "fasta"))
    if not records:
        sys.exit("Error: no sequences found in input FASTA.")
    if len(records) > 1:
        print(f"Warning: multiple sequences in FASTA; using the first ({records[0].id}).")
    genome_seq = str(records[0].seq)

    # ── Load genes ──
    if args.gff:
        genes = parse_gff_genes(args.gff)
    elif args.bed:
        genes = parse_bed_genes(args.bed)
    else:
        genes = parse_tsv_genes(args.orfs)

    if not genes:
        sys.exit("Error: no genes/ORFs loaded from annotation file.")

    # Filter requested shifts
    requested = {s.strip() for s in args.shifts.split(",")}
    global SHIFT_OFFSETS
    SHIFT_OFFSETS = {k: v for k, v in SHIFT_OFFSETS.items() if k in requested}

    # ── Generate products ──
    out_fa = open(args.output, "w")
    out_tsv = open(args.output_tsv, "w") if args.output_tsv else None
    if out_tsv:
        out_tsv.write(
            "gene\tgene_start\tgene_end\tstrand\tprf_site\tshift\t"
            "n_term_len_aa\tc_term_len_aa\tfusion_len_aa\n"
        )

    count = 0
    for shift_label, gene_name, n_term, c_term, fusion, details in generate_prf_products(
        genome_seq, genes, args.prf_site, table=args.table
    ):
        count += 1
        header = (
            f">{gene_name}_PRF{shift_label} "
            f"site={args.prf_site} nterm={details['n_term_len_aa']}aa "
            f"cterm={details['c_term_len_aa']}aa total={details['fusion_len_aa']}aa"
        )
        out_fa.write(f"{header}\n{fusion}\n")

        if out_tsv:
            vals = [str(details[k]) for k in [
                "gene", "gene_start", "gene_end", "strand", "prf_site",
                "shift", "n_term_len_aa", "c_term_len_aa", "fusion_len_aa",
            ]]
            out_tsv.write("\t".join(vals) + "\n")

    out_fa.close()
    if out_tsv:
        out_tsv.close()

    print(f"Generated {count} PRF fusion product(s).")
    print(f"  Protein FASTA: {args.output}")
    if args.output_tsv:
        print(f"  Summary TSV:   {args.output_tsv}")

    if count == 0:
        print(
            "\nNote: No products generated. Verify that the PRF site position "
            "falls within at least one annotated gene/ORF."
        )


if __name__ == "__main__":
    main()
