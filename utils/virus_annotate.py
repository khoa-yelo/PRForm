#!/usr/bin/env python3
"""
virus_annotate.py
─────────────────
Wrapper for viral genome annotation using Prokka.

Requirements:
    conda install -c conda-forge -c bioconda prokka

Usage examples:
    python virus_annotate.py -i sequences.fasta -o prokka_out/ --threads 4
    python virus_annotate.py -i phage.fasta -o prokka_out/ --threads 8 --prefix my_phage
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# ────────────────────────────────────────────────────────────
#  Prokka
# ────────────────────────────────────────────────────────────
def run_prokka(args):
    """Run Prokka on any viral input (eukaryotic or phage)."""
    exe = shutil.which("prokka")
    if exe is None:
        sys.exit("Error: prokka not found on PATH. Is Prokka installed and env activated?")

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Error: input file '{args.input}' not found.")

    out_dir = Path(args.output)

    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)

    prefix = args.prefix or input_path.stem

    cmd = [
        exe,
        "--outdir", str(out_dir),
        "--prefix", prefix,
        "--cpus", str(args.threads),
        "--kingdom", "Viruses",
    ]

    if args.metagenome:
        cmd.append("--metagenome")

    if args.force:
        cmd.append("--force")

    cmd.append(str(input_path))

    print(f"Running Prokka:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Prokka exited with code {result.returncode}")

    print(f"\nProkka finished.  Results in: {out_dir}/")


# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Virus genome annotation via Prokka (works for eukaryotic viruses and phages)."
    )
    parser.add_argument("-i", "--input", required=True, help="Input nucleotide FASTA.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4).")
    parser.add_argument("--prefix", default=None, help="Output filename prefix (default: input stem).")
    parser.add_argument(
        "--metagenome",
        action="store_true",
        help="Enable metagenome mode (less strict gene filtering).",
    )
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output directory.")

    args = parser.parse_args()
    run_prokka(args)


if __name__ == "__main__":
    main()
