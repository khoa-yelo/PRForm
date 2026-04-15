"""
extract_protein.py
------------------
Reads a CSV file containing (at minimum) columns:
    accession_id, record_id, prf_position

For each row, automatically locates the GFF and FAA files for the accession
under:
    viral_data_all/ncbi_dataset/data_subset/<accession_id>/
    viral_data_all/ncbi_dataset/data/<accession_id>/

Then looks up which protein CDS covers the nucleotide position (prf_position)
and appends protein_id, protein_name, and protein_sequence columns.

All other columns in the input CSV are preserved as-is.
Rows that fall in intergenic regions get empty protein fields.

Usage
-----
python extract_protein.py --input model_data_full_string2.csv --output output.csv

Output columns
--------------
<all original columns> | protein_id | protein_name | protein_sequence
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

# csv.reader still parses every field per row; raise limit for the large
# sequence column even though we only access a few columns by index.
csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base directory of the project (parent of scripts/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directories to search for accession data (in priority order)
DATA_SEARCH_DIRS = [
    os.path.join(BASE_DIR, "viral_data_all", "ncbi_dataset", "data_subset"),
    os.path.join(BASE_DIR, "viral_data_all", "ncbi_dataset", "data"),
]

# Default column names
COL_ACCESSION = "accession_id"
COL_SITE = "prf_position"

PROTEIN_FIELDS = ["protein_id", "protein_name", "protein_sequence"]


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------

def find_accession_dir(accession_id):
    """
    Find the directory for an accession ID by searching data_subset/ then data/.
    Returns the path if found, or None.
    """
    for search_dir in DATA_SEARCH_DIRS:
        candidate = os.path.join(search_dir, accession_id)
        if os.path.isdir(candidate):
            return candidate
    return None


def find_gff(accession_dir):
    """Find genomic.gff inside an accession directory."""
    path = os.path.join(accession_dir, "genomic.gff")
    return path if os.path.isfile(path) else None


def find_faa(accession_dir):
    """Find protein.faa inside an accession directory."""
    path = os.path.join(accession_dir, "protein.faa")
    return path if os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_gff(gff_path):
    """
    Parse CDS features from a GFF3 file.

    Returns a list of dicts, each with:
        accession_id, protein_id, protein_name, start, end
    Coordinates are 1-based and inclusive (as stored in GFF3).
    """
    cds_records = []
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or "\t" not in line:
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "CDS":
                continue
            accession_id = cols[0]
            start        = int(cols[3])
            end          = int(cols[4])
            attrs        = cols[8]
            protein_id   = _attr(attrs, "protein_id")
            product      = _attr(attrs, "product")
            if protein_id:
                cds_records.append({
                    "accession_id": accession_id,
                    "protein_id":   protein_id,
                    "protein_name": product or "",
                    "start":        start,
                    "end":          end,
                })
    return cds_records


def parse_faa(faa_path):
    """
    Parse a FASTA protein file.

    Returns a dict: protein_id -> sequence string.
    """
    sequences = {}
    current_id = None
    buf = []
    with open(faa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(buf)
                current_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if current_id:
        sequences[current_id] = "".join(buf)
    return sequences


def _attr(attrs_str, key):
    """Extract a single attribute value from a GFF3 attribute string."""
    m = re.search(rf'(?:^|;){re.escape(key)}=([^;]+)', attrs_str)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Lookup index
# ---------------------------------------------------------------------------

def build_index(cds_records):
    """
    Build a per-accession (record_id in GFF = sequence accession) list of
    CDS records for interval lookup.
    """
    index = defaultdict(list)
    for rec in cds_records:
        index[rec["accession_id"]].append(rec)
    return index


def lookup_site(index, faa_sequences, record_id, site):
    """
    Return all CDS records that overlap `site` on `record_id`.
    `site` is 1-based.  Returns a list (may have >1 entry for overlapping genes).

    We use record_id here because the GFF accession_id column actually
    contains the sequence record ID (e.g. the nucleotide accession).
    """
    hits = []
    for rec in index.get(record_id, []):
        if rec["start"] <= site <= rec["end"]:
            hits.append({
                "protein_id":       rec["protein_id"],
                "protein_name":     rec["protein_name"],
                "protein_sequence": faa_sequences.get(rec["protein_id"], ""),
            })
    if not hits:
        hits.append({
            "protein_id":       "",
            "protein_name":     "intergenic",
            "protein_sequence": "",
        })
    return hits


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(input_csv, output_csv):
    """
    Read input CSV, auto-locate GFF/FAA per accession_id, look up protein
    for each row, write enriched CSV.

    Only reads the columns we need (accession_id, record_id, prf_position)
    to avoid loading very large fields like 'sequence'.
    """
    NEEDED_COLS = [COL_ACCESSION, "record_id", COL_SITE]

    # --- Read header and find column indices --------------------------------
    with open(input_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)

    # Validate required columns
    required = {COL_ACCESSION, COL_SITE}
    missing = required - set(header)
    if missing:
        sys.exit(
            f"ERROR: column(s) not found in CSV: {missing}\n"
            f"Available columns: {header}"
        )

    col_idx = {col: header.index(col) for col in NEEDED_COLS if col in header}

    # --- Pre-scan: collect unique accession_ids -----------------------------
    print("Scanning input CSV for unique accession IDs...", file=sys.stderr)

    accession_ids = set()
    with open(input_csv, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for fields in reader:
            accession_ids.add(fields[col_idx[COL_ACCESSION]].strip())

    print(f"Found {len(accession_ids)} unique accession IDs.", file=sys.stderr)

    # --- Load GFF/FAA data per accession -----------------------------------
    combined_index = defaultdict(list)
    combined_faa = {}
    missing_gff = []
    missing_faa = []
    missing_dir = []

    for acc_id in sorted(accession_ids):
        acc_dir = find_accession_dir(acc_id)
        if acc_dir is None:
            missing_dir.append(acc_id)
            continue

        gff_path = find_gff(acc_dir)
        faa_path = find_faa(acc_dir)

        if gff_path is None:
            missing_gff.append(acc_id)
            continue
        if faa_path is None:
            missing_faa.append(acc_id)
            continue

        cds_records = parse_gff(gff_path)
        faa_sequences = parse_faa(faa_path)

        for rec in cds_records:
            combined_index[rec["accession_id"]].append(rec)
        combined_faa.update(faa_sequences)

    # Report missing data
    if missing_dir:
        print(
            f"WARNING: {len(missing_dir)} accession(s) have no directory "
            f"in data_subset/ or data/. First 5: {missing_dir[:5]}",
            file=sys.stderr,
        )
    if missing_gff:
        print(
            f"WARNING: {len(missing_gff)} accession(s) have no genomic.gff. "
            f"First 5: {missing_gff[:5]}",
            file=sys.stderr,
        )
    if missing_faa:
        print(
            f"WARNING: {len(missing_faa)} accession(s) have no protein.faa. "
            f"First 5: {missing_faa[:5]}",
            file=sys.stderr,
        )

    loaded = len(accession_ids) - len(missing_dir) - len(missing_gff) - len(missing_faa)
    print(f"Loaded GFF/FAA data for {loaded} accession(s).", file=sys.stderr)

    # --- Process rows and write output -------------------------------------
    out_fields = NEEDED_COLS + PROTEIN_FIELDS

    with open(input_csv, newline="") as in_fh, \
         open(output_csv, "w", newline="") as out_fh:
        reader = csv.reader(in_fh)
        next(reader)  # skip header
        writer = csv.DictWriter(out_fh, fieldnames=out_fields)
        writer.writeheader()

        row_count = out_count = skipped = 0
        for fields in reader:
            row_count += 1

            accession_id = fields[col_idx[COL_ACCESSION]].strip()
            record_id = fields[col_idx["record_id"]].strip() if "record_id" in col_idx else ""

            # Skip rows with no prf_position value
            raw_site = fields[col_idx[COL_SITE]].strip() if COL_SITE in col_idx else ""
            if not raw_site or raw_site.upper() in ("NA", "NAN", "NONE"):
                skipped += 1
                continue

            try:
                site = int(float(raw_site))
            except (ValueError, TypeError):
                skipped += 1
                continue

            row = {
                COL_ACCESSION: accession_id,
                "record_id": record_id,
                COL_SITE: raw_site,
            }

            for hit in lookup_site(combined_index, combined_faa, record_id, site):
                writer.writerow({**row, **hit})
                out_count += 1

    print(
        f"Done: {row_count} input rows → {out_count} output rows "
        f"({skipped} skipped due to missing prf_position). "
        f"Saved to {output_csv}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Annotate a CSV with protein info based on prf_position. "
                    "Automatically locates GFF/FAA files per accession_id."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file (e.g. model_data_full_string2.csv)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: input file not found: {args.input}")

    process(args.input, args.output)


if __name__ == "__main__":
    main()
