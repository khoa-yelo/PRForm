"""
Run PRF prediction and output per-nucleotide probabilities with metadata.

Loads a trained checkpoint, runs inference on a dataset, and writes:

  predictions.h5
      Per-block nucleotide-level data:
        - probabilities  (N_blocks, block_len)  float32
        - targets        (N_blocks, block_len)  float32
        - metadata/{accession_id, record_id, cluster_id,
                    species_taxid, genus_taxid, species_name,
                    genus_name, block_idx}

  predictions_per_record.tsv
      One row per unique (accession_id, record_id) with:
        - all metadata columns
        - n_blocks, n_positions (total nucleotides across blocks)
        - max_prob, mean_prob, n_positive_targets, n_predicted_positive
        - per_nucleotide_probs  (comma-separated floats for every position,
                                 blocks concatenated in order)

Usage
-----
# From a pre-processed dataset file (.pkl or .h5):
python predict.py \\
    --checkpoint output/model_final.pth \\
    --data       data/test.h5 \\
    --flank      5000 \\
    --output_dir predictions

# From raw sequences (FASTA or TSV):
python predict.py \\
    --checkpoint  output/model_final.pth \\
    --sequences   seqs.fasta \\
    --flank       5000 \\
    --block_len   15000 \\
    --output_dir  predictions

FASTA format: standard FASTA; sequence ID becomes accession_id/record_id.
  Strand assumed '+'; no PRF labels (targets will be all zeros).

TSV format: tab-separated with at least a 'sequence' column.
  Optional columns: accession_id, record_id, strand, prf_position,
  cluster_id, species_taxid, genus_taxid, species_name, genus_name.
  prf_position may be comma-separated for multiple sites (1-based).
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import h5py
import pandas as pd
from Bio.Seq import Seq
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataloader import PRFDataset, METADATA_STR_KEYS
from model import PRForm_10k, PRForm_2k, PRForm_400nt, PRForm_80nt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PRF prediction and output per-nucleotide probabilities",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth)",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data", type=str,
        help="Path to pre-processed dataset (.pkl or .h5)",
    )
    input_group.add_argument(
        "--sequences", type=str,
        help="Path to raw sequences file (.fasta/.fa or .tsv)",
    )

    parser.add_argument(
        "--flank", type=int, default=5000,
        help="Flank size used during training (determines model architecture)",
    )
    parser.add_argument(
        "--block_len", type=int, default=15000,
        help="Core block length (only used with --sequences). Default: 15000",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--mid_channels", type=int, default=32,
        help="Mid channels (must match the trained model). Default: 32",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (must match the trained model)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="predictions",
        help="Directory for output files",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of dataset to predict on (only used with --data)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sequence processing helpers (mirrors utils/create_datasets.py)
# ---------------------------------------------------------------------------

def _one_hot_encode(seq: str) -> np.ndarray:
    """A=0, C=1, G=2, T/U=3. Ambiguous bases get all-zero rows."""
    nt_map = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    L = len(seq)
    oh = np.zeros((L, 4), dtype=np.uint8)
    for i, base in enumerate(seq):
        idx = nt_map.get(base.upper())
        if idx is not None:
            oh[i, idx] = 1
    return oh


def _reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def _sequence_to_blocks(seq_id, sequence, strand, prf_positions_1b,
                         block_len, flank, extra_meta):
    """
    Convert a single sequence into a list of block dicts using the same
    SpliceAI-style blocking as create_datasets.py.

    Parameters
    ----------
    seq_id : str
    sequence : str
    strand : str  ('+' or '-')
    prf_positions_1b : list[int]  1-based; may be empty
    block_len : int
    flank : int
    extra_meta : dict  additional metadata to attach to every block

    Returns
    -------
    list of dicts compatible with PRFDataset pickle format
    """
    raw_seq = sequence.strip()
    L = len(raw_seq)

    if strand in ("-", "-1"):
        raw_seq = _reverse_complement(raw_seq)
        prf_positions_1b = [L - p + 1 for p in prf_positions_1b]

    prf_0b = [p - 1 for p in prf_positions_1b]

    oh = _one_hot_encode(raw_seq)
    L = oh.shape[0]

    pad = (-L) % block_len
    if pad:
        oh = np.pad(oh, ((0, pad), (0, 0)), constant_values=0)

    oh = np.pad(oh, ((flank, flank), (0, 0)), constant_values=0)
    total = oh.shape[0]

    window = block_len + 2 * flank
    blocks = []
    idx = 0
    blk_idx = 0
    while idx + window <= total:
        block_x = oh[idx: idx + window]

        y = np.zeros((block_len, 1), dtype=np.uint8)
        for pos in prf_0b:
            rel = pos - idx
            if 0 <= rel < block_len:
                y[rel, 0] = 1

        blocks.append({
            "accession_id": seq_id,
            "record_id": extra_meta.get("record_id", seq_id),
            "cluster_id": extra_meta.get("cluster_id", ""),
            "species_taxid": str(extra_meta.get("species_taxid", "")),
            "genus_taxid": str(extra_meta.get("genus_taxid", "")),
            "species_name": str(extra_meta.get("species_name", "")),
            "genus_name": str(extra_meta.get("genus_name", "")),
            "sample_weight": float(extra_meta.get("sample_weight", 1.0)),
            "block_idx": blk_idx,
            "sequence": block_x,
            "y": y,
        })

        idx += block_len
        blk_idx += 1

    return blocks


def _parse_fasta(path):
    """
    Yield (seq_id, sequence) pairs from a FASTA file.
    Does not require Biopython SeqIO — works with plain text parsing.
    """
    seq_id = None
    seq_parts = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq_parts)
                seq_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if seq_id is not None:
        yield seq_id, "".join(seq_parts)


def _load_blocks_from_sequences(path, block_len, flank, logger):
    """
    Parse a FASTA or TSV file and return a list of block dicts.

    FASTA: each record becomes one sequence (strand='+', no PRF labels).
    TSV:   must have a 'sequence' column; other columns are optional metadata.
    """
    ext = os.path.splitext(path)[1].lower()
    is_fasta = ext in (".fasta", ".fa", ".fna")
    is_tsv = ext in (".tsv", ".csv", ".txt")

    # Auto-detect by peeking at first non-empty line if extension is ambiguous
    if not is_fasta and not is_tsv:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    is_fasta = line.startswith(">")
                    is_tsv = not is_fasta
                    break

    blocks = []

    if is_fasta:
        logger.info("Reading FASTA from %s", path)
        for seq_id, sequence in _parse_fasta(path):
            new_blocks = _sequence_to_blocks(
                seq_id=seq_id,
                sequence=sequence,
                strand="+",
                prf_positions_1b=[],
                block_len=block_len,
                flank=flank,
                extra_meta={},
            )
            blocks.extend(new_blocks)
        logger.info("Parsed %d blocks from FASTA", len(blocks))

    else:  # TSV / CSV
        sep = "\t" if ext == ".tsv" else ","
        logger.info("Reading TSV/CSV from %s", path)
        df = pd.read_csv(path, sep=sep, dtype=str)
        if "sequence" not in df.columns:
            raise ValueError(
                f"TSV file must have a 'sequence' column. Found: {list(df.columns)}"
            )
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
            sequence = str(row["sequence"]).strip()
            seq_id = str(row.get("accession_id", i)) if "accession_id" in df.columns else str(i)
            strand = str(row.get("strand", "+")) if "strand" in df.columns else "+"

            prf_positions_1b = []
            if "prf_position" in df.columns:
                raw = str(row["prf_position"]).strip()
                for p in raw.split(","):
                    p = p.strip()
                    if p and p.lower() != "nan":
                        prf_positions_1b.append(int(float(p)))

            extra_meta = {k: row[k] for k in
                          ["record_id", "cluster_id", "species_taxid",
                           "genus_taxid", "species_name", "genus_name",
                           "sample_weight"]
                          if k in df.columns}

            try:
                new_blocks = _sequence_to_blocks(
                    seq_id=seq_id,
                    sequence=sequence,
                    strand=strand,
                    prf_positions_1b=prf_positions_1b,
                    block_len=block_len,
                    flank=flank,
                    extra_meta=extra_meta,
                )
                blocks.extend(new_blocks)
            except Exception as e:
                logger.warning("Error processing row %d (%s): %s", i, seq_id, e)

        logger.info("Parsed %d blocks from TSV", len(blocks))

    return blocks


class _InMemoryDataset(Dataset):
    """Wraps a list of block dicts (same format as PRFDataset pickle mode)."""

    def __init__(self, blocks):
        self.blocks = blocks
        self._in_channels = blocks[0]["sequence"].shape[-1] if blocks else 4

    @property
    def in_channels(self):
        return self._in_channels

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        rec = self.blocks[idx]
        x = np.asarray(rec["sequence"], dtype=np.float32).T  # (C, L)
        y = np.asarray(rec["y"], dtype=np.float32).squeeze(-1)  # (block_len,)
        meta = {k: str(rec.get(k, "")) for k in METADATA_STR_KEYS}
        meta["block_idx"] = int(rec.get("block_idx", 0))
        meta["sample_weight"] = float(rec.get("sample_weight", 1.0))
        return torch.from_numpy(x), torch.from_numpy(y), meta


_MODEL_MAP = {
    5000: PRForm_10k,
    1000: PRForm_2k,
    200: PRForm_400nt,
    40: PRForm_80nt,
}


def _load_model(checkpoint, flank, in_channels, mid_channels, dropout, device):
    if flank not in _MODEL_MAP:
        raise ValueError(
            f"Invalid flank {flank}. Choose from {list(_MODEL_MAP.keys())}"
        )
    model = _MODEL_MAP[flank](
        in_channels=in_channels, mid_channels=mid_channels,
        out_channels=1, dropout=dropout,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _run_inference(model, loader, device, logger):
    """Run forward pass on every batch; return arrays + metadata lists."""
    all_probs, all_targets = [], []
    all_meta = defaultdict(list)

    with torch.no_grad():
        for inputs, targets, meta in tqdm(loader, desc="Inference", unit="batch"):
            inputs = inputs.to(device)
            logits = model(inputs).squeeze(1).cpu().numpy()  # (B, block_len)
            probs = 1.0 / (1.0 + np.exp(-logits))

            all_probs.append(probs)
            all_targets.append(targets.numpy())

            for k, v in meta.items():
                if isinstance(v, torch.Tensor):
                    all_meta[k].extend(v.tolist())
                elif isinstance(v, list):
                    all_meta[k].extend(v)
                else:
                    all_meta[k].append(v)

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return probs, targets, dict(all_meta)


def _save_h5(probs, targets, meta, output_dir, logger):
    """Write per-block probabilities, targets, and metadata to HDF5."""
    N = probs.shape[0]
    h5_path = os.path.join(output_dir, "predictions.h5")

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("probabilities", data=probs, dtype=np.float32)
        hf.create_dataset("targets", data=targets, dtype=np.float32)

        mg = hf.create_group("metadata")
        vlen_str = h5py.special_dtype(vlen=str)
        for k in METADATA_STR_KEYS:
            vals = meta.get(k, [""] * N)
            ds = mg.create_dataset(k, shape=(N,), dtype=vlen_str)
            for i, v in enumerate(vals):
                ds[i] = str(v)

        blk = np.array(meta.get("block_idx", [0] * N), dtype=np.int32)
        mg.create_dataset("block_idx", data=blk)

    logger.info("Per-nucleotide HDF5 saved to %s", h5_path)
    return h5_path


def _save_per_record_tsv(probs, targets, meta, output_dir, logger):
    """
    Group blocks by (accession_id, record_id), reconstruct the full
    per-nucleotide probability vector for each record, and write a TSV
    with one row per record.
    """
    N = probs.shape[0]
    acc_ids = meta.get("accession_id", [""] * N)
    rec_ids = meta.get("record_id", [""] * N)
    blk_idxs = meta.get("block_idx", [0] * N)

    records = defaultdict(lambda: {
        "block_probs": [], "block_targets": [], "block_idxs": [],
    })

    for i in range(N):
        key = (str(acc_ids[i]), str(rec_ids[i]))
        records[key]["block_probs"].append(probs[i])
        records[key]["block_targets"].append(targets[i])
        records[key]["block_idxs"].append(blk_idxs[i])
        if "first_idx" not in records[key]:
            records[key]["first_idx"] = i

    rows = []
    for (acc_id, rec_id), data in records.items():
        order = np.argsort(data["block_idxs"])
        rec_probs = np.concatenate([data["block_probs"][j] for j in order])
        rec_targets = np.concatenate([data["block_targets"][j] for j in order])

        fi = data["first_idx"]
        row = {
            "accession_id": acc_id,
            "record_id": rec_id,
        }
        for k in METADATA_STR_KEYS:
            if k not in row:
                row[k] = str(meta.get(k, [""] * N)[fi])
        row["block_idx_min"] = int(np.min(data["block_idxs"]))
        row["block_idx_max"] = int(np.max(data["block_idxs"]))
        row["n_blocks"] = len(data["block_idxs"])
        row["n_positions"] = len(rec_probs)
        row["max_prob"] = float(rec_probs.max())
        row["mean_prob"] = float(rec_probs.mean())
        row["n_positive_targets"] = int(rec_targets.sum())
        row["n_predicted_positive_0.5"] = int((rec_probs >= 0.5).sum())
        top3_idx = np.argsort(rec_probs)[-3:][::-1]
        for rank, idx in enumerate(top3_idx, start=1):
            row[f"top{rank}_pos"] = int(idx) + 1  # 1-based, consistent with prf_position in input CSV
            row[f"top{rank}_prob"] = float(rec_probs[idx])
        row["per_nucleotide_probs"] = ",".join(
            f"{p:.6f}" for p in rec_probs
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    tsv_path = os.path.join(output_dir, "predictions_per_record.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info(
        "Per-record summary saved to %s  (%d records)", tsv_path, len(df),
    )
    return tsv_path


def predict(args, logger):
    if args.sequences:
        blocks = _load_blocks_from_sequences(
            args.sequences, args.block_len, args.flank, logger,
        )
        if not blocks:
            logger.error("No blocks produced from %s — aborting.", args.sequences)
            sys.exit(1)
        dataset = _InMemoryDataset(blocks)
        logger.info(
            "Dataset: %d blocks from %s (block_len=%d, flank=%d)",
            len(dataset), args.sequences, args.block_len, args.flank,
        )
    else:
        dataset = PRFDataset(args.data, fraction=args.fraction, flank=args.flank)
        logger.info("Dataset: %d blocks from %s", len(dataset), args.data)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = _load_model(
        args.checkpoint, args.flank,
        in_channels=dataset.in_channels,
        mid_channels=args.mid_channels,
        dropout=args.dropout, device=device,
    )
    logger.info(
        "Loaded %s from %s",
        model.__class__.__name__, args.checkpoint,
    )

    logger.info("Running inference ...")
    probs, targets, meta = _run_inference(model, loader, device, logger)
    N, block_len = probs.shape
    logger.info("Done: %d blocks × %d positions per block", N, block_len)

    _save_h5(probs, targets, meta, args.output_dir, logger)
    _save_per_record_tsv(probs, targets, meta, args.output_dir, logger)

    logger.info("All outputs written to %s", args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "predict.log")),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting PRF prediction ...")
    predict(args, logger)
