"""
Run PRF prediction and output per-nucleotide probabilities with metadata.

Loads a trained checkpoint, runs inference on a dataset, and writes:

  predictions.h5
      Per-block nucleotide-level data:
        - probabilities  (N_blocks, block_len)      float32  p(PRF) = p(dir-1)+p(dir+1)
        - probs_3        (N_blocks, 3, block_len)   float32  full softmax over classes
        - targets        (N_blocks, block_len)      uint8    binary PRF target
        - target_cls     (N_blocks, block_len)      uint8    class index (0/1/2)
        - valid_mask     (N_blocks, block_len)      uint8    1 = real input,
                                                             0 = right-padding
                                                             added to fill the
                                                             final block
        - metadata/{accession_id, record_id, cluster_id,
                    species_taxid, genus_taxid, species_name,
                    genus_name, block_idx}

  predictions_per_record.tsv
      One row per unique (accession_id, record_id) with:
        - all metadata columns
        - n_blocks, n_positions (total nucleotides across blocks)
        - max_prob, mean_prob, n_positive_targets,
          n_predicted_positive_0.5,
          n_predicted_minus1_0.5, n_predicted_plus1_0.5
        - top{1,2,3}_pos, top{1,2,3}_prob, top{1,2,3}_type,
          top{1,2,3}_prob_minus1, top{1,2,3}_prob_plus1
          (predicted PRF type and per-type probabilities at each top position)
        - per_nucleotide_probs, per_nucleotide_probs_minus1,
          per_nucleotide_probs_plus1
          (comma-separated probs for every position, blocks concatenated
           in order; the latter two give per-type softmax probabilities)

  metrics.csv / metrics.json
      Only written if the input has any positive labels. Uses the same
      positive-heavy subsampling as train.py (keep all positives, sample
      negatives at neg_ratio * n_pos per block, floored at min_neg_per_batch;
      final downsample to neg_ratio * total_positives). Reports PR-AUC /
      ROC-AUC / top-k / best-F1 / type-accuracy / per-type metrics, plus
      per-block argmax_hit_rate, argmax3_hit_rate and argmax_flank_hit_rate.

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
  Optional columns: accession_id, record_id, strand, prf_position, prf_type,
  cluster_id, species_taxid, genus_taxid, species_name, genus_name.
  prf_position may be comma-separated for multiple sites (1-based).
  prf_type may be comma-separated floats aligned with prf_position
  (negative = dir-1, positive/default = dir+1).
"""

import argparse
import json
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
from metrics import (
    compute_all_metrics,
    to_serializable,
    best_f1_threshold,
    classification_metrics_at_threshold,
)
from sklearn.metrics import average_precision_score, roc_auc_score


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
    parser.add_argument(
        "--neg_ratio", type=int, default=10,
        help="Negatives kept per positive when computing metrics (mirrors train.py).",
    )
    parser.add_argument(
        "--min_neg_per_batch", type=int, default=500,
        help="Minimum negatives retained per block before final downsample (mirrors train.py).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for negative subsampling in metrics.",
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
                         prf_types, block_len, flank, extra_meta):
    """
    Convert a single sequence into a list of block dicts using the same
    SpliceAI-style blocking as create_datasets.py.

    Parameters
    ----------
    seq_id : str
    sequence : str
    strand : str  ('+' or '-')
    prf_positions_1b : list[int]  1-based; may be empty
    prf_types : list[float]       aligned with prf_positions_1b; negative values
                                  map to class 1 (dir-1), otherwise class 2
                                  (dir+1). Padded with +1.0 if shorter.
    block_len : int
    flank : int
    extra_meta : dict  additional metadata to attach to every block

    Returns
    -------
    list of dicts compatible with PRFDataset pickle format
    """
    raw_seq = sequence.strip()
    L = len(raw_seq)

    prf_types = list(prf_types)
    while len(prf_types) < len(prf_positions_1b):
        prf_types.append(1.0)

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

        # 3-class one-hot: [no-PRF, dir-1, dir+1] — matches create_datasets.py
        y = np.zeros((block_len, 3), dtype=np.uint8)
        y[:, 0] = 1
        for pos, prf_type in zip(prf_0b, prf_types):
            rel = pos - idx
            if 0 <= rel < block_len:
                y[rel, 0] = 0
                if prf_type < 0:
                    y[rel, 1] = 1
                else:
                    y[rel, 2] = 1

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
                prf_types=[],
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

            prf_types = []
            if "prf_type" in df.columns:
                raw = str(row["prf_type"]).strip()
                for t in raw.split(","):
                    t = t.strip()
                    if t and t.lower() != "nan":
                        prf_types.append(float(t))

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
                    prf_types=prf_types,
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
        y = np.asarray(rec["y"], dtype=np.float32)           # (block_len, 3)
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
        out_channels=3, dropout=dropout,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _run_inference(model, loader, device, logger):
    """
    Run forward pass on every batch; return arrays + metadata lists.

    Returns
    -------
    probs_3    : (N, 3, block_len)  float32 softmax probabilities
    target_cls : (N, block_len)     int8 class indices derived from argmax of one-hot targets
    valid_mask : (N, block_len)     bool — core positions with non-padded input
    meta       : dict of metadata lists
    """
    all_probs_3, all_target_cls, all_valid_mask = [], [], []
    all_meta = defaultdict(list)

    with torch.no_grad():
        for inputs, targets, meta in tqdm(loader, desc="Inference", unit="batch"):
            inputs = inputs.to(device)
            logits = model(inputs)                                     # (B, 3, L)
            probs_3 = torch.softmax(logits, dim=1).cpu().float().numpy()
            # targets is one-hot (B, L, 3); convert to class indices
            target_cls = targets.argmax(dim=2).cpu().numpy().astype(np.int8)

            # Valid mask mirrors train.py: core positions where one-hot input is non-zero.
            block_len = logits.shape[2]
            flank_local = (inputs.shape[2] - block_len) // 2
            core = inputs[:, :, flank_local:flank_local + block_len]
            valid_mask = (core.sum(dim=1) > 0).cpu().numpy()            # (B, L) bool

            all_probs_3.append(probs_3.astype(np.float32))
            all_target_cls.append(target_cls)
            all_valid_mask.append(valid_mask)

            for k, v in meta.items():
                if isinstance(v, torch.Tensor):
                    all_meta[k].extend(v.tolist())
                elif isinstance(v, list):
                    all_meta[k].extend(v)
                else:
                    all_meta[k].append(v)

    probs_3 = np.concatenate(all_probs_3, axis=0)
    target_cls = np.concatenate(all_target_cls, axis=0)
    valid_mask = np.concatenate(all_valid_mask, axis=0)
    return probs_3, target_cls, valid_mask, dict(all_meta)


def _save_h5(probs_3, target_cls, valid_mask, meta, output_dir, logger):
    """Write per-block 3-class probabilities, targets, and metadata to HDF5."""
    N = probs_3.shape[0]
    h5_path = os.path.join(output_dir, "predictions.h5")

    prf_probs = (probs_3[:, 1, :] + probs_3[:, 2, :]).astype(np.float32)
    binary_targets = (target_cls > 0).astype(np.uint8)

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("probabilities", data=prf_probs, dtype=np.float32)
        hf.create_dataset("probs_3", data=probs_3.astype(np.float32))
        hf.create_dataset("targets", data=binary_targets, dtype=np.uint8)
        hf.create_dataset("target_cls", data=target_cls.astype(np.uint8))
        hf.create_dataset("valid_mask", data=valid_mask.astype(np.uint8))

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


def _save_per_record_tsv(probs_3, target_cls, valid_mask, meta, output_dir,
                         logger):
    """
    Group blocks by (accession_id, record_id), reconstruct the full
    per-nucleotide 3-class probability arrays for each record, and write
    a TSV with one row per record.

    PRF probability per position = p(dir-1) + p(dir+1). Predicted PRF type
    is taken from argmax over the three classes (emitted as -1, +1, or 0).

    Right-padding added by the block builder is stripped via ``valid_mask``
    before any summary stat or per-position string is computed, so records
    reflect only real input positions.
    """
    N = probs_3.shape[0]
    acc_ids = meta.get("accession_id", [""] * N)
    rec_ids = meta.get("record_id", [""] * N)
    blk_idxs = meta.get("block_idx", [0] * N)

    records = defaultdict(lambda: {
        "block_probs_3": [], "block_target_cls": [],
        "block_valid": [], "block_idxs": [],
    })

    for i in range(N):
        key = (str(acc_ids[i]), str(rec_ids[i]))
        records[key]["block_probs_3"].append(probs_3[i])        # (3, block_len)
        records[key]["block_target_cls"].append(target_cls[i])  # (block_len,)
        records[key]["block_valid"].append(valid_mask[i])       # (block_len,)
        records[key]["block_idxs"].append(blk_idxs[i])
        if "first_idx" not in records[key]:
            records[key]["first_idx"] = i

    _TYPE_LABEL = {0: 0, 1: -1, 2: 1}

    rows = []
    for (acc_id, rec_id), data in records.items():
        order = np.argsort(data["block_idxs"])
        # Concatenate blocks along position axis → (3, total_len)
        rec_probs_3_full = np.concatenate(
            [data["block_probs_3"][j] for j in order], axis=1,
        )
        rec_target_cls_full = np.concatenate(
            [data["block_target_cls"][j] for j in order],
        )
        rec_valid = np.concatenate(
            [data["block_valid"][j] for j in order],
        ).astype(bool)

        # Drop padded positions. Padding is always right-trailing, so a
        # boolean index preserves order and yields the real sequence.
        rec_probs_3 = rec_probs_3_full[:, rec_valid]
        rec_target_cls = rec_target_cls_full[rec_valid]
        rec_probs = rec_probs_3[1] + rec_probs_3[2]          # (total_len,)
        rec_pred_cls = rec_probs_3.argmax(axis=0)            # (total_len,)
        rec_binary_targets = (rec_target_cls > 0).astype(np.int32)

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
        row["n_positive_targets"] = int(rec_binary_targets.sum())
        row["n_predicted_positive_0.5"] = int((rec_probs >= 0.5).sum())
        row["n_predicted_minus1_0.5"] = int((rec_probs_3[1] >= 0.5).sum())
        row["n_predicted_plus1_0.5"] = int((rec_probs_3[2] >= 0.5).sum())
        top3_idx = np.argsort(rec_probs)[-3:][::-1]
        for rank, idx in enumerate(top3_idx, start=1):
            row[f"top{rank}_pos"] = int(idx) + 1  # 1-based
            row[f"top{rank}_prob"] = float(rec_probs[idx])
            row[f"top{rank}_type"] = int(_TYPE_LABEL[int(rec_pred_cls[idx])])
            row[f"top{rank}_prob_minus1"] = float(rec_probs_3[1, idx])
            row[f"top{rank}_prob_plus1"] = float(rec_probs_3[2, idx])
        row["per_nucleotide_probs"] = ",".join(
            f"{p:.6f}" for p in rec_probs
        )
        row["per_nucleotide_probs_minus1"] = ",".join(
            f"{p:.6f}" for p in rec_probs_3[1]
        )
        row["per_nucleotide_probs_plus1"] = ",".join(
            f"{p:.6f}" for p in rec_probs_3[2]
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    tsv_path = os.path.join(output_dir, "predictions_per_record.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info(
        "Per-record summary saved to %s  (%d records)", tsv_path, len(df),
    )
    return tsv_path


def _save_metrics(probs_3, target_cls, valid_mask, meta, output_dir, logger,
                  neg_ratio=10, min_neg_per_batch=500, seed=42):
    """
    Compute evaluation metrics using the same sampling strategy as train.py.

      - Skip padded positions via ``valid_mask`` (derived from one-hot input).
      - Per block: keep every positive (class 1 or 2); sample negatives at
        ``max(min_neg_per_batch, neg_ratio * n_pos)``.
      - Final downsample: at most ``neg_ratio * total_positives`` negatives.
      - Metrics are then computed with ``compute_all_metrics``.
      - Argmax / Top-3 / Flank hit rates are computed per-block (matching
        train.py's ``run_epoch``), using ``valid_mask`` to ignore padding.
      - Block-level metrics score each block by ``max(prf_probs)`` over
        valid positions, label a block positive iff it contains any true
        site, and report PR-AUC / ROC-AUC / best-F1 + max-prob and
        top1-top2 margin quantiles per class. This is the metric a user
        should consult to set a threshold for rejecting sequences with
        no predicted site.
      - Record-level metrics concatenate all blocks belonging to the same
        ``(accession_id, record_id)`` along the position axis and apply
        the same scoring; this is the full-sequence analogue of the
        block-level metrics and includes record-level argmax hit rates.

    Skips silently if the input has no positive labels (e.g. FASTA-only input).
    """
    N, _, L = probs_3.shape
    if int((target_cls > 0).sum()) == 0:
        logger.info("No positive labels present — skipping metrics.csv / metrics.json")
        return None, None

    rng = np.random.default_rng(seed)
    pos_probs3_buf, pos_cls_buf = [], []
    neg_probs3_buf, neg_cls_buf = [], []
    argmax_hits = argmax3_hits = argmax_flank_hits = argmax_total = 0
    flank_bp = 3

    # Block-level buffers: max PRF probability per block and whether the
    # block contains any true positive. Used to evaluate sequence-level
    # positive/negative discrimination (the end-user workflow of rejecting
    # sequences with no site).
    block_max_prob = np.full(N, np.nan, dtype=np.float32)
    block_top2_margin = np.full(N, np.nan, dtype=np.float32)
    block_is_pos = np.zeros(N, dtype=np.int8)
    block_valid = np.zeros(N, dtype=bool)

    for b in range(N):
        v = valid_mask[b]                        # (L,) bool
        t_cls = target_cls[b]                    # (L,) int8
        p3 = probs_3[b]                          # (3, L) float32
        prf_probs_b = p3[1] + p3[2]              # (L,)
        t_bin = (t_cls > 0)

        # Block-level score: max PRF probability over valid positions.
        if v.any():
            valid_probs = prf_probs_b[v]
            block_valid[b] = True
            block_max_prob[b] = float(valid_probs.max())
            if valid_probs.size >= 2:
                top2 = np.partition(valid_probs, -2)[-2:]
                block_top2_margin[b] = float(top2[1] - top2[0])
            else:
                block_top2_margin[b] = float(valid_probs.max())
            block_is_pos[b] = int((t_bin & v).any())

        # Per-block argmax hit rates (padded positions excluded via -inf).
        if int((t_bin & v).sum()) > 0:
            p = prf_probs_b.copy()
            p[~v] = -np.inf
            best = int(p.argmax())
            argmax_hits += int(t_bin[best])
            top3 = p.argsort()[::-1][:3]
            argmax3_hits += int(t_bin[top3].sum() > 0)
            lo, hi = max(0, best - flank_bp), min(len(t_bin) - 1, best + flank_bp)
            argmax_flank_hits += int(t_bin[lo:hi + 1].sum() > 0)
            argmax_total += 1

        # Per-block sampling for metric buffers (restrict to valid positions).
        flat_p3 = p3.T[v].astype(np.float32)     # (V, 3)
        flat_cls = t_cls[v].astype(np.int8)      # (V,)

        pos_mask = flat_cls > 0
        pos_probs3_buf.append(flat_p3[pos_mask])
        pos_cls_buf.append(flat_cls[pos_mask])

        neg_idx = np.where(~pos_mask)[0]
        n_neg_keep = max(min_neg_per_batch, neg_ratio * int(pos_mask.sum()))
        n_neg_keep = min(n_neg_keep, len(neg_idx))
        if n_neg_keep > 0:
            sampled = rng.choice(neg_idx, n_neg_keep, replace=False)
            neg_probs3_buf.append(flat_p3[sampled])
            neg_cls_buf.append(flat_cls[sampled])

    all_pos_probs3 = (np.concatenate(pos_probs3_buf) if pos_probs3_buf
                     else np.empty((0, 3), dtype=np.float32))
    all_pos_cls = (np.concatenate(pos_cls_buf) if pos_cls_buf
                   else np.empty(0, dtype=np.int8))
    all_neg_probs3 = (np.concatenate(neg_probs3_buf) if neg_probs3_buf
                     else np.empty((0, 3), dtype=np.float32))
    all_neg_cls = (np.concatenate(neg_cls_buf) if neg_cls_buf
                   else np.empty(0, dtype=np.int8))

    n_pos_total = len(all_pos_probs3)
    n_neg_keep_total = min(len(all_neg_probs3), neg_ratio * n_pos_total)
    if n_neg_keep_total < len(all_neg_probs3):
        idx = rng.choice(len(all_neg_probs3), n_neg_keep_total, replace=False)
        all_neg_probs3 = all_neg_probs3[idx]
        all_neg_cls = all_neg_cls[idx]

    probs_3_all = np.concatenate([all_pos_probs3, all_neg_probs3]).astype(np.float32)
    cls_all = np.concatenate([all_pos_cls, all_neg_cls]).astype(np.int64)

    metrics = compute_all_metrics(probs_3_all, cls_all)
    prf_probs_all = probs_3_all[:, 1] + probs_3_all[:, 2]
    metrics["pred_pos"] = int((prf_probs_all >= 0.5).sum())
    metrics["true_pos"] = int((cls_all > 0).sum())
    metrics["argmax_hit_rate"] = (
        float(argmax_hits / argmax_total) if argmax_total else float("nan")
    )
    metrics["argmax3_hit_rate"] = (
        float(argmax3_hits / argmax_total) if argmax_total else float("nan")
    )
    metrics["argmax_flank_hit_rate"] = (
        float(argmax_flank_hits / argmax_total) if argmax_total else float("nan")
    )

    # ------------------------------------------------------------------
    # Block-level evaluation: score each block by max(prf_probs); label
    # positive iff the block contains any true PRF site. Tells users how
    # well they can reject sequences with no site — something the
    # position-level PR-AUC and argmax hit rates don't measure.
    # ------------------------------------------------------------------
    bm_score = block_max_prob[block_valid]
    bm_label = block_is_pos[block_valid].astype(np.int64)
    bm_margin = block_top2_margin[block_valid]
    n_block = int(block_valid.sum())
    n_block_pos = int((bm_label == 1).sum())
    n_block_neg = int((bm_label == 0).sum())

    block_metrics = {
        "n_block": n_block,
        "n_block_pos": n_block_pos,
        "n_block_neg": n_block_neg,
    }
    if n_block_pos > 0 and n_block_neg > 0:
        bf1, bthr = best_f1_threshold(bm_score, bm_label)
        block_metrics.update({
            "pr_auc": float(average_precision_score(bm_label, bm_score)),
            "roc_auc": float(roc_auc_score(bm_label, bm_score)),
            "best_f1": bf1,
            "best_f1_threshold": bthr,
            "metrics_at_0.5": classification_metrics_at_threshold(
                bm_score, bm_label, threshold=0.5
            ),
        })

    def _quantiles(a):
        if a.size == 0:
            return {"median": float("nan"), "p10": float("nan"),
                    "p90": float("nan")}
        q = np.quantile(a, [0.1, 0.5, 0.9])
        return {"p10": float(q[0]), "median": float(q[1]), "p90": float(q[2])}

    block_metrics["max_prob_pos_quantiles"] = _quantiles(bm_score[bm_label == 1])
    block_metrics["max_prob_neg_quantiles"] = _quantiles(bm_score[bm_label == 0])
    block_metrics["top2_margin_pos_quantiles"] = _quantiles(bm_margin[bm_label == 1])
    block_metrics["top2_margin_neg_quantiles"] = _quantiles(bm_margin[bm_label == 0])

    metrics["block_level"] = block_metrics

    # ------------------------------------------------------------------
    # Record-level evaluation: group blocks by (accession_id, record_id),
    # concatenate along the position axis, score each record by
    # ``max(prf_probs)`` over valid positions, and label positive iff any
    # true PRF site exists anywhere in the record. Record is the user-
    # facing sequence unit, so these metrics are what a user should
    # consult to reject full sequences that contain no site.
    # ------------------------------------------------------------------
    acc_ids = meta.get("accession_id", [""] * N)
    rec_ids = meta.get("record_id", [""] * N)
    blk_idxs = meta.get("block_idx", [0] * N)

    rec_groups = defaultdict(list)
    for i in range(N):
        rec_groups[(str(acc_ids[i]), str(rec_ids[i]))].append(i)

    rec_max_prob_list = []
    rec_top2_margin_list = []
    rec_is_pos_list = []
    rec_argmax_hits = rec_argmax3_hits = rec_argmax_flank_hits = rec_argmax_total = 0

    for _, idx_list in rec_groups.items():
        order = sorted(idx_list, key=lambda i: blk_idxs[i])
        prf = np.concatenate(
            [probs_3[i][1] + probs_3[i][2] for i in order]
        )
        v = np.concatenate([valid_mask[i] for i in order])
        t_cls = np.concatenate([target_cls[i] for i in order])
        t_bin = (t_cls > 0)
        if not v.any():
            continue

        valid_probs = prf[v]
        rec_max_prob_list.append(float(valid_probs.max()))
        if valid_probs.size >= 2:
            top2 = np.partition(valid_probs, -2)[-2:]
            rec_top2_margin_list.append(float(top2[1] - top2[0]))
        else:
            rec_top2_margin_list.append(float(valid_probs.max()))
        rec_is_pos_list.append(int((t_bin & v).any()))

        if (t_bin & v).any():
            p = prf.copy()
            p[~v] = -np.inf
            best = int(p.argmax())
            rec_argmax_hits += int(t_bin[best])
            top3 = p.argsort()[::-1][:3]
            rec_argmax3_hits += int(t_bin[top3].sum() > 0)
            lo = max(0, best - flank_bp)
            hi = min(len(t_bin) - 1, best + flank_bp)
            rec_argmax_flank_hits += int(t_bin[lo:hi + 1].sum() > 0)
            rec_argmax_total += 1

    rm_score = np.asarray(rec_max_prob_list, dtype=np.float32)
    rm_label = np.asarray(rec_is_pos_list, dtype=np.int64)
    rm_margin = np.asarray(rec_top2_margin_list, dtype=np.float32)
    n_record = int(rm_score.size)
    n_record_pos = int((rm_label == 1).sum())
    n_record_neg = int((rm_label == 0).sum())

    record_metrics = {
        "n_record": n_record,
        "n_record_pos": n_record_pos,
        "n_record_neg": n_record_neg,
        "argmax_hit_rate": (
            float(rec_argmax_hits / rec_argmax_total)
            if rec_argmax_total else float("nan")
        ),
        "argmax3_hit_rate": (
            float(rec_argmax3_hits / rec_argmax_total)
            if rec_argmax_total else float("nan")
        ),
        "argmax_flank_hit_rate": (
            float(rec_argmax_flank_hits / rec_argmax_total)
            if rec_argmax_total else float("nan")
        ),
    }
    if n_record_pos > 0 and n_record_neg > 0:
        rf1, rthr = best_f1_threshold(rm_score, rm_label)
        record_metrics.update({
            "pr_auc": float(average_precision_score(rm_label, rm_score)),
            "roc_auc": float(roc_auc_score(rm_label, rm_score)),
            "best_f1": rf1,
            "best_f1_threshold": rthr,
            "metrics_at_0.5": classification_metrics_at_threshold(
                rm_score, rm_label, threshold=0.5
            ),
        })

    record_metrics["max_prob_pos_quantiles"] = _quantiles(rm_score[rm_label == 1])
    record_metrics["max_prob_neg_quantiles"] = _quantiles(rm_score[rm_label == 0])
    record_metrics["top2_margin_pos_quantiles"] = _quantiles(rm_margin[rm_label == 1])
    record_metrics["top2_margin_neg_quantiles"] = _quantiles(rm_margin[rm_label == 0])

    metrics["record_level"] = record_metrics

    metrics["neg_ratio"] = neg_ratio
    metrics["min_neg_per_batch"] = min_neg_per_batch

    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(metrics), f, indent=4)

    row = {}
    for key in ["pr_auc", "roc_auc", "topk_acc", "best_f1",
                "argmax_hit_rate", "argmax3_hit_rate", "argmax_flank_hit_rate",
                "type_accuracy", "true_pos", "pred_pos", "n_total"]:
        row[key] = metrics.get(key, float("nan"))
    for prf_type in ["prf_minus1", "prf_plus1"]:
        sub = metrics.get(prf_type, {})
        for key in ["topk_acc", "pr_auc", "roc_auc", "best_f1", "n_positive"]:
            row[f"{prf_type}_{key}"] = sub.get(key, float("nan"))

    bl = metrics["block_level"]
    for key in ["pr_auc", "roc_auc", "best_f1", "best_f1_threshold",
                "n_block", "n_block_pos", "n_block_neg"]:
        row[f"block_{key}"] = bl.get(key, float("nan"))

    rl = metrics["record_level"]
    for key in ["pr_auc", "roc_auc", "best_f1", "best_f1_threshold",
                "argmax_hit_rate", "argmax3_hit_rate", "argmax_flank_hit_rate",
                "n_record", "n_record_pos", "n_record_neg"]:
        row[f"record_{key}"] = rl.get(key, float("nan"))

    csv_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    logger.info(
        "Metrics — PR-AUC: %.4f  ROC-AUC: %.4f  Top-k: %.4f  Best-F1: %.4f  "
        "TypeAcc: %.4f  ArgmaxHit: %.4f  Argmax3Hit: %.4f  ArgmaxFlankHit: %.4f",
        metrics.get("pr_auc", float("nan")), metrics.get("roc_auc", float("nan")),
        metrics.get("topk_acc", float("nan")), metrics.get("best_f1", float("nan")),
        metrics.get("type_accuracy", float("nan")),
        metrics["argmax_hit_rate"], metrics["argmax3_hit_rate"],
        metrics["argmax_flank_hit_rate"],
    )
    logger.info(
        "Block-level — n_block: %d (pos=%d neg=%d)  PR-AUC: %.4f  ROC-AUC: %.4f  "
        "Best-F1: %.4f @ thr=%.4g",
        bl.get("n_block", 0), bl.get("n_block_pos", 0), bl.get("n_block_neg", 0),
        bl.get("pr_auc", float("nan")), bl.get("roc_auc", float("nan")),
        bl.get("best_f1", float("nan")), bl.get("best_f1_threshold", float("nan")),
    )
    logger.info(
        "Record-level — n_record: %d (pos=%d neg=%d)  PR-AUC: %.4f  ROC-AUC: %.4f  "
        "Best-F1: %.4f @ thr=%.4g  ArgmaxHit: %.4f  ArgmaxFlankHit: %.4f",
        rl.get("n_record", 0), rl.get("n_record_pos", 0), rl.get("n_record_neg", 0),
        rl.get("pr_auc", float("nan")), rl.get("roc_auc", float("nan")),
        rl.get("best_f1", float("nan")), rl.get("best_f1_threshold", float("nan")),
        rl.get("argmax_hit_rate", float("nan")),
        rl.get("argmax_flank_hit_rate", float("nan")),
    )
    logger.info("Metrics saved to %s and %s", json_path, csv_path)
    return json_path, csv_path


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
    probs_3, target_cls, valid_mask, meta = _run_inference(
        model, loader, device, logger,
    )
    N, _, block_len = probs_3.shape
    logger.info("Done: %d blocks × %d positions per block (3 classes)", N, block_len)

    _save_h5(probs_3, target_cls, valid_mask, meta, args.output_dir, logger)
    _save_per_record_tsv(
        probs_3, target_cls, valid_mask, meta, args.output_dir, logger,
    )
    _save_metrics(
        probs_3, target_cls, valid_mask, meta, args.output_dir, logger,
        neg_ratio=args.neg_ratio,
        min_neg_per_batch=args.min_neg_per_batch,
        seed=args.seed,
    )

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
