#!/usr/bin/env python
"""
annotate.py — Annotate new-sequence predictions with smooth, calibrated FDR values.

Workflow
--------
  Step 1 (once, on validation set):
      python predict.py   --data val.h5    --output_dir predictions/val
      python interpret.py --predictions predictions/val/predictions.h5 \\
                          --output_dir predictions/val/interpret
      → produces calibration_models.pkl  (smooth splines: score → FDR)

  Step 2 (for each new batch):
      python predict.py   --sequences new.fasta --output_dir predictions/new
      python annotate.py  --predictions predictions/new/predictions.h5 \\
                          --calibration predictions/val/interpret/calibration_models.pkl \\
                          --output_dir  predictions/new/annotated

FDR lookup
----------
For each position, four scores are computed (raw, zscore, fold, prominence)
and each is fed into its fitted PCHIP spline to get a continuous, smooth FDR
estimate.  The spline was derived from the validation set PR curve with
monotonicity enforced, so the FDR varies smoothly — no discrete jumps.

  site_fdr   — FDR when counting a predicted site as correct if any true PRF
                falls within ±site_k nt  (matches how interpret.py was calibrated)
  pos_fdr    — strict position-exact FDR

Output
------
  annotated_predictions.tsv
      One row per (record, rank).  Top --top_k positions per record, sorted by
      raw probability.  Columns include all four scores and their site/pos FDRs.

Usage
-----
python annotate.py \\
    --predictions  predictions/new/predictions.h5 \\
    --calibration  predictions/val/interpret/calibration_models.pkl \\
    --output_dir   predictions/new/annotated \\
    [--top_k       3]
"""

import argparse
import logging
import os
import pickle
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Scoring functions (must match interpret.py)
# ---------------------------------------------------------------------------

def _score_zscore(p):
    sigma = p.std()
    return (p - p.mean()) / sigma if sigma > 0 else np.zeros_like(p)


def _score_fold(p, eps=1e-8):
    return p / (p.mean() + eps)


def _score_prominence(p, window):
    L = len(p)
    if L < 3:
        return np.zeros_like(p)
    w = min(window, (L - 1) // 2)
    padded = np.pad(p.astype(np.float32), w, constant_values=-np.inf)
    wins = sliding_window_view(padded, 2 * w + 1)
    bg = np.maximum(wins[:, :w].max(axis=1), wins[:, w + 1:].max(axis=1))
    return np.clip(p - bg, 0, None).astype(np.float32)


SCORE_METHODS = {
    "raw":        lambda p, w: p.copy(),
    "zscore":     lambda p, w: _score_zscore(p),
    "fold":       lambda p, w: _score_fold(p),
    "prominence": lambda p, w: _score_prominence(p, w),
}


# ---------------------------------------------------------------------------
# FDR lookup via calibration spline
# ---------------------------------------------------------------------------

def _lookup_fdr(score, model_entry, fdr_key):
    """
    Evaluate the calibrated FDR for a single scalar score.

    Out-of-range scores are clamped to the boundary FDR values rather than
    extrapolated, which is the conservative choice:
      - score below calibration range → highest FDR seen (most uncertain)
      - score above calibration range → lowest FDR seen  (most confident)
    """
    if score <= model_entry["score_min"]:
        return model_entry[f"{fdr_key}_at_min"]
    if score >= model_entry["score_max"]:
        return model_entry[f"{fdr_key}_at_max"]
    return float(np.clip(model_entry[fdr_key](score), 0.0, 1.0))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_predictions(h5_path):
    with h5py.File(h5_path, "r") as hf:
        probs = hf["probabilities"][:].astype(np.float32)
        targets = hf["targets"][:].astype(np.int8) if "targets" in hf else None
        meta = {}
        for k in hf["metadata"].keys():
            v = hf["metadata"][k][:]
            if v.dtype.kind in ("O", "S"):
                meta[k] = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
            else:
                meta[k] = v.tolist()
    return probs, targets, meta


def _reconstruct_records(probs, targets, meta):
    N = probs.shape[0]
    acc_ids  = meta.get("accession_id", [""] * N)
    rec_ids  = meta.get("record_id",    [""] * N)
    blk_idxs = np.array(meta.get("block_idx", list(range(N))), dtype=int)

    EXTRA_KEYS = ("species_name", "genus_name", "cluster_id",
                  "species_taxid", "genus_taxid")

    groups = defaultdict(lambda: {"p": [], "t": [], "b": [], "first": None})
    for i in range(N):
        key = f"{acc_ids[i]}::{rec_ids[i]}"
        groups[key]["p"].append(probs[i])
        groups[key]["t"].append(
            targets[i] if targets is not None
            else np.zeros(probs.shape[1], dtype=np.int8)
        )
        groups[key]["b"].append(int(blk_idxs[i]))
        if groups[key]["first"] is None:
            groups[key]["first"] = i

    records = []
    for key, data in groups.items():
        acc, rec = key.split("::", 1)
        order = np.argsort(data["b"])
        fi = data["first"]
        extra = {k: meta[k][fi] for k in EXTRA_KEYS if k in meta}
        records.append({
            "accession_id": acc,
            "record_id":    rec,
            "probs":   np.concatenate([data["p"][j] for j in order]),
            "targets": np.concatenate([data["t"][j] for j in order]).astype(np.int8),
            **extra,
        })
    return records


# ---------------------------------------------------------------------------
# Annotate one record
# ---------------------------------------------------------------------------

def _annotate_record(rec, cal_models, top_k, prom_window):
    """
    Return a list of top_k rows for this record, each with scores and
    smooth calibrated FDR values from the fitted splines.
    """
    p = rec["probs"]
    scores = {m: fn(p, prom_window) for m, fn in SCORE_METHODS.items()}

    # Rank positions by raw probability
    top_idx = np.argsort(p)[::-1][:top_k]

    rows = []
    for rank, pos in enumerate(top_idx, start=1):
        row = {
            "accession_id":    rec["accession_id"],
            "record_id":       rec["record_id"],
            "rank":            rank,
            "position_1based": int(pos) + 1,
        }
        for k in ("species_name", "genus_name", "cluster_id"):
            if k in rec:
                row[k] = rec[k]

        for method in SCORE_METHODS:
            s = float(scores[method][pos])
            row[f"{method}_score"] = s
            if method in cal_models:
                m = cal_models[method]
                row[f"{method}_site_fdr"] = _lookup_fdr(s, m, "site_fdr")
                row[f"{method}_pos_fdr"]  = _lookup_fdr(s, m, "pos_fdr")

        # Include true label if the dataset was labelled
        if rec["targets"].sum() > 0:
            row["true_label"] = int(rec["targets"][pos])

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Annotate new-sequence predictions with smooth calibrated FDR"
    )
    p.add_argument("--predictions", required=True,
                   help="predictions.h5 produced by predict.py")
    p.add_argument("--calibration", required=True,
                   help="calibration_models.pkl produced by interpret.py")
    p.add_argument("--output_dir", default="annotated",
                   help="Output directory (default: annotated)")
    p.add_argument("--top_k", type=int, default=3,
                   help="Top-k positions to report per record (default: 3)")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "annotate.log")),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- Load calibration ---
    logger.info("Loading calibration models: %s", args.calibration)
    with open(args.calibration, "rb") as f:
        cal_bundle = pickle.load(f)
    cal_models  = cal_bundle["models"]
    prom_window = cal_bundle["prom_window"]
    site_k      = cal_bundle["site_k"]
    logger.info("  Calibrated methods : %s", list(cal_models.keys()))
    logger.info("  Calibrated site_k  : %d nt", site_k)
    logger.info("  Prominence window  : %d", prom_window)

    # --- Load predictions ---
    logger.info("Loading predictions: %s", args.predictions)
    probs, targets, meta = _load_predictions(args.predictions)
    logger.info("  %d blocks × %d positions", *probs.shape)

    records = _reconstruct_records(probs, targets, meta)
    logger.info("  %d records", len(records))

    # --- Annotate ---
    all_rows = []
    for rec in records:
        all_rows.extend(_annotate_record(rec, cal_models, args.top_k, prom_window))

    # --- Save ---
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.output_dir, "annotated_predictions.tsv")
    df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")
    logger.info("Annotated predictions → %s  (%d records, top-%d each)",
                out_path, len(records), args.top_k)
    logger.info("Done.")


if __name__ == "__main__":
    main()
