#!/usr/bin/env python
"""
interpret.py — Calibration and peak-score interpretation for PRF predictions.

Reads predictions.h5 (output of predict.py) and produces:

  calibration_table.tsv
      For each scoring method × threshold: position-level and site-level
      precision, recall, and FDR.  Thresholds are evenly-spaced percentiles
      of the score distribution so all four methods are comparable.

  top_hit_summary.tsv
      For each scoring method: top-hit recall and FDR.
      For each record that has ≥1 true PRF site, the highest-scoring position
      is checked against true sites within ±site_k nt.

  pr_curves.png
      Position-level and site-level PR curves for all four methods side-by-side.

  reliability.png
      Reliability diagram (raw scores only): mean predicted probability vs.
      observed fraction of true positives in equal-frequency score bins.
      Reveals whether the model is well-calibrated.

Scoring methods
---------------
  raw        : model sigmoid output p_i (absolute score)
  zscore     : (p_i - mean_record) / std_record  (relative to genome background)
  fold       : p_i / mean_record                 (fold enrichment over background)
  prominence : p_i - max(p in ±window neighbours) clipped to 0
               (how much the position stands above its local neighbourhood)

Usage
-----
python interpret.py \\
    --predictions  predictions/predictions.h5 \\
    --output_dir   predictions/interpret \\
    [--site_k      3]    \\
    [--prom_window 50]   \\
    [--n_thresholds 100]
"""

import argparse
import logging
import os
import pickle
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_predictions(h5_path):
    """Load predictions.h5 → (probs, targets, valid_mask, meta).

    ``valid_mask`` marks real input positions (``True``) vs right-padding
    added by the block builder (``False``). Older h5 files that predate the
    field fall back to all-valid.
    """
    with h5py.File(h5_path, "r") as hf:
        probs = hf["probabilities"][:].astype(np.float32)   # (N, block_len)
        targets = hf["targets"][:].astype(np.int8)           # (N, block_len)
        if "valid_mask" in hf:
            valid_mask = hf["valid_mask"][:].astype(bool)
        else:
            valid_mask = np.ones_like(targets, dtype=bool)
        meta = {}
        for k in hf["metadata"].keys():
            v = hf["metadata"][k][:]
            if v.dtype.kind in ("O", "S"):
                meta[k] = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
            else:
                meta[k] = v.tolist()
    return probs, targets, valid_mask, meta


def _reconstruct_records(probs, targets, valid_mask, meta):
    """Group blocks by (accession_id, record_id), sort by block_idx, and
    drop right-padding so each record covers only real input positions."""
    N = probs.shape[0]
    acc_ids = meta.get("accession_id", [""] * N)
    rec_ids = meta.get("record_id", [""] * N)
    blk_idxs = np.array(meta.get("block_idx", list(range(N))), dtype=int)

    groups = defaultdict(lambda: {"p": [], "t": [], "v": [], "b": []})
    for i in range(N):
        key = f"{acc_ids[i]}::{rec_ids[i]}"
        groups[key]["p"].append(probs[i])
        groups[key]["t"].append(targets[i])
        groups[key]["v"].append(valid_mask[i])
        groups[key]["b"].append(int(blk_idxs[i]))

    records = []
    for rid, data in groups.items():
        order = np.argsort(data["b"])
        full_p = np.concatenate([data["p"][j] for j in order])
        full_t = np.concatenate([data["t"][j] for j in order]).astype(np.int8)
        full_v = np.concatenate([data["v"][j] for j in order]).astype(bool)
        if not full_v.all():
            full_p = full_p[full_v]
            full_t = full_t[full_v]
        records.append({
            "record_id": rid,
            "probs":   full_p,
            "targets": full_t,
        })
    return records


# ---------------------------------------------------------------------------
# Per-record peak scoring functions
# ---------------------------------------------------------------------------

def _score_zscore(p):
    sigma = p.std()
    return (p - p.mean()) / sigma if sigma > 0 else np.zeros_like(p)


def _score_fold(p, eps=1e-8):
    return p / (p.mean() + eps)


def _score_prominence(p, window):
    """p_i minus the max of its ±window neighbours (excluding i), clipped at 0."""
    L = len(p)
    if L < 3:
        return np.zeros_like(p)
    w = min(window, (L - 1) // 2)
    padded = np.pad(p.astype(np.float32), w, constant_values=-np.inf)
    wins = sliding_window_view(padded, 2 * w + 1)  # (L, 2w+1)
    bg = np.maximum(wins[:, :w].max(axis=1), wins[:, w + 1:].max(axis=1))
    return np.clip(p - bg, 0, None).astype(np.float32)


SCORE_METHODS = {
    "raw":        lambda p, w: p.copy(),
    "zscore":     lambda p, w: _score_zscore(p),
    "fold":       lambda p, w: _score_fold(p),
    "prominence": lambda p, w: _score_prominence(p, w),
}


# ---------------------------------------------------------------------------
# Site-tolerant PR computation
# ---------------------------------------------------------------------------

def _build_site_arrays(records, scores_per_rec, site_k):
    """
    Precompute arrays for efficient site-tolerant PR computation.

    site_labels[i] = 1  if any true positive lies within ±site_k of position i
                        (used to compute site-level precision per threshold)

    coverage_scores    = for each true positive j, the max score in its ±site_k
                        window  (used to compute site-level recall per threshold:
                        true positive j is "covered" at threshold τ iff
                        coverage_score[j] >= τ)
    """
    all_scores, all_targets, all_site_labels = [], [], []
    coverage_scores = []

    for rec, s in zip(records, scores_per_rec):
        t = rec["targets"]
        L = len(t)
        site_lbl = np.zeros(L, dtype=np.int8)
        for tp in np.where(t == 1)[0]:
            lo = max(0, tp - site_k)
            hi = min(L - 1, tp + site_k)
            site_lbl[lo:hi + 1] = 1
            coverage_scores.append(float(s[lo:hi + 1].max()))
        all_scores.append(s)
        all_targets.append(t)
        all_site_labels.append(site_lbl)

    return (
        np.concatenate(all_scores),
        np.concatenate(all_targets),
        np.concatenate(all_site_labels),
        np.array(coverage_scores, dtype=np.float32),
    )


def _compute_pr_data(scores, targets, site_labels, coverage_scores, n_thresholds):
    """
    Evaluate position-level and site-tolerant PR at n_thresholds score
    percentile points.  Returns a dict of arrays indexed by threshold.
    """
    # Thresholds: evenly-spaced percentiles (high → low so PR curve goes left→right)
    qs = np.linspace(0, 100, n_thresholds + 2)[1:-1][::-1]
    thresholds = np.percentile(scores, qs)

    n_true_pos = int(targets.sum())
    n_true_sites = len(coverage_scores)

    pos_prec, pos_rec = [], []
    site_prec, site_rec = [], []

    for tau in thresholds:
        pred_mask = scores >= tau
        n_pred = int(pred_mask.sum())

        if n_pred == 0:
            pos_prec.append(1.0); pos_rec.append(0.0)
            site_prec.append(1.0); site_rec.append(0.0)
            continue

        # position-level
        tp = int(targets[pred_mask].sum())
        pos_prec.append(tp / n_pred)
        pos_rec.append(tp / n_true_pos if n_true_pos > 0 else 0.0)

        # site-level
        site_prec.append(int(site_labels[pred_mask].sum()) / n_pred)
        n_covered = int((coverage_scores >= tau).sum())
        site_rec.append(n_covered / n_true_sites if n_true_sites > 0 else 0.0)

    pos_prec = np.array(pos_prec)
    site_prec = np.array(site_prec)

    return {
        "thresholds":      thresholds,
        "pos_precision":   pos_prec,
        "pos_recall":      np.array(pos_rec),
        "pos_fdr":         1.0 - pos_prec,
        "site_precision":  site_prec,
        "site_recall":     np.array(site_rec),
        "site_fdr":        1.0 - site_prec,
    }


# ---------------------------------------------------------------------------
# Top-hit FDR
# ---------------------------------------------------------------------------

def _has_nearby_positive(t, pos, k):
    lo = max(0, pos - k)
    hi = min(len(t) - 1, pos + k)
    return bool(t[lo:hi + 1].sum() > 0)


def compute_top_hit_fdr(records, site_k, prom_window):
    """
    For each scoring method, for every record with ≥1 true positive site:
      - take the position with the highest score
      - count it a hit if it falls within ±site_k nt of any true positive

    Returns dict: method → {top_hit_recall, top_hit_fdr, n_records, n_hits}
    """
    results = {}
    for method, fn in SCORE_METHODS.items():
        hits = total = 0
        for rec in records:
            t = rec["targets"]
            if t.sum() == 0:
                continue
            s = fn(rec["probs"], prom_window)
            if _has_nearby_positive(t, int(s.argmax()), site_k):
                hits += 1
            total += 1
        recall = hits / total if total > 0 else float("nan")
        results[method] = {
            "top_hit_recall": recall,
            "top_hit_fdr":    1.0 - recall if not np.isnan(recall) else float("nan"),
            "n_records":      total,
            "n_hits":         hits,
        }
    return results


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def compute_reliability(scores_flat, targets_flat, n_bins=10):
    """
    Equal-frequency binning of scores.
    Returns (mean_score_per_bin, fraction_positive_per_bin).
    """
    order = np.argsort(scores_flat)
    s = scores_flat[order]
    t = targets_flat[order]
    mean_scores, frac_pos = [], []
    for chunk in np.array_split(np.arange(len(s)), n_bins):
        if len(chunk) == 0:
            continue
        mean_scores.append(float(s[chunk].mean()))
        frac_pos.append(float(t[chunk].mean()))
    return np.array(mean_scores), np.array(frac_pos)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = {"raw": "#1f77b4", "zscore": "#ff7f0e", "fold": "#2ca02c", "prominence": "#d62728"}
_LABELS = {"raw": "Raw prob", "zscore": "Z-score", "fold": "Fold/mean", "prominence": "Prominence"}


def plot_pr_curves(pr_data_by_method, path, site_k):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, data in pr_data_by_method.items():
        c, lbl = _COLORS[method], _LABELS[method]
        axes[0].plot(data["pos_recall"],  data["pos_precision"],  color=c, label=lbl, lw=1.5)
        axes[1].plot(data["site_recall"], data["site_precision"], color=c, label=lbl, lw=1.5)

    for ax, title in zip(axes, ["Position-level PR", f"Site-level PR (±{site_k} nt)"]):
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision  (= 1 − FDR)")
        ax.set_title(title)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(mean_scores, frac_pos, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_scores, frac_pos, "o-", color="#1f77b4", label="Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (raw)")
    ax.set_ylabel("Fraction of true positives")
    ax.set_title("Reliability diagram (raw scores)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration model fitting
# ---------------------------------------------------------------------------

def fit_calibration_models(pr_data_by_method):
    """
    For each scoring method, fit a monotone PCHIP spline mapping
    score → (site_fdr, pos_fdr) using the empirical PR curve data.

    Monotonicity is enforced first via isotonic regression on precision
    (= 1 − FDR), then PCHIP interpolates smoothly between those points.
    This guarantees a continuous, smooth, monotone function — no discrete
    jumps regardless of the input score.

    Returns dict: method → {
        "site_fdr": PchipInterpolator,
        "pos_fdr":  PchipInterpolator,
        "score_min": float,   # scores below this → fdr_at_min
        "score_max": float,   # scores above this → fdr_at_max
        "site_fdr_at_min": float,
        "site_fdr_at_max": float,
        "pos_fdr_at_min":  float,
        "pos_fdr_at_max":  float,
    }
    """
    models = {}
    for method, data in pr_data_by_method.items():
        # Sort thresholds ascending and remove duplicates
        order = np.argsort(data["thresholds"])
        t_all = data["thresholds"][order]
        _, uniq = np.unique(t_all, return_index=True)
        t = t_all[uniq]

        entry = {"score_min": float(t[0]), "score_max": float(t[-1])}

        for fdr_key in ("site_fdr", "pos_fdr"):
            fdr = data[fdr_key][order][uniq]
            # Enforce monotone decreasing FDR via isotonic regression on precision
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            precision_mono = ir.fit_transform(t, 1.0 - fdr)
            fdr_mono = np.clip(1.0 - precision_mono, 0.0, 1.0)
            entry[fdr_key] = PchipInterpolator(t, fdr_mono, extrapolate=False)
            entry[f"{fdr_key}_at_min"] = float(fdr_mono[0])
            entry[f"{fdr_key}_at_max"] = float(fdr_mono[-1])

        models[method] = entry
    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Calibration and peak-score analysis for PRF predictions")
    p.add_argument("--predictions", required=True,
                   help="Path to predictions.h5 (output of predict.py)")
    p.add_argument("--output_dir", default="interpret",
                   help="Directory for outputs (default: interpret)")
    p.add_argument("--site_k", type=int, default=3,
                   help="Site tolerance ±k nt for site-level metrics (default: 3)")
    p.add_argument("--prom_window", type=int, default=50,
                   help="Neighbourhood window for prominence score (default: 50)")
    p.add_argument("--n_thresholds", type=int, default=100,
                   help="Number of threshold points in PR curves (default: 100)")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "interpret.log")),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- Load ---
    logger.info("Loading %s", args.predictions)
    probs, targets, valid_mask, meta = _load_predictions(args.predictions)
    logger.info("  %d blocks × %d positions per block", *probs.shape)
    n_valid = int(valid_mask.sum())
    logger.info(
        "  %d real positions, %d padded positions",
        n_valid, int(valid_mask.size - n_valid),
    )
    logger.info(
        "  %d true positive positions out of %d real positions",
        int(targets[valid_mask].sum()), n_valid,
    )

    if targets[valid_mask].sum() == 0:
        logger.warning("No positive labels found — PR and FDR metrics are undefined. "
                       "Run on a labelled dataset (e.g. validation set predictions).")
        return

    records = _reconstruct_records(probs, targets, valid_mask, meta)
    n_with_pos = sum(1 for r in records if r["targets"].sum() > 0)
    logger.info("  %d records (%d with ≥1 true positive)", len(records), n_with_pos)

    # --- PR curves and calibration table for each method ---
    pr_data_by_method = {}
    table_rows = []

    for method, fn in SCORE_METHODS.items():
        logger.info("Computing PR data — method: %s", method)
        scores_per_rec = [fn(r["probs"], args.prom_window) for r in records]
        scores_flat, targets_flat, site_labels, coverage_scores = _build_site_arrays(
            records, scores_per_rec, args.site_k
        )
        data = _compute_pr_data(
            scores_flat, targets_flat, site_labels, coverage_scores, args.n_thresholds
        )
        pr_data_by_method[method] = data

        for i, tau in enumerate(data["thresholds"]):
            table_rows.append({
                "method":         method,
                "threshold":      float(tau),
                "pos_precision":  float(data["pos_precision"][i]),
                "pos_recall":     float(data["pos_recall"][i]),
                "pos_fdr":        float(data["pos_fdr"][i]),
                "site_precision": float(data["site_precision"][i]),
                "site_recall":    float(data["site_recall"][i]),
                "site_fdr":       float(data["site_fdr"][i]),
            })

    cal_path = os.path.join(args.output_dir, "calibration_table.tsv")
    pd.DataFrame(table_rows).to_csv(cal_path, sep="\t", index=False, float_format="%.4f")
    logger.info("Calibration table → %s", cal_path)

    # Fit smooth calibration splines and save for use by annotate.py
    logger.info("Fitting calibration splines ...")
    cal_models = fit_calibration_models(pr_data_by_method)
    models_path = os.path.join(args.output_dir, "calibration_models.pkl")
    with open(models_path, "wb") as f:
        pickle.dump({"models": cal_models, "site_k": args.site_k, "prom_window": args.prom_window}, f)
    logger.info("Calibration models → %s", models_path)

    # --- Top-hit FDR ---
    logger.info("Computing top-hit FDR ...")
    top_hit = compute_top_hit_fdr(records, args.site_k, args.prom_window)
    th_path = os.path.join(args.output_dir, "top_hit_summary.tsv")
    pd.DataFrame([{"method": m, **v} for m, v in top_hit.items()]).to_csv(
        th_path, sep="\t", index=False, float_format="%.4f"
    )
    logger.info("Top-hit summary → %s", th_path)
    logger.info("  %-12s  %-12s  %-10s  hits/records", "method", "top_hit_recall", "top_hit_fdr")
    for m, v in top_hit.items():
        logger.info("  %-12s  %-12.3f  %-10.3f  %d/%d",
                    m, v["top_hit_recall"], v["top_hit_fdr"], v["n_hits"], v["n_records"])

    # --- Plots ---
    pr_path = os.path.join(args.output_dir, "pr_curves.png")
    plot_pr_curves(pr_data_by_method, pr_path, args.site_k)
    logger.info("PR curves → %s", pr_path)

    raw_scores_flat = np.concatenate([r["probs"] for r in records])
    raw_targets_flat = np.concatenate([r["targets"] for r in records])
    mean_s, frac_p = compute_reliability(raw_scores_flat, raw_targets_flat)
    rel_path = os.path.join(args.output_dir, "reliability.png")
    plot_reliability(mean_s, frac_p, rel_path)
    logger.info("Reliability diagram → %s", rel_path)

    logger.info("Done. All outputs in %s/", args.output_dir)


if __name__ == "__main__":
    main()
