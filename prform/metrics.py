"""
Metrics for PRF (Programmed Ribosomal Frameshift) prediction model evaluation.

Adapted from SpliceAI metrics for 3-class classification:
  - Classes: 0 = no PRF, 1 = PRF type -1, 2 = PRF type +1
  - Binary PRF detection metrics (PR-AUC, ROC-AUC, F1) using p(PRF) = p(cls=1) + p(cls=2)
  - Type accuracy: among true PRF positions, fraction where PRF type is correctly predicted

compute_all_metrics expects:
  probs_3:     np.ndarray of shape (N, 3) — softmax probabilities for each class
  targets_cls: np.ndarray of shape (N,)   — class indices (0, 1, or 2)
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)


def topk_accuracy(probs, targets):
    """
    Top-k accuracy for the positive class.

    k = number of true positives in targets.
    Among the top-k predicted positions, what fraction are actual PRF sites?

    Args:
        probs (np.ndarray):   shape (N,) predicted probabilities.
        targets (np.ndarray): shape (N,) binary labels.
    Returns:
        float: top-k accuracy, or NaN if no positives exist.
    """
    k = int(targets.sum())
    if k == 0:
        return float("nan")
    topk_indices = probs.argsort()[::-1][:k]
    correct = targets[topk_indices].sum()
    return float(correct / k)


def pr_auc(probs, targets):
    """
    Area under the Precision-Recall curve.

    Args:
        probs (np.ndarray):   shape (N,) predicted probabilities.
        targets (np.ndarray): shape (N,) binary labels.
    Returns:
        float: PR-AUC score, or NaN if only one class present.
    """
    if len(np.unique(targets)) < 2:
        return float("nan")
    return float(average_precision_score(targets, probs))


def roc_auc(probs, targets):
    """
    Area under the ROC curve.

    Args:
        probs (np.ndarray):   shape (N,) predicted probabilities.
        targets (np.ndarray): shape (N,) binary labels.
    Returns:
        float: ROC-AUC score, or NaN if only one class present.
    """
    if len(np.unique(targets)) < 2:
        return float("nan")
    return float(roc_auc_score(targets, probs))


def best_f1_threshold(probs, targets):
    """
    Find the threshold that maximises F1 on the precision-recall curve.

    Args:
        probs (np.ndarray):   shape (N,) predicted probabilities.
        targets (np.ndarray): shape (N,) binary labels.
    Returns:
        tuple: (best_f1, best_threshold)
    """
    if len(np.unique(targets)) < 2:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = f1_scores.argmax()
    return float(f1_scores[best_idx]), float(thresholds[best_idx])


def classification_metrics_at_threshold(probs, targets, threshold=0.5):
    """
    Compute precision, recall, F1 at a given threshold.

    Args:
        probs (np.ndarray):   shape (N,) predicted probabilities.
        targets (np.ndarray): shape (N,) binary labels.
        threshold (float):    decision threshold.
    Returns:
        dict with keys: precision, recall, f1, threshold
    """
    preds = (probs >= threshold).astype(int)
    return {
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "threshold": threshold,
    }


def argmax_hit_rate(probs, targets, record_ids):
    """
    Per-record argmax hit rate (Hit@1).

    For each record_id that contains at least one positive label, check whether
    the position with the highest predicted probability is actually positive.
    Returns the fraction of such records where this holds.

    Args:
        probs (np.ndarray):      shape (N,) predicted probabilities.
        targets (np.ndarray):    shape (N,) binary labels.
        record_ids (np.ndarray): shape (N,) identifier grouping positions into records.
    Returns:
        float: fraction of positive-containing records where argmax is a positive,
               or NaN if no records with positives exist.
    """
    record_ids = np.asarray(record_ids)
    hits = 0
    total = 0
    for rid in np.unique(record_ids):
        mask = record_ids == rid
        t = targets[mask]
        if t.sum() == 0:
            continue  # skip records with no positive labels
        p = probs[mask]
        if t[p.argmax()] == 1:
            hits += 1
        total += 1
    return float(hits / total) if total > 0 else float("nan")


def argmax3_hit_rate(probs, targets, record_ids):
    """
    Per-record top-3 argmax hit rate (Hit@3).

    For each record with at least one positive label, check whether any of the
    top-3 predicted positions is actually positive.

    Args:
        probs (np.ndarray):      shape (N,) predicted probabilities.
        targets (np.ndarray):    shape (N,) binary labels.
        record_ids (np.ndarray): shape (N,) identifier grouping positions into records.
    Returns:
        float: fraction of positive-containing records where any top-3 position is positive,
               or NaN if no records with positives exist.
    """
    record_ids = np.asarray(record_ids)
    hits = 0
    total = 0
    for rid in np.unique(record_ids):
        mask = record_ids == rid
        t = targets[mask]
        if t.sum() == 0:
            continue
        p = probs[mask]
        top3_indices = p.argsort()[::-1][:3]
        if t[top3_indices].sum() > 0:
            hits += 1
        total += 1
    return float(hits / total) if total > 0 else float("nan")


def argmax_flank_hit_rate(probs, targets, record_ids, flank=3):
    """
    Per-record argmax hit rate with flanking window (Hit@1 with ±flank bp).

    For each record with at least one positive label, take the argmax predicted
    position and expand a window of ±flank bp around it. If any position in that
    window is actually positive, it counts as a hit.

    Args:
        probs (np.ndarray):      shape (N,) predicted probabilities.
        targets (np.ndarray):    shape (N,) binary labels.
        record_ids (np.ndarray): shape (N,) identifier grouping positions into records.
        flank (int):             number of flanking bases on each side (default 3).
    Returns:
        float: fraction of positive-containing records where the argmax window contains
               a positive, or NaN if no records with positives exist.
    """
    record_ids = np.asarray(record_ids)
    hits = 0
    total = 0
    for rid in np.unique(record_ids):
        mask = record_ids == rid
        t = targets[mask]
        if t.sum() == 0:
            continue
        p = probs[mask]
        best = p.argmax()
        lo = max(0, best - flank)
        hi = min(len(t) - 1, best + flank)
        if t[lo : hi + 1].sum() > 0:
            hits += 1
        total += 1
    return float(hits / total) if total > 0 else float("nan")


def type_accuracy(probs_3, targets_cls):
    """
    Type accuracy among true PRF positions.

    Among positions where the true label is PRF (class 1 or 2), what fraction
    has the predicted class (argmax of probs_3) matching the true class?

    Args:
        probs_3 (np.ndarray):     shape (N, 3) softmax probabilities.
        targets_cls (np.ndarray): shape (N,) class indices (0, 1, 2).
    Returns:
        float: type accuracy, or NaN if no true PRF positions exist.
    """
    prf_mask = targets_cls > 0
    if prf_mask.sum() == 0:
        return float("nan")
    pred_cls = probs_3[prf_mask].argmax(axis=1)
    true_cls = targets_cls[prf_mask]
    return float((pred_cls == true_cls).mean())


def compute_type_metrics(probs_3, targets_cls, type_cls):
    """
    Binary metrics for a single PRF type vs all others.

    Uses probs_3[:, type_cls] as the predicted score and
    (targets_cls == type_cls) as the binary label.

    Args:
        probs_3 (np.ndarray):     shape (N, 3) softmax probabilities.
        targets_cls (np.ndarray): shape (N,) class indices (0, 1, 2).
        type_cls (int):           1 for -1 PRF type, 2 for +1 PRF type.
    Returns:
        dict with topk_acc, pr_auc, roc_auc, best_f1, best_f1_threshold,
        metrics_at_0.5, n_positive.
    """
    scores = probs_3[:, type_cls]
    binary_targets = (targets_cls == type_cls).astype(np.int8)
    best_f1_val, best_thresh = best_f1_threshold(scores, binary_targets)
    return {
        "topk_acc": topk_accuracy(scores, binary_targets),
        "pr_auc": pr_auc(scores, binary_targets),
        "roc_auc": roc_auc(scores, binary_targets),
        "best_f1": best_f1_val,
        "best_f1_threshold": best_thresh,
        "metrics_at_0.5": classification_metrics_at_threshold(scores, binary_targets, 0.5),
        "n_positive": int(binary_targets.sum()),
    }


def compute_all_metrics(probs_3, targets_cls):
    """
    Compute all metrics for the 3-class PRF prediction task.

    Binary PRF detection metrics use prf_prob = p(class=1) + p(class=2) as the
    positive score and binary_target = (class > 0) as the label.

    Per-type metrics ('prf_minus1', 'prf_plus1') treat each PRF type as binary
    vs all others, using probs_3[:, c] as the score and (targets_cls == c) as
    the label.

    Args:
        probs_3 (np.ndarray):     shape (N, 3) softmax probabilities [no-PRF, type-1, type+1].
        targets_cls (np.ndarray): shape (N,) class indices (0=no-PRF, 1=type-1, 2=type+1).
    Returns:
        dict with all metric values, including 'prf_minus1' and 'prf_plus1' sub-dicts.
    """
    prf_probs = probs_3[:, 1] + probs_3[:, 2]   # (N,) probability of any PRF
    binary_targets = (targets_cls > 0).astype(np.int8)

    best_f1, best_thresh = best_f1_threshold(prf_probs, binary_targets)
    return {
        "topk_acc": topk_accuracy(prf_probs, binary_targets),
        "pr_auc": pr_auc(prf_probs, binary_targets),
        "roc_auc": roc_auc(prf_probs, binary_targets),
        "best_f1": best_f1,
        "best_f1_threshold": best_thresh,
        "metrics_at_0.5": classification_metrics_at_threshold(prf_probs, binary_targets, 0.5),
        "type_accuracy": type_accuracy(probs_3, targets_cls),
        "prf_minus1": compute_type_metrics(probs_3, targets_cls, type_cls=1),
        "prf_plus1": compute_type_metrics(probs_3, targets_cls, type_cls=2),
        "n_positive": int(binary_targets.sum()),
        "n_total": len(targets_cls),
    }


def to_serializable(obj):
    """
    Convert an object to a JSON-serializable format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj
