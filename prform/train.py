"""
Train PRF (Programmed Ribosomal Frameshift) prediction model.

Adapted from SpliceAI training script. Key changes:
  - 3-class classification: CrossEntropyLoss (no-PRF, PRF dir-1, PRF dir+1)
  - Three output channels corresponding to the three classes
  - Optional per-class weight balancing
  - Binary PRF detection metrics (PR-AUC, ROC-AUC, F1) + type accuracy
"""

import argparse
import logging
import os
import sys
import json

import pandas as pd

import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from dataloader import PRFDataset
from model import PRForm_10k, PRForm_2k, PRForm_400nt, PRForm_80nt
from metrics import compute_all_metrics, to_serializable
from augment import Augmentor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PRF prediction model")
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation data"
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=1.0,
        help="Fraction of validation data to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--mid_channels", type=int, default=32, help="Mid channels for model. Default: 32"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate for model"
    )
    parser.add_argument(
        "--flank", type=int, default=5000, help="Flank size for training data"
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use positive class weight for loss function (auto-computed as neg/pos ratio)",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Explicit positive class weight for BCEWithLogitsLoss (e.g. 2.0). "
             "Overrides --use_class_weights if both are set.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Number of epochs for linear LR warm-up before cosine decay (0 = no warm-up)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save model checkpoints",
    )
    # --- Augmentation ---
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply on-the-fly sequence augmentation to the training set",
    )
    parser.add_argument(
        "--aug_point_sub_rate",
        type=float,
        default=0.005,
        help="Per-nucleotide mutation rate for point substitution (default: 0.005)",
    )
    parser.add_argument(
        "--aug_p_deletion",
        type=float,
        default=0.3,
        help="Probability of applying deletion augmentation per sample (default: 0.3)",
    )
    parser.add_argument(
        "--aug_p_insertion",
        type=float,
        default=0.2,
        help="Probability of applying insertion augmentation per sample (default: 0.2)",
    )
    parser.add_argument(
        "--aug_p_inversion",
        type=float,
        default=0.2,
        help="Probability of applying inversion augmentation per sample (default: 0.2)",
    )
    parser.add_argument(
        "--aug_p_rc",
        type=float,
        default=0.1,
        help="Probability of reverse-complementing the whole sequence (default: 0.1)",
    )
    return parser.parse_args()


def get_dataloader(args):
    """Create data loaders for training and validation datasets."""
    augmentor = None
    if args.augment:
        augmentor = Augmentor(
            flank=args.flank,
            point_sub_rate=args.aug_point_sub_rate,
            p_deletion=args.aug_p_deletion,
            p_insertion=args.aug_p_insertion,
            p_inversion=args.aug_p_inversion,
            p_rc=args.aug_p_rc,
            seed=args.seed,
        )

    train_dataset = PRFDataset(
        args.train_data, args.train_fraction, flank=args.flank, augmentor=augmentor
    )
    val_dataset = PRFDataset(args.val_data, args.val_fraction, flank=args.flank)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )
    return train_loader, val_loader, train_dataset, val_dataset


def get_model(args, in_channels):
    """Create model based on the specified flank size.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    in_channels : int
        Number of input channels (read from dataset, not hardcoded).
    """
    model_map = {
        5000: PRForm_10k,
        1000: PRForm_2k,
        200: PRForm_400nt,
        40: PRForm_80nt,
    }
    if args.flank not in model_map:
        raise ValueError(f"Invalid flank size {args.flank}. Choose from {list(model_map.keys())}")
    return model_map[args.flank](
        in_channels=in_channels, mid_channels=args.mid_channels,
        out_channels=3, dropout=args.dropout,
    )


def compute_class_weights(dataset):
    """
    Compute per-class weights for CrossEntropyLoss using inverse frequency.

    weights[c] = n_total / (3 * count[c])

    Classes: 0 = no PRF, 1 = PRF dir -1, 2 = PRF dir +1

    Returns:
        torch.Tensor: shape (3,) weight per class.
    """
    counts = np.zeros(3, dtype=np.float64)
    for _, targets, _meta in dataset:
        # targets: (block_len, 3) one-hot tensor
        cls = targets.argmax(dim=1).numpy()  # (block_len,)
        for c in range(3):
            counts[c] += int((cls == c).sum())
    n_total = counts.sum()
    if n_total == 0:
        return torch.ones(3, dtype=torch.float32)
    weights = n_total / (3.0 * counts.clip(min=1))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True, neg_ratio=10, min_neg_per_batch=500, flank=None):
    """
    Run a single epoch of training or validation.

    For 3-class classification:
      - Model outputs shape (B, 3, L) — raw logits (classes: no-PRF, dir-1, dir+1)
      - Targets shape (B, L, 3) — one-hot labels from the dataset
      - Loss: CrossEntropyLoss on class indices derived from argmax(targets)

    For metrics, all positive positions (class 1 or 2) are kept and negatives are
    subsampled at neg_ratio * n_pos (floored at min_neg_per_batch) to avoid OOM.
    Metric buffers store prf_probs = p(cls=1) + p(cls=2) and binary targets.
    """
    if train:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if train else torch.no_grad()
    total_loss = 0.0
    # Store full (N, 3) probs and class indices for all sampled positions
    pos_probs3_buf, pos_cls_buf = [], []
    neg_probs3_buf, neg_cls_buf = [], []
    argmax_hits, argmax_total = 0, 0
    argmax3_hits = 0
    argmax_flank_hits = 0

    assert flank is not None, "flank must be passed explicitly to run_epoch"

    phase = "train" if train else "val"
    pbar = tqdm(loader, desc=phase, leave=False, unit="batch", file=sys.stderr)
    with context:
        for inputs, targets, _meta in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            sample_weights = torch.tensor(
                _meta["sample_weight"], dtype=torch.float32, device=device
            )  # (B,)
            if train:
                optimizer.zero_grad()

            outputs = model(inputs)                          # (B, 3, L)

            # Convert one-hot targets to class indices for CrossEntropyLoss
            target_cls = targets.argmax(dim=2)              # (B, L)

            # Validity mask: core positions where the one-hot input is non-zero.
            # Padded positions (and augmentation-induced zero-padding) are all-zero
            # and are excluded from loss and metrics.
            block_len = outputs.shape[2]
            core = inputs[:, :, flank:flank + block_len]    # (B, C, block_len)
            valid_mask = core.sum(dim=1) > 0                # (B, block_len) bool

            loss = criterion(outputs, target_cls)            # (B, L)
            n_valid = valid_mask.float().sum().clamp(min=1)
            loss = (loss * valid_mask.float() * sample_weights.unsqueeze(1)).sum() / n_valid
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Compute softmax probs and derived quantities
            probs_3 = torch.softmax(outputs.detach(), dim=1).cpu().float().numpy()  # (B, 3, L)
            prf_probs_np = probs_3[:, 1, :] + probs_3[:, 2, :]                     # (B, L)
            target_cls_np = target_cls.detach().cpu().numpy()                       # (B, L)
            binary_targets_np = (target_cls_np > 0)                                # (B, L)
            valid_mask_np = valid_mask.cpu().numpy()                                # (B, L)

            # Argmax hit rate: per-record, check if argmax of prf_probs lands on a PRF site
            for b in range(prf_probs_np.shape[0]):
                t = binary_targets_np[b]
                v = valid_mask_np[b]
                if t.sum() > 0:
                    p = prf_probs_np[b].copy()
                    p[~v] = -np.inf  # ignore padded positions in argmax
                    best = p.argmax()
                    argmax_hits += int(t[best])
                    top3 = p.argsort()[::-1][:3]
                    argmax3_hits += int(t[top3].sum() > 0)
                    lo, hi = max(0, best - 3), min(len(t) - 1, best + 3)
                    argmax_flank_hits += int(t[lo : hi + 1].sum() > 0)
                    argmax_total += 1

            flat_valid = valid_mask_np.ravel()

            # Restrict to valid (non-padded) positions before building metric buffers
            flat_probs_3 = probs_3.transpose(0, 2, 1).reshape(-1, 3)[flat_valid].astype(np.float16)  # (V, 3)
            flat_cls = target_cls_np.ravel()[flat_valid].astype(np.int8)                              # (V,)

            pos_mask = flat_cls > 0
            pos_probs3_buf.append(flat_probs_3[pos_mask])
            pos_cls_buf.append(flat_cls[pos_mask])

            neg_idx = np.where(~pos_mask)[0]
            n_neg_keep = max(min_neg_per_batch, neg_ratio * int(pos_mask.sum()))
            n_neg_keep = min(n_neg_keep, len(neg_idx))
            if n_neg_keep > 0:
                sampled = np.random.choice(neg_idx, n_neg_keep, replace=False)
                neg_probs3_buf.append(flat_probs_3[sampled])
                neg_cls_buf.append(flat_cls[sampled])

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_pos_probs3 = np.concatenate(pos_probs3_buf) if pos_probs3_buf else np.empty((0, 3), dtype=np.float16)
    all_pos_cls = np.concatenate(pos_cls_buf) if pos_cls_buf else np.empty(0, dtype=np.int8)
    all_neg_probs3 = np.concatenate(neg_probs3_buf) if neg_probs3_buf else np.empty((0, 3), dtype=np.float16)
    all_neg_cls = np.concatenate(neg_cls_buf) if neg_cls_buf else np.empty(0, dtype=np.int8)

    # Final downsample: keep at most neg_ratio * total positives negatives
    n_pos_total = len(all_pos_probs3)
    n_neg_keep_total = min(len(all_neg_probs3), neg_ratio * n_pos_total)
    if n_neg_keep_total < len(all_neg_probs3):
        idx = np.random.choice(len(all_neg_probs3), n_neg_keep_total, replace=False)
        all_neg_probs3 = all_neg_probs3[idx]
        all_neg_cls = all_neg_cls[idx]

    probs_3_all = np.concatenate([all_pos_probs3, all_neg_probs3]).astype(np.float32)
    cls_all = np.concatenate([all_pos_cls, all_neg_cls]).astype(np.int64)

    if len(probs_3_all) == 0:
        metrics = {"loss": total_loss / len(loader), "pred_pos": 0, "true_pos": 0,
                   "argmax_hit_rate": float("nan"),
                   "argmax3_hit_rate": float("nan"),
                   "argmax_flank_hit_rate": float("nan"),
                   "note": "no samples collected for metrics (zero positives in epoch)"}
        return metrics

    prf_probs = probs_3_all[:, 1] + probs_3_all[:, 2]
    metrics = compute_all_metrics(probs_3_all, cls_all)
    metrics["loss"] = total_loss / len(loader)
    metrics["pred_pos"] = int((prf_probs >= 0.5).sum())
    metrics["true_pos"] = int((cls_all > 0).sum())
    metrics["argmax_hit_rate"] = float(argmax_hits / argmax_total) if argmax_total > 0 else float("nan")
    metrics["argmax3_hit_rate"] = float(argmax3_hits / argmax_total) if argmax_total > 0 else float("nan")
    metrics["argmax_flank_hit_rate"] = float(argmax_flank_hits / argmax_total) if argmax_total > 0 else float("nan")

    return metrics


def train(args, logger):
    """Train the PRF prediction model."""
    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(args)
    logger.info(
        "Loaded datasets: %d training samples, %d validation samples",
        len(train_dataset),
        len(val_dataset),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = get_model(args, in_channels=train_dataset.in_channels).to(device)
    logger.info("Model architecture: %s", model.__class__.__name__)
    logger.info("Input channels: %d (auto-detected from data)", train_dataset.in_channels)
    logger.info(
        "Model parameters: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Set up loss function
    # For CrossEntropyLoss the weight tensor has shape (3,): [w_no_prf, w_dir-1, w_dir+1]
    if args.pos_weight is not None:
        # --pos_weight scales both PRF classes (1 and 2) relative to background (0)
        class_weight = torch.tensor(
            [1.0, args.pos_weight, args.pos_weight], dtype=torch.float32
        ).to(device)
        logger.info("Using explicit class weights: [1.0, %.2f, %.2f]",
                    args.pos_weight, args.pos_weight)
    elif args.use_class_weights:
        class_weight = compute_class_weights(train_dataset).to(device)
        logger.info("Using auto-computed class weights: %s", class_weight.tolist())
    else:
        class_weight = None
        logger.info("Not using class weights for loss function")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-10)

    # Build LR scheduler: optional linear warm-up then cosine annealing
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8 / args.learning_rate,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.num_epochs - args.warmup_epochs),
            eta_min=0.0,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
        logger.info(
            "LR schedule: linear warm-up for %d epochs, then cosine annealing for %d epochs",
            args.warmup_epochs,
            args.num_epochs - args.warmup_epochs,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=0.0
        )
        logger.info("LR schedule: cosine annealing for %d epochs (no warm-up)", args.num_epochs)

    logger.info("Training with batch size: %d", args.batch_size)
    logger.info("Training with learning rate: %.6f", args.learning_rate)
    logger.info("Training with num epochs: %d", args.num_epochs)
    logger.info("Training with flank size: %d", args.flank)
    logger.info("Training with random seed: %d", args.seed)
    if args.augment:
        logger.info(
            "Augmentation enabled — point_sub_rate=%.4f  p_deletion=%.2f  "
            "p_insertion=%.2f  p_inversion=%.2f  p_rc=%.2f",
            args.aug_point_sub_rate, args.aug_p_deletion,
            args.aug_p_insertion, args.aug_p_inversion, args.aug_p_rc,
        )
    else:
        logger.info("Augmentation disabled")

    metrics_history = []
    best_val_loss = float("inf")
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    json_path = os.path.join(args.output_dir, "metrics.json")

    epoch_pbar = tqdm(range(args.num_epochs), desc="epochs", unit="epoch", file=sys.stderr)
    for epoch in epoch_pbar:
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, flank=args.flank
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False, flank=args.flank
        )
        epoch_pbar.set_postfix(
            train_loss=f"{train_metrics['loss']:.6f}",
            val_loss=f"{val_metrics['loss']:.6f}",
            val_roc_auc=f"{val_metrics.get('roc_auc', float('nan')):.4f}",
            val_pr_auc=f"{val_metrics.get('pr_auc', float('nan')):.4f}",
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train": to_serializable(train_metrics),
            "val": to_serializable(val_metrics),
        }
        metrics_history.append(epoch_record)

        # Overwrite consolidated JSON each epoch
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_history, f, indent=4)

        # Append one row to the CSV (write header only on first epoch)
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]}
        for split, m in [("train", train_metrics), ("val", val_metrics)]:
            for key in ["loss", "pr_auc", "roc_auc", "topk_acc", "best_f1",
                        "argmax_hit_rate", "argmax3_hit_rate", "argmax_flank_hit_rate",
                        "type_accuracy", "true_pos", "pred_pos", "n_total"]:
                row[f"{split}_{key}"] = m.get(key, float("nan"))
            for prf_type in ["prf_minus1", "prf_plus1"]:
                sub = m.get(prf_type, {})
                for key in ["topk_acc", "pr_auc", "roc_auc", "best_f1", "n_positive"]:
                    row[f"{split}_{prf_type}_{key}"] = sub.get(key, float("nan"))
        pd.DataFrame([row]).to_csv(
            csv_path, mode="w" if epoch == 0 else "a", header=(epoch == 0), index=False
        )

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info("Epoch %d/%d  LR: %.8f", epoch + 1, args.num_epochs, current_lr)
        for tag, m in [("Train", train_metrics), ("Val  ", val_metrics)]:
            logger.info(
                "  %s — Loss: %.6f  PR-AUC: %.4f  ROC-AUC: %.4f  Top-k: %.4f  Best-F1: %.4f  "
                "ArgmaxHit: %.4f  Argmax3Hit: %.4f  ArgmaxFlankHit: %.4f  TypeAcc: %.4f",
                tag, m["loss"],
                m.get("pr_auc", float("nan")), m.get("roc_auc", float("nan")),
                m.get("topk_acc", float("nan")), m.get("best_f1", float("nan")),
                m.get("argmax_hit_rate", float("nan")), m.get("argmax3_hit_rate", float("nan")),
                m.get("argmax_flank_hit_rate", float("nan")), m.get("type_accuracy", float("nan")),
            )
            for prf_type, label in [("prf_minus1", "-1 PRF"), ("prf_plus1", "+1 PRF")]:
                sub = m.get(prf_type, {})
                logger.info(
                    "  %s   [%s] Top-k: %.4f  PR-AUC: %.4f  ROC-AUC: %.4f  Best-F1: %.4f  n_pos: %d",
                    tag, label,
                    sub.get("topk_acc", float("nan")), sub.get("pr_auc", float("nan")),
                    sub.get("roc_auc", float("nan")), sub.get("best_f1", float("nan")),
                    sub.get("n_positive", 0),
                )
        for tag, m in [("Train", train_metrics), ("Val  ", val_metrics)]:
            logger.info("  %s   counts: %d true positives, %d predicted positives out of %d",
                        tag, m["true_pos"], m["pred_pos"], m.get("n_total", 0))
        logger.info("-" * 60)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = os.path.join(args.output_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            logger.info("  New best val loss: %.6f — saved to %s", best_val_loss, best_path)

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Model checkpoint saved to %s", checkpoint_path)

        scheduler.step()

    # Save final model
    final_path = os.path.join(args.output_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info("Training completed. Final model saved to %s", final_path)
    logger.info("Metrics saved to %s and %s", json_path, csv_path)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting PRF model training...")
    train(args, logger)
