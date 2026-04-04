"""
Train PRF (Programmed Ribosomal Frameshift) prediction model.

Adapted from SpliceAI training script. Key changes:
  - Binary classification: BCEWithLogitsLoss instead of CrossEntropyLoss
  - Single output channel instead of 3
  - Positive-weight class balancing instead of multi-class weights
  - Binary metrics (PR-AUC, ROC-AUC, F1) instead of per-class metrics
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
        help="Use positive class weight for loss function",
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
    return parser.parse_args()


def get_dataloader(args):
    """Create data loaders for training and validation datasets."""
    train_dataset = PRFDataset(args.train_data, args.train_fraction, flank=args.flank)
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
        out_channels=1, dropout=args.dropout,
    )


def compute_pos_weight(dataset):
    """
    Compute positive class weight for BCEWithLogitsLoss.

    pos_weight = num_negatives / num_positives
    This balances the loss so the rare positive class contributes equally.

    Returns:
        torch.Tensor: scalar positive weight.
    """
    n_pos = 0
    n_total = 0
    for _, targets, _meta in dataset:
        n_pos += targets.sum().item()
        n_total += targets.numel()
    n_neg = n_total - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    weight = n_neg / n_pos
    return torch.tensor(weight, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True, neg_ratio=10, min_neg_per_batch=500):
    """
    Run a single epoch of training or validation.

    For binary classification:
      - Model outputs shape (B, 1, L) — raw logits
      - Targets shape (B, L) — binary labels
      - Loss: BCEWithLogitsLoss applied to flattened logits and targets

    For metrics, all positive positions are kept and negatives are subsampled
    at neg_ratio * n_pos (floored at min_neg_per_batch) to avoid OOM from
    concatenating the full (B, L) output across all batches.
    """
    if train:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if train else torch.no_grad()
    total_loss = 0.0
    pos_logits_buf, pos_targets_buf = [], []
    neg_logits_buf, neg_targets_buf = [], []

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

            outputs = model(inputs)          # (B, 1, L)
            outputs = outputs.squeeze(1)     # (B, L)

            loss = criterion(outputs, targets)               # (B, L)
            loss = (loss * sample_weights.unsqueeze(1)).mean()
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            flat_logits = outputs.detach().cpu().half().numpy().ravel()
            flat_targets = targets.detach().cpu().numpy().astype(np.int8).ravel()

            pos_mask = flat_targets.astype(bool)
            pos_logits_buf.append(flat_logits[pos_mask])
            pos_targets_buf.append(flat_targets[pos_mask])

            neg_idx = np.where(~pos_mask)[0]
            n_neg_keep = max(min_neg_per_batch, neg_ratio * int(pos_mask.sum()))
            n_neg_keep = min(n_neg_keep, len(neg_idx))
            if n_neg_keep > 0:
                sampled = np.random.choice(neg_idx, n_neg_keep, replace=False)
                neg_logits_buf.append(flat_logits[sampled])
                neg_targets_buf.append(flat_targets[sampled])

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_pos_logits = np.concatenate(pos_logits_buf) if pos_logits_buf else np.array([], dtype=np.float16)
    all_pos_targets = np.concatenate(pos_targets_buf) if pos_targets_buf else np.array([], dtype=np.int8)
    all_neg_logits = np.concatenate(neg_logits_buf) if neg_logits_buf else np.array([], dtype=np.float16)
    all_neg_targets = np.concatenate(neg_targets_buf) if neg_targets_buf else np.array([], dtype=np.int8)

    # Final downsample: keep at most neg_ratio * total positives negatives
    n_pos_total = len(all_pos_logits)
    n_neg_keep_total = min(len(all_neg_logits), neg_ratio * n_pos_total)
    if n_neg_keep_total < len(all_neg_logits):
        idx = np.random.choice(len(all_neg_logits), n_neg_keep_total, replace=False)
        all_neg_logits = all_neg_logits[idx]
        all_neg_targets = all_neg_targets[idx]

    logits = np.concatenate([all_pos_logits, all_neg_logits]).astype(np.float32)
    targets = np.concatenate([all_pos_targets, all_neg_targets])

    # Convert logits to probabilities via sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))

    if len(probs) == 0:
        metrics = {"loss": total_loss / len(loader), "pred_pos": 0, "true_pos": 0,
                   "note": "no samples collected for metrics (zero positives in epoch)"}
        return metrics

    metrics = compute_all_metrics(probs, targets)
    metrics["loss"] = total_loss / len(loader)
    metrics["pred_pos"] = int((probs >= 0.5).sum())
    metrics["true_pos"] = int(targets.sum())

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
    if args.use_class_weights:
        pos_weight = compute_pos_weight(train_dataset).to(device)
        logger.info("Using positive class weight: %.2f", pos_weight.item())
    else:
        pos_weight = None
        logger.info("Not using class weights for loss function")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info("Training with batch size: %d", args.batch_size)
    logger.info("Training with learning rate: %.6f", args.learning_rate)
    logger.info("Training with num epochs: %d", args.num_epochs)
    logger.info("Training with flank size: %d", args.flank)
    logger.info("Training with random seed: %d", args.seed)

    metrics_history = []
    best_val_loss = float("inf")
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    json_path = os.path.join(args.output_dir, "metrics.json")

    epoch_pbar = tqdm(range(args.num_epochs), desc="epochs", unit="epoch", file=sys.stderr)
    for epoch in epoch_pbar:
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
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
        row = {"epoch": epoch + 1}
        for split, m in [("train", train_metrics), ("val", val_metrics)]:
            for key in ["loss", "pr_auc", "roc_auc", "topk_acc", "best_f1", "true_pos", "pred_pos", "n_total"]:
                row[f"{split}_{key}"] = m.get(key, float("nan"))
        pd.DataFrame([row]).to_csv(
            csv_path, mode="w" if epoch == 0 else "a", header=(epoch == 0), index=False
        )

        logger.info("Epoch %d/%d", epoch + 1, args.num_epochs)
        logger.info("  Train — Loss: %.6f  PR-AUC: %.4f  ROC-AUC: %.4f  Top-k: %.4f  Best-F1: %.4f",
                     train_metrics["loss"],
                     train_metrics.get("pr_auc", float("nan")),
                     train_metrics.get("roc_auc", float("nan")),
                     train_metrics.get("topk_acc", float("nan")),
                     train_metrics.get("best_f1", float("nan")))
        logger.info("  Val   — Loss: %.6f  PR-AUC: %.4f  ROC-AUC: %.4f  Top-k: %.4f  Best-F1: %.4f",
                     val_metrics["loss"],
                     val_metrics.get("pr_auc", float("nan")),
                     val_metrics.get("roc_auc", float("nan")),
                     val_metrics.get("topk_acc", float("nan")),
                     val_metrics.get("best_f1", float("nan")))
        logger.info("  Train counts: %d true positives, %d predicted positives out of %d",
                     train_metrics["true_pos"], train_metrics["pred_pos"], train_metrics.get("n_total", 0))
        logger.info("  Val   counts: %d true positives, %d predicted positives out of %d",
                     val_metrics["true_pos"], val_metrics["pred_pos"], val_metrics.get("n_total", 0))
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
