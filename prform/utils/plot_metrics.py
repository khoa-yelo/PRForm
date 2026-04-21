"""
Visualize training metrics from metrics.csv output files.

Handles the 3-class PRF training CSV written by train.py, with columns for:
  - binary PRF detection (loss, pr_auc, roc_auc, topk_acc, best_f1,
    argmax/argmax3/argmax_flank hit rates, type_accuracy)
  - per-class metrics (prf_minus1_*, prf_plus1_*)
  - learning rate schedule (lr)

Usage:
    python plot_metrics.py <metrics.csv> [<metrics2.csv> ...] [--labels l1 l2] [--output fig.png]

When --output is given and multiple figure panels are produced (overall, per-class,
and LR), the extra panels are written with suffixes next to the base name, e.g.
``fig.png``, ``fig_prf_minus1.png``, ``fig_prf_plus1.png``, ``fig_lr.png``.

Examples:
    python plot_metrics.py ../output_tiny/metrics.csv
    python plot_metrics.py ../output_tiny/metrics.csv ../output_80/metrics.csv --labels tiny full
    python plot_metrics.py ../output_tiny/metrics.csv --output training_curves.png
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Pairs for side-by-side train vs val plots (binary PRF detection).
METRIC_PAIRS = [
    ("train_loss",                  "val_loss",                  "Loss"),
    ("train_pr_auc",                "val_pr_auc",                "PR-AUC"),
    ("train_roc_auc",               "val_roc_auc",               "ROC-AUC"),
    ("train_topk_acc",              "val_topk_acc",              "Top-k Accuracy"),
    ("train_best_f1",               "val_best_f1",               "Best F1"),
    ("train_argmax_hit_rate",       "val_argmax_hit_rate",       "Argmax Hit Rate (Hit@1)"),
    ("train_argmax3_hit_rate",      "val_argmax3_hit_rate",      "Argmax Top-3 Hit Rate (Hit@3)"),
    ("train_argmax_flank_hit_rate", "val_argmax_flank_hit_rate", "Argmax Flank Hit Rate (±3 bp)"),
    ("train_type_accuracy",         "val_type_accuracy",         "Type Accuracy (−1 vs +1 PRF)"),
]

# Per-class (−1 PRF and +1 PRF) train vs val pairs.
PER_CLASS_PAIRS = [
    ("pr_auc",   "PR-AUC"),
    ("roc_auc",  "ROC-AUC"),
    ("topk_acc", "Top-k Accuracy"),
    ("best_f1",  "Best F1"),
]

PRF_CLASSES = [
    ("prf_minus1", "−1 PRF"),
    ("prf_plus1",  "+1 PRF"),
]

# Summary columns (overall binary PRF detection).
SUMMARY_COLS = [
    "train_loss", "train_pr_auc", "train_roc_auc", "train_topk_acc", "train_best_f1",
    "train_argmax_hit_rate", "train_argmax3_hit_rate", "train_argmax_flank_hit_rate",
    "train_type_accuracy", "train_true_pos", "train_pred_pos", "train_n_total",
    "val_loss", "val_pr_auc", "val_roc_auc", "val_topk_acc", "val_best_f1",
    "val_argmax_hit_rate", "val_argmax3_hit_rate", "val_argmax_flank_hit_rate",
    "val_type_accuracy", "val_true_pos", "val_pred_pos", "val_n_total",
]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"No 'epoch' column in {path}")
    return df


def _plot_pair_grid(pairs, dfs, labels, title, skip_epochs):
    """Plot a grid of train vs val metric pairs; returns the figure (or None if empty)."""
    rows = [
        (t, v, lbl) for (t, v, lbl) in pairs
        if any(t in df.columns or v in df.columns for df in dfs)
    ]
    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 4 * len(rows)), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row_idx, (train_col, val_col, row_title) in enumerate(rows):
        ax_train, ax_val = axes[row_idx]
        for i, (df, label) in enumerate(zip(dfs, labels)):
            color = colors[i % len(colors)]
            df = df.iloc[skip_epochs:]
            epochs = df["epoch"]
            if train_col in df.columns:
                ax_train.plot(epochs, df[train_col], color=color, label=label, marker="o", markersize=3)
            if val_col in df.columns:
                ax_val.plot(epochs, df[val_col], color=color, label=label, marker="o", markersize=3)
        for ax, split in [(ax_train, "Train"), (ax_val, "Val")]:
            ax.set_title(f"{split} — {row_title}")
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            if len(dfs) > 1:
                ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def _plot_lr(dfs, labels, skip_epochs):
    if not any("lr" in df.columns for df in dfs):
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if "lr" not in df.columns:
            continue
        df = df.iloc[skip_epochs:]
        ax.plot(df["epoch"], df["lr"], color=colors[i % len(colors)],
                label=label, marker="o", markersize=3)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    if len(dfs) > 1:
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def _save_or_show(fig, output_path):
    if output_path is None:
        plt.show()
        return
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def plot_metrics(dfs: list[pd.DataFrame], labels: list[str], output: str | None, skip_epochs: int = 0) -> None:
    figs = []

    main_fig = _plot_pair_grid(METRIC_PAIRS, dfs, labels,
                               "Training vs Validation Metrics (binary PRF)", skip_epochs)
    if main_fig is not None:
        figs.append((main_fig, ""))

    for cls_key, cls_label in PRF_CLASSES:
        pairs = [(f"train_{cls_key}_{k}", f"val_{cls_key}_{k}", lbl) for k, lbl in PER_CLASS_PAIRS]
        fig = _plot_pair_grid(pairs, dfs, labels,
                              f"Training vs Validation Metrics — {cls_label}", skip_epochs)
        if fig is not None:
            figs.append((fig, f"_{cls_key}"))

    lr_fig = _plot_lr(dfs, labels, skip_epochs)
    if lr_fig is not None:
        figs.append((lr_fig, "_lr"))

    if output is None:
        plt.show()
        return

    out_path = Path(output)
    if len(figs) == 1:
        _save_or_show(figs[0][0], str(out_path))
        return
    for fig, suffix in figs:
        target = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix}")
        _save_or_show(fig, str(target))


def print_summary(dfs: list[pd.DataFrame], labels: list[str]) -> None:
    for df, label in zip(dfs, labels):
        print(f"\n{'='*60}")
        print(f"  {label}  ({len(df)} epochs)")
        print(f"{'='*60}")

        last = df.iloc[-1]
        best_val_loss_epoch = df.loc[df["val_loss"].idxmin(), "epoch"] if "val_loss" in df.columns else None
        best_val_f1_epoch   = df.loc[df["val_best_f1"].idxmax(), "epoch"] if "val_best_f1" in df.columns else None

        print(f"  Final epoch {int(last['epoch'])}:")
        for col in SUMMARY_COLS:
            if col in df.columns:
                print(f"    {col:<40} {last[col]:.4f}")

        for cls_key, cls_label in PRF_CLASSES:
            printed_header = False
            for split in ["train", "val"]:
                for key, _ in PER_CLASS_PAIRS + [("n_positive", "n_positive")]:
                    col = f"{split}_{cls_key}_{key}"
                    if col in df.columns:
                        if not printed_header:
                            print(f"  [{cls_label}]")
                            printed_header = True
                        print(f"    {col:<40} {last[col]:.4f}")

        if best_val_loss_epoch is not None:
            row = df[df["epoch"] == best_val_loss_epoch].iloc[0]
            print(f"\n  Best val_loss  @ epoch {int(best_val_loss_epoch)}: {row['val_loss']:.4f}")
        if best_val_f1_epoch is not None:
            row = df[df["epoch"] == best_val_f1_epoch].iloc[0]
            print(f"  Best val_best_f1 @ epoch {int(best_val_f1_epoch)}: {row['val_best_f1']:.4f}")

        for hit_col in ["val_argmax_hit_rate", "val_argmax3_hit_rate",
                        "val_argmax_flank_hit_rate", "val_type_accuracy"]:
            if hit_col in df.columns:
                best_epoch = df.loc[df[hit_col].idxmax(), "epoch"]
                row = df[df["epoch"] == best_epoch].iloc[0]
                print(f"  Best {hit_col} @ epoch {int(best_epoch)}: {row[hit_col]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize metrics.csv from PRForm training.")
    parser.add_argument("csv_files", nargs="+", help="Path(s) to metrics.csv file(s)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend labels for each CSV (defaults to file paths)")
    parser.add_argument("--output", default=None,
                        help="Output image path (e.g. metrics.png). If omitted, display interactively.")
    parser.add_argument("--no-plot", action="store_true", help="Print summary only, skip plotting.")
    parser.add_argument("--skip-epochs", type=int, default=0,
                        help="Exclude the first N epochs from the plot (default: 0).")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.csv_files):
        parser.error("--labels count must match number of CSV files")

    dfs = []
    for path in args.csv_files:
        try:
            dfs.append(load_csv(path))
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            sys.exit(1)

    labels = args.labels if args.labels else [Path(p).parent.name or p for p in args.csv_files]

    print_summary(dfs, labels)

    if not args.no_plot:
        plot_metrics(dfs, labels, args.output, skip_epochs=args.skip_epochs)


if __name__ == "__main__":
    main()
