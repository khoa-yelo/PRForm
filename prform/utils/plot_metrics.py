"""
Visualize training metrics from metrics.csv output files.

Usage:
    python plot_metrics.py <metrics.csv> [<metrics2.csv> ...] [--labels l1 l2] [--output fig.png]

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
import matplotlib.gridspec as gridspec


TRAIN_METRICS = [
    ("train_loss",    "Loss"),
    ("train_pr_auc",  "PR-AUC"),
    ("train_roc_auc", "ROC-AUC"),
    ("train_topk_acc","Top-k Acc"),
    ("train_best_f1", "Best F1"),
]

VAL_METRICS = [
    ("val_loss",    "Loss"),
    ("val_pr_auc",  "PR-AUC"),
    ("val_roc_auc", "ROC-AUC"),
    ("val_topk_acc","Top-k Acc"),
    ("val_best_f1", "Best F1"),
]

# Pairs for side-by-side train vs val plots
METRIC_PAIRS = [
    ("train_loss",               "val_loss",               "Loss"),
    ("train_pr_auc",             "val_pr_auc",             "PR-AUC"),
    ("train_roc_auc",            "val_roc_auc",            "ROC-AUC"),
    ("train_topk_acc",           "val_topk_acc",           "Top-k Accuracy"),
    ("train_best_f1",            "val_best_f1",            "Best F1"),
    ("train_argmax_hit_rate",    "val_argmax_hit_rate",    "Argmax Hit Rate (Hit@1)"),
    ("train_argmax3_hit_rate",   "val_argmax3_hit_rate",   "Argmax Top-3 Hit Rate (Hit@3)"),
    ("train_argmax_flank_hit_rate", "val_argmax_flank_hit_rate", "Argmax Flank Hit Rate (±3 bp)"),
]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"No 'epoch' column in {path}")
    return df


def plot_metrics(dfs: list[pd.DataFrame], labels: list[str], output: str | None, skip_epochs: int = 0) -> None:
    n_pairs = len(METRIC_PAIRS)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4 * n_pairs), squeeze=False)
    fig.suptitle("Training vs Validation Metrics", fontsize=16, fontweight="bold", y=1.01)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row, (train_col, val_col, title) in enumerate(METRIC_PAIRS):
        ax_train = axes[row][0]
        ax_val   = axes[row][1]

        for i, (df, label) in enumerate(zip(dfs, labels)):
            color = colors[i % len(colors)]
            df = df.iloc[skip_epochs:]
            epochs = df["epoch"]

            if train_col in df.columns:
                ax_train.plot(epochs, df[train_col], color=color, label=label, marker="o", markersize=3)
            if val_col in df.columns:
                ax_val.plot(epochs, df[val_col], color=color, label=label, marker="o", markersize=3)

        for ax, split in [(ax_train, "Train"), (ax_val, "Val")]:
            ax.set_title(f"{split} — {title}")
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            if len(dfs) > 1:
                ax.legend(fontsize=8)

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def print_summary(dfs: list[pd.DataFrame], labels: list[str]) -> None:
    for df, label in zip(dfs, labels):
        print(f"\n{'='*60}")
        print(f"  {label}  ({len(df)} epochs)")
        print(f"{'='*60}")

        last = df.iloc[-1]
        best_val_loss_epoch = df.loc[df["val_loss"].idxmin(), "epoch"] if "val_loss" in df.columns else None
        best_val_f1_epoch   = df.loc[df["val_best_f1"].idxmax(), "epoch"] if "val_best_f1" in df.columns else None

        print(f"  Final epoch {int(last['epoch'])}:")
        for col in [
            "train_loss", "train_pr_auc", "train_roc_auc", "train_best_f1",
            "train_argmax_hit_rate", "train_argmax3_hit_rate", "train_argmax_flank_hit_rate",
            "val_loss",   "val_pr_auc",   "val_roc_auc",   "val_best_f1",
            "val_argmax_hit_rate", "val_argmax3_hit_rate", "val_argmax_flank_hit_rate",
        ]:
            if col in df.columns:
                print(f"    {col:<34} {last[col]:.4f}")

        if best_val_loss_epoch is not None:
            row = df[df["epoch"] == best_val_loss_epoch].iloc[0]
            print(f"\n  Best val_loss  @ epoch {int(best_val_loss_epoch)}: {row['val_loss']:.4f}")
        if best_val_f1_epoch is not None:
            row = df[df["epoch"] == best_val_f1_epoch].iloc[0]
            print(f"  Best val_best_f1 @ epoch {int(best_val_f1_epoch)}: {row['val_best_f1']:.4f}")

        for hit_col in ["val_argmax_hit_rate", "val_argmax3_hit_rate", "val_argmax_flank_hit_rate"]:
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
