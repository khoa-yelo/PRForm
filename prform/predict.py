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
python predict.py \\
    --checkpoint output/model_final.pth \\
    --data       data/test.h5 \\
    --flank      5000 \\
    --output_dir predictions
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
from torch.utils.data import DataLoader

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
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset (.pkl or .h5)",
    )
    parser.add_argument(
        "--flank", type=int, default=5000,
        help="Flank size used during training (determines model architecture)",
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
        help="Fraction of dataset to predict on",
    )
    return parser.parse_args()


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
        for batch_idx, (inputs, targets, meta) in enumerate(loader):
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

            if (batch_idx + 1) % 50 == 0:
                logger.info("  batch %d / %d", batch_idx + 1, len(loader))

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
