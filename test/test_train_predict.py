#!/usr/bin/env python3
"""
End-to-end tests for the PRForm pipeline:
  1. create_datasets  → generate small h5 / pkl datasets
  2. PRFDataset       → verify 3-tuple (x, y, meta) return
  3. train.py helpers → compute_pos_weight, run_epoch
  4. predict.py       → inference + per-nucleotide output

Uses tiny sequences and the smallest model (PRForm_80nt, flank=40)
so everything runs in seconds on CPU.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = Path(__file__).resolve().parent
PRFORM_DIR = REPO_ROOT / "prform"
sys.path.insert(0, str(PRFORM_DIR))
sys.path.insert(0, str(REPO_ROOT))

BLOCK_LEN = 200
FLANK = 40


def _create_datasets(fmt: str) -> Path:
    """Run create_datasets.py and return output directory."""
    csv_path = TEST_DIR / "test_data.csv"
    out_dir = TEST_DIR / f"out_{fmt}_e2e"
    out_dir.mkdir(exist_ok=True)
    script = REPO_ROOT / "prform" / "utils" / "create_datasets.py"
    cmd = [
        sys.executable, str(script),
        "--csv", str(csv_path),
        "--outdir", str(out_dir),
        "--block_len", str(BLOCK_LEN),
        "--flank", str(FLANK),
        "--format", fmt,
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, (
        f"create_datasets ({fmt}) failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return out_dir


def test_dataloader_pkl():
    """PRFDataset returns (x, y, meta) for pkl format."""
    data_dir = _create_datasets("pkl")
    from dataloader import PRFDataset

    ds = PRFDataset(str(data_dir / "train.pkl"), flank=FLANK)
    assert len(ds) > 0, "Dataset is empty"

    x, y, meta = ds[0]
    assert x.shape[0] == 4, f"Expected 4 channels, got {x.shape[0]}"
    assert x.shape[1] == BLOCK_LEN + 2 * FLANK
    assert y.shape[0] == BLOCK_LEN
    assert isinstance(meta, dict)
    assert "accession_id" in meta
    assert "species_taxid" in meta
    assert "genus_name" in meta
    assert isinstance(meta["block_idx"], int)
    print(f"test_dataloader_pkl: PASSED  (dataset size={len(ds)}, meta keys={list(meta.keys())})")


def test_dataloader_h5():
    """PRFDataset returns (x, y, meta) for h5 format."""
    data_dir = _create_datasets("h5")
    from dataloader import PRFDataset

    ds = PRFDataset(str(data_dir / "train.h5"), flank=FLANK)
    assert len(ds) > 0

    x, y, meta = ds[0]
    assert x.shape == (4, BLOCK_LEN + 2 * FLANK)
    assert y.shape == (BLOCK_LEN,)
    assert meta["accession_id"] != ""
    assert meta["genus_name"] != ""
    print(f"test_dataloader_h5: PASSED  (dataset size={len(ds)})")


def test_dataloader_collate():
    """DataLoader correctly collates the 3-tuple batches."""
    data_dir = _create_datasets("h5")
    import torch
    from torch.utils.data import DataLoader
    from dataloader import PRFDataset

    ds = PRFDataset(str(data_dir / "train.h5"), flank=FLANK)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert len(batch) == 3, f"Expected 3-tuple, got {len(batch)}-tuple"
    inputs, targets, meta = batch
    assert inputs.shape[0] == min(2, len(ds))
    assert targets.shape[0] == min(2, len(ds))
    assert isinstance(meta, dict)
    assert isinstance(meta["accession_id"], list)
    print("test_dataloader_collate: PASSED")


def test_compute_pos_weight():
    """compute_pos_weight handles 3-tuple iteration."""
    data_dir = _create_datasets("h5")
    import torch
    from dataloader import PRFDataset
    sys.path.insert(0, str(PRFORM_DIR))
    from train import compute_pos_weight

    ds = PRFDataset(str(data_dir / "train.h5"), flank=FLANK)
    w = compute_pos_weight(ds)
    assert isinstance(w, torch.Tensor)
    assert w.ndim == 0
    assert w.item() > 0
    print(f"test_compute_pos_weight: PASSED  (weight={w.item():.2f})")


def test_train_one_epoch():
    """Run one epoch of training + validation with the smallest model."""
    data_dir = _create_datasets("h5")
    import torch
    from torch.utils.data import DataLoader
    from dataloader import PRFDataset
    from model import PRForm_80nt
    from train import run_epoch

    train_ds = PRFDataset(str(data_dir / "train.h5"), flank=FLANK)
    val_path = data_dir / "val.h5"
    if not val_path.exists():
        print("test_train_one_epoch: SKIPPED (no val split)")
        return

    val_ds = PRFDataset(str(val_path), flank=FLANK)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    device = torch.device("cpu")
    model = PRForm_80nt(
        in_channels=train_ds.in_channels, mid_channels=8,
        out_channels=1, dropout=0.0,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
    assert "loss" in train_metrics
    assert "pr_auc" in train_metrics

    val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    assert "loss" in val_metrics

    print(f"test_train_one_epoch: PASSED  "
          f"(train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f})")

    checkpoint_dir = TEST_DIR / "out_checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)
    ckpt_path = checkpoint_dir / "model_test.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  checkpoint saved to {ckpt_path}")
    return ckpt_path


def test_predict():
    """Run predict.py end-to-end and validate outputs."""
    data_dir = _create_datasets("h5")
    test_h5 = data_dir / "test.h5"
    if not test_h5.exists():
        train_h5 = data_dir / "train.h5"
        test_h5 = train_h5

    ckpt_path = _train_and_get_checkpoint(data_dir)

    pred_dir = TEST_DIR / "out_predictions"
    pred_dir.mkdir(exist_ok=True)
    script = PRFORM_DIR / "predict.py"

    cmd = [
        sys.executable, str(script),
        "--checkpoint", str(ckpt_path),
        "--data", str(test_h5),
        "--flank", str(FLANK),
        "--batch_size", "4",
        "--mid_channels", "8",
        "--dropout", "0.0",
        "--output_dir", str(pred_dir),
    ]
    result = subprocess.run(cmd, cwd=str(PRFORM_DIR), capture_output=True, text=True)
    assert result.returncode == 0, (
        f"predict.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    h5_out = pred_dir / "predictions.h5"
    assert h5_out.exists(), f"Missing {h5_out}"

    import h5py
    import pandas as pd

    with h5py.File(h5_out, "r") as hf:
        assert "probabilities" in hf
        assert "targets" in hf
        assert "metadata" in hf
        probs = hf["probabilities"][:]
        assert probs.ndim == 2
        assert 0.0 <= probs.min() and probs.max() <= 1.0
        assert "accession_id" in hf["metadata"]
        assert "genus_name" in hf["metadata"]
        print(f"  predictions.h5: {probs.shape[0]} blocks × {probs.shape[1]} positions")

    tsv_out = pred_dir / "predictions_per_record.tsv"
    assert tsv_out.exists(), f"Missing {tsv_out}"
    df = pd.read_csv(tsv_out, sep="\t")
    assert "accession_id" in df.columns
    assert "per_nucleotide_probs" in df.columns
    assert "genus_name" in df.columns
    assert len(df) > 0

    first_probs_str = df.iloc[0]["per_nucleotide_probs"]
    probs_arr = np.array([float(x) for x in first_probs_str.split(",")])
    assert len(probs_arr) > 0
    assert 0.0 <= probs_arr.min() and probs_arr.max() <= 1.0

    print(f"test_predict: PASSED  ({len(df)} records in TSV, nucleotide probs verified)")


def _train_and_get_checkpoint(data_dir: Path) -> Path:
    """Train a tiny model and return checkpoint path."""
    import torch
    from torch.utils.data import DataLoader
    from dataloader import PRFDataset
    from model import PRForm_80nt

    train_h5 = data_dir / "train.h5"
    ds = PRFDataset(str(train_h5), flank=FLANK)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    device = torch.device("cpu")
    model = PRForm_80nt(
        in_channels=ds.in_channels, mid_channels=8,
        out_channels=1, dropout=0.0,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for inputs, targets, _meta in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    ckpt_dir = TEST_DIR / "out_checkpoint"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "model_test.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    print("=" * 60)
    print("PRForm end-to-end tests")
    print("=" * 60)
    test_dataloader_pkl()
    test_dataloader_h5()
    test_dataloader_collate()
    test_compute_pos_weight()
    test_train_one_epoch()
    test_predict()
    print("\n" + "=" * 60)
    print("All end-to-end tests passed.")
    print("=" * 60)
