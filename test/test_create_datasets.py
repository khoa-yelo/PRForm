#!/usr/bin/env python3
"""
Test PRForm create_datasets.py (and SpliceAI-style blocking logic).

Runs create_prf_datasets on test_data.csv and validates output structure.
"""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def test_create_datasets_pkl():
    """Run create_datasets via subprocess and validate pickle output."""
    csv_path = TEST_DIR / "test_data.csv"
    out_dir = TEST_DIR / "out_pkl"
    out_dir.mkdir(exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    script = REPO_ROOT / "prform" / "utils" / "create_datasets.py"
    cmd = [
        sys.executable,
        str(script),
        "--csv", str(csv_path),
        "--outdir", str(out_dir),
        "--block_len", "200",
        "--flank", "50",
        "--format", "pkl",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, f"create_datasets failed: {result.stderr}"

    for split in ("train", "val", "test"):
        pkl_path = out_dir / f"{split}.pkl"
        assert pkl_path.exists(), f"Expected {pkl_path}"
        with open(pkl_path, "rb") as f:
            recs = pickle.load(f)
        for r in recs:
            assert "sequence" in r
            assert "y" in r
            assert r["sequence"].shape[1] == 4
            assert r["sequence"].shape[0] == 200 + 100  # block_len + 2*flank
            assert r["y"].shape == (200, 1)
            assert r["y"].dtype == np.uint8
    print("test_create_datasets_pkl: PASSED")


def test_create_datasets_h5():
    """Run create_datasets with HDF5 output."""
    csv_path = TEST_DIR / "test_data.csv"
    out_dir = TEST_DIR / "out_h5"
    out_dir.mkdir(exist_ok=True)

    script = REPO_ROOT / "prform" / "utils" / "create_datasets.py"
    cmd = [
        sys.executable,
        str(script),
        "--csv", str(csv_path),
        "--outdir", str(out_dir),
        "--block_len", "200",
        "--flank", "50",
        "--format", "h5",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, f"create_datasets failed: {result.stderr}"

    import h5py
    for split in ("train", "val", "test"):
        h5_path = out_dir / f"{split}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as hf:
            assert "X" in hf and "Y" in hf and "metadata" in hf
            assert hf["X"].shape[1] == 300
            assert hf["X"].shape[2] == 4
            assert hf["Y"].shape[1] == 200
            assert hf["Y"].shape[2] == 1
    print("test_create_datasets_h5: PASSED")


def test_process_record_blocks():
    """Unit test for process_record_blocks."""
    from prform.utils.create_datasets import process_record_blocks
    import pandas as pd

    row = pd.Series({
        "accession_id": "ACC1",
        "record_id": "REC1",
        "cluster_id": "CLU1",
        "prf_position": "10",
        "strand": "+",
        "sequence": "A" * 100 + "C" * 100,  # 200 bp
    })
    blocks = list(process_record_blocks(row, block_len=200, flank=50))
    assert len(blocks) >= 1
    blk = blocks[0]
    assert blk["sequence"].shape == (300, 4)
    assert blk["y"].shape == (200, 1)
    assert blk["accession_id"] == "ACC1"
    # PRF at 1-based 10 -> 0-based 9, should be in first block
    assert blk["y"][9, 0] == 1
    print("test_process_record_blocks: PASSED")


def test_one_hot_encode():
    """Unit test for one_hot_encode."""
    from prform.utils.create_datasets import one_hot_encode

    oh = one_hot_encode("ACGT")
    assert oh.shape == (4, 4)
    assert oh[0, 0] == 1  # A
    assert oh[1, 1] == 1  # C
    assert oh[2, 2] == 1  # G
    assert oh[3, 3] == 1  # T
    print("test_one_hot_encode: PASSED")


if __name__ == "__main__":
    test_one_hot_encode()
    test_process_record_blocks()
    test_create_datasets_pkl()
    test_create_datasets_h5()
    print("\nAll tests passed.")
