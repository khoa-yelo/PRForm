#!/usr/bin/env python3
"""
Test PRForm create_datasets.py (and SpliceAI-style blocking logic).

Runs create_prf_datasets on test_data.csv and validates output structure,
including taxonomy metadata columns.
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

METADATA_STR_KEYS = [
    "accession_id", "record_id", "cluster_id",
    "species_taxid", "genus_taxid", "species_name", "genus_name",
]


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
            for k in METADATA_STR_KEYS:
                assert k in r, f"Missing metadata key '{k}' in pkl record"
            assert "block_idx" in r
            assert isinstance(r["block_idx"], int)
    print("test_create_datasets_pkl: PASSED")


def test_create_datasets_h5():
    """Run create_datasets with HDF5 output and validate metadata."""
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
            mg = hf["metadata"]
            for k in METADATA_STR_KEYS:
                assert k in mg, f"Missing metadata key '{k}' in h5"
                assert mg[k].shape[0] == hf["X"].shape[0]
            assert "block_idx" in mg
            assert mg["block_idx"].dtype == np.int32
    print("test_create_datasets_h5: PASSED")


def test_process_record_blocks():
    """Unit test for process_record_blocks with taxonomy metadata."""
    from prform.utils.create_datasets import process_record_blocks
    import pandas as pd

    row = pd.Series({
        "accession_id": "ACC1",
        "record_id": "REC1",
        "cluster_id": "CLU1",
        "prf_position": "10",
        "strand": "+",
        "sequence": "A" * 100 + "C" * 100,  # 200 bp
        "species_taxid": "12345",
        "genus_taxid": "6789",
        "species_name": "Test virus",
        "genus_name": "Testvirus",
    })
    blocks = list(process_record_blocks(row, block_len=200, flank=50))
    assert len(blocks) >= 1
    blk = blocks[0]
    assert blk["sequence"].shape == (300, 4)
    assert blk["y"].shape == (200, 1)
    assert blk["accession_id"] == "ACC1"
    assert blk["species_taxid"] == "12345"
    assert blk["genus_taxid"] == "6789"
    assert blk["species_name"] == "Test virus"
    assert blk["genus_name"] == "Testvirus"
    assert blk["y"][9, 0] == 1
    print("test_process_record_blocks: PASSED")


def test_process_record_blocks_no_metadata():
    """Verify process_record_blocks works when taxonomy columns are missing."""
    from prform.utils.create_datasets import process_record_blocks
    import pandas as pd

    row = pd.Series({
        "accession_id": "ACC2",
        "record_id": "REC2",
        "cluster_id": "CLU2",
        "prf_position": "5",
        "strand": "+",
        "sequence": "G" * 200,
    })
    blocks = list(process_record_blocks(row, block_len=200, flank=50))
    blk = blocks[0]
    assert blk["species_taxid"] == ""
    assert blk["genus_taxid"] == ""
    assert blk["species_name"] == ""
    assert blk["genus_name"] == ""
    print("test_process_record_blocks_no_metadata: PASSED")


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
    test_process_record_blocks_no_metadata()
    test_create_datasets_pkl()
    test_create_datasets_h5()
    print("\nAll create_datasets tests passed.")
