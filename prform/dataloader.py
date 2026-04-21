"""
PyTorch Dataset for PRF (Programmed Ribosomal Frameshift) prediction.

Supports both output formats from create_datasets.py:
  - Pickle (.pkl): list of dicts with 'sequence' (L, C) and 'y' (block_len, K)
  - HDF5 (.h5): datasets X (N, L, C) and Y (N, block_len, K)

where C is the number of encoding channels (e.g. 4 for standard ACGT)
and K is the number of label classes (3 for PRF: [no-PRF, dir-1, dir+1]).

Returns tensors in the format expected by the PRForm model:
  - X: (C, L) float32  — channels-first for Conv1d
  - Y: (block_len, K) float32 — one-hot class labels for CrossEntropyLoss
  - meta: dict of metadata strings + block_idx int

The encoding dimension C is auto-detected from the data and exposed via
the `in_channels` property so the model can be constructed accordingly.

Augmentation is applied on-the-fly in __getitem__ when an Augmentor is provided.
Pass an augment.Augmentor instance to the constructor; it is only applied during
training (not validation/test) — the caller is responsible for not passing one
to the validation dataset.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

METADATA_STR_KEYS = [
    "accession_id", "record_id", "cluster_id",
    "species_taxid", "genus_taxid", "species_name", "genus_name",
]


class PRFDataset(Dataset):
    """
    Dataset for PRF prediction that auto-detects pickle vs HDF5 format.

    Parameters
    ----------
    path : str
        Path to a .pkl or .h5 file produced by create_prf_datasets.py.
    fraction : float
        Fraction of data to use (for quick experiments). Default: 1.0.
    flank : int
        Expected flank size. Used only for validation logging; the actual
        flank is baked into the data file. Default: 5000.

    Attributes
    ----------
    in_channels : int
        Number of encoding channels (last dim of sequence array).
        Read from the data — not hardcoded.
    augmentor : augment.Augmentor or None
        Optional on-the-fly augmentor applied in __getitem__ before
        transposing x to channels-first format.  Pass None (default) to
        disable augmentation (e.g. for validation / test sets).
    """

    def __init__(self, path, fraction=1.0, flank=5000, augmentor=None):
        super().__init__()
        self.path = path
        self.flank = flank

        self.augmentor = augmentor

        if path.endswith(".h5"):
            self._init_h5(path, fraction)
        elif path.endswith(".pkl"):
            self._init_pkl(path, fraction)
        else:
            raise ValueError(
                f"Unsupported file format: {path}. Expected .pkl or .h5"
            )

    def _init_pkl(self, path, fraction):
        """Load pickle format: list of dicts with 'sequence' and 'y'."""
        self.format = "pkl"
        with open(path, "rb") as f:
            records = pickle.load(f)
        n = max(1, int(len(records) * fraction))
        self.records = records[:n]
        self._h5_file = None
        self._meta = None
        self._in_channels = self.records[0]["sequence"].shape[-1]

    def _init_h5(self, path, fraction):
        """
        Load HDF5 format. We use lazy loading via h5py to avoid reading
        the entire file into memory.
        """
        import h5py

        self.format = "h5"
        self._h5_file = h5py.File(path, "r")
        self._X = self._h5_file["X"]
        self._Y = self._h5_file["Y"]
        total = self._X.shape[0]
        self._n = max(1, int(total * fraction))
        self.records = None
        self._meta = self._h5_file["metadata"] if "metadata" in self._h5_file else {}
        self._in_channels = self._X.shape[-1]

    @property
    def in_channels(self):
        """Number of encoding channels (auto-detected from data)."""
        return self._in_channels

    def __len__(self):
        if self.format == "pkl":
            return len(self.records)
        else:
            return self._n

    def _read_metadata(self, idx):
        """Read metadata dict for sample at index ``idx``."""
        if self.format == "pkl":
            rec = self.records[idx]
            meta = {k: str(rec.get(k, "")) for k in METADATA_STR_KEYS}
            meta["block_idx"] = int(rec.get("block_idx", 0))
            meta["sample_weight"] = float(rec.get("sample_weight", 1.0))
        else:
            meta = {}
            for k in METADATA_STR_KEYS:
                if k in self._meta:
                    val = self._meta[k][idx]
                    meta[k] = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                else:
                    meta[k] = ""
            if "block_idx" in self._meta:
                meta["block_idx"] = int(self._meta["block_idx"][idx])
            else:
                meta["block_idx"] = 0
            if "sample_weight" in self._meta:
                meta["sample_weight"] = float(self._meta["sample_weight"][idx])
            else:
                meta["sample_weight"] = 1.0
        return meta

    def __getitem__(self, idx):
        if self.format == "pkl":
            rec = self.records[idx]
            x = rec["sequence"]  # (L, C) uint8
            y = rec["y"]         # (block_len, K) uint8 one-hot
        else:
            x = self._X[idx]     # (L, C) uint8
            y = self._Y[idx]     # (block_len, K) uint8 one-hot

        x = np.asarray(x, dtype=np.float32)              # (L, C)
        y = np.asarray(y, dtype=np.float32)              # (block_len, K)

        # Apply on-the-fly augmentation (train only; caller passes None for val/test)
        if self.augmentor is not None:
            x, y = self.augmentor(x, y)

        # Transpose X from (L, C) → (C, L) for Conv1d
        x = x.T  # (C, L)

        meta = self._read_metadata(idx)

        return torch.from_numpy(x), torch.from_numpy(y), meta

    def __del__(self):
        """Close HDF5 file handle if open."""
        if hasattr(self, "_h5_file") and self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
