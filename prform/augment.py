"""
augment.py — On-the-fly sequence augmentation for one-hot-encoded nucleotide sequences.

All transforms operate on:
  x : np.ndarray, shape (L, C)       — one-hot sequence (L positions, C channels; stored as (L,C))
  y : np.ndarray, shape (block_len, K) — one-hot class labels (K classes; channel 0 = background)
                 or shape (block_len,) — binary labels (legacy)

For 2D y, positions with no signal default to background one-hot [1, 0, ..., 0]
rather than all-zero, so CrossEntropyLoss sees a valid class index everywhere.

Encoding convention (matches create_datasets.py):
  A=0, C=1, G=2, T/U=3
  All-zero rows represent ambiguous / padded positions.

Length-preserving transforms (point_substitution, inversion):
  Labels are updated in place — positions within the label window that are inverted
  get their label order reversed to stay consistent with the new sequence.

Length-changing transforms (deletion, insertion):
  The sequence is kept at L positions (gaps are zero-padded at the right end for
  deletions, or the sequence is truncated at the right for insertions).
  Labels are shifted accordingly so y[i] still describes position flank+i in new_x.

Usage — standalone:
    from augment import Augmentor
    aug = Augmentor(flank=5000)
    x_aug, y_aug = aug(x, y)          # x: (L, 4), y: (block_len,)

Usage — inside PRFDataset.__getitem__:
    x, y, meta = ...                   # after loading, before transpose
    if self.augmentor is not None:
        x, y = self.augmentor(x, y)
"""

from __future__ import annotations

import numpy as np

# Complement map for 4-channel one-hot: A↔T (0↔3), C↔G (1↔2)
_COMPLEMENT_IDX = np.array([3, 2, 1, 0], dtype=np.intp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _complement(oh: np.ndarray) -> np.ndarray:
    """Return the complement of a one-hot array, shape (..., 4).

    All-zero rows (ambiguous/padded) remain all-zero because every channel
    flips from 0 → 0 symmetrically.
    """
    return oh[..., _COMPLEMENT_IDX]


def _random_onehot(size: int, rng: np.random.Generator, n_channels: int = 4) -> np.ndarray:
    """Return ``size`` random one-hot rows drawn uniformly over [0, n_channels)."""
    out = np.zeros((size, n_channels), dtype=np.uint8)
    idx = rng.integers(0, n_channels, size=size)
    out[np.arange(size), idx] = 1
    return out


def _label_window(L: int, block_len: int, flank: int) -> tuple[int, int]:
    """Return (start, stop) of the label window inside a sequence of length L."""
    return flank, flank + block_len


def _empty_like_labels(y: np.ndarray) -> np.ndarray:
    """Allocate a labels array matching y, filled with the background class.

    For 2D one-hot y, channel 0 is background → sets [:, 0] = 1.
    For 1D binary y, returns zeros.
    """
    new_y = np.zeros_like(y)
    if new_y.ndim == 2:
        new_y[:, 0] = 1
    return new_y


def _bg_row(y: np.ndarray):
    """Return the background value for a single row of y.

    Scalar 0 for 1D binary y; one-hot row [1, 0, ..., 0] for 2D y.
    """
    if y.ndim == 2:
        row = np.zeros(y.shape[1], dtype=y.dtype)
        row[0] = 1
        return row
    return 0


# ---------------------------------------------------------------------------
# Individual augmentation functions
# All accept and return (x, y) with the same dtype/shape.
# ---------------------------------------------------------------------------

def point_substitution(
    x: np.ndarray,
    y: np.ndarray,
    rate: float = 0.01,
    flank: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly replace nucleotides with a different base.

    Each position is mutated independently with probability ``rate``.
    The replacement is chosen uniformly from the *other* three bases so that
    the mutation always changes the sequence.  Ambiguous (all-zero) positions
    are left unchanged.

    Labels are not shifted (length-preserving); positions that are mutated
    carry the original label because point mutations at a PRF slippery site
    may create or destroy frameshifting — retaining the label is conservative
    and reflects the original annotation.

    Parameters
    ----------
    x : (L, C)
    y : (block_len,)
    rate : float, mutation rate per position (default 0.01 → ~1 % of bases)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = x.copy()
    L, C = x.shape

    # Only consider non-ambiguous positions (rows with exactly one 1)
    current_base = x.argmax(axis=1)           # (L,) — 0..C-1
    is_real = x.sum(axis=1) == 1              # (L,) bool — ignore all-zero rows

    mask = is_real & (rng.random(L) < rate)   # positions to mutate
    if not mask.any():
        return x, y

    n_mut = int(mask.sum())
    cur = current_base[mask]                   # (n_mut,) current base indices

    # Draw offsets 1..C-1 to guarantee a different base
    offsets = rng.integers(1, C, size=n_mut)
    new_bases = (cur + offsets) % C

    x[mask] = 0
    x[mask, new_bases] = 1

    return x, y


def deletion(
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
    start: int | None = None,
    flank: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Delete a contiguous chunk of ``chunk_size`` nucleotides.

    The sequence is kept at length L by zero-padding at the right end.
    Labels in the window are shifted left to reflect which original positions
    now fall within the label region.

    Parameters
    ----------
    x : (L, C)
    y : (block_len,)
    chunk_size : number of nucleotides to delete
    start : 0-based start of deletion (random if None)
    """
    if rng is None:
        rng = np.random.default_rng()

    L, C = x.shape
    block_len = y.shape[0]
    chunk_size = min(chunk_size, L)

    if start is None:
        start = int(rng.integers(0, L - chunk_size + 1))
    end = start + chunk_size

    # New sequence: delete [start, end), pad with zeros on the right
    new_x = np.concatenate(
        [x[:start], x[end:], np.zeros((chunk_size, C), dtype=x.dtype)]
    )

    # Recompute labels: new_y[i] is the label for position (flank+i) in new_x.
    # Position p in new_x corresponds to old position:
    #   p            if p < start
    #   p + chunk    if start <= p < L - chunk_size  (after deletion, content shifts)
    #   padding      if p >= L - chunk_size
    new_y = _empty_like_labels(y)
    lbl_start = flank
    lbl_end = flank + block_len

    for i in range(block_len):
        p = lbl_start + i                   # position in new_x
        if p < start:
            old_p = p
        elif p < L - chunk_size:
            old_p = p + chunk_size
        else:
            continue                         # padding region → background

        if lbl_start <= old_p < lbl_end:
            new_y[i] = y[old_p - lbl_start]
        # else: old_p was in flank → background (default)

    return new_x, new_y


def insertion(
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
    pos: int | None = None,
    flank: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert ``chunk_size`` random nucleotides at position ``pos``.

    The sequence is kept at length L by truncating from the right end.
    Labels are shifted right by the amount of inserted nucleotides that fall
    before or within the label window.

    Parameters
    ----------
    x : (L, C)
    y : (block_len,)
    chunk_size : number of nucleotides to insert
    pos : 0-based insertion site (random if None)
    """
    if rng is None:
        rng = np.random.default_rng()

    L, C = x.shape
    block_len = y.shape[0]

    if pos is None:
        pos = int(rng.integers(0, L - chunk_size + 1))

    inserted = _random_onehot(chunk_size, rng, C)

    # New sequence: insert at pos, truncate last chunk_size positions
    new_x = np.concatenate([x[:pos], inserted, x[pos : L - chunk_size]])

    # Recompute labels: position p in new_x came from:
    #   p            if p < pos                   (before insertion)
    #   random nt    if pos <= p < pos+chunk_size (inserted region)
    #   p - chunk    if p >= pos + chunk_size      (after insertion)
    new_y = _empty_like_labels(y)
    lbl_start = flank
    lbl_end = flank + block_len

    for i in range(block_len):
        p = lbl_start + i                    # position in new_x

        if p < pos:
            old_p = p
        elif p < pos + chunk_size:
            continue                          # inserted random nt → background
        else:
            old_p = p - chunk_size

        if lbl_start <= old_p < lbl_end:
            new_y[i] = y[old_p - lbl_start]

    return new_x, new_y


def inversion(
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
    start: int | None = None,
    flank: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse-complement a contiguous chunk (length-preserving).

    The chunk is simultaneously reversed in position and complemented
    (A↔T, C↔G), mimicking an inversion mutation.  Only works cleanly for
    C=4; for other channel counts the complement step is skipped and only
    the positional reversal is applied.

    Labels within the inverted region of the label window are reversed to
    stay consistent with the new position order.

    Parameters
    ----------
    x : (L, C)
    y : (block_len,)
    chunk_size : length of the inverted segment
    start : 0-based start of inversion (random if None)
    """
    if rng is None:
        rng = np.random.default_rng()

    L, C = x.shape
    block_len = y.shape[0]
    chunk_size = min(chunk_size, L)

    if start is None:
        start = int(rng.integers(0, L - chunk_size + 1))
    end = start + chunk_size

    new_x = x.copy()
    chunk = x[start:end][::-1]               # reverse positions
    if C == 4:
        chunk = _complement(chunk)            # complement bases
    new_x[start:end] = chunk

    # Recompute labels: for positions in [start, end) ∩ [lbl_start, lbl_end):
    # new position (start + end - 1 - p) now holds what was at old position p.
    # So new_y[i] where lbl_start+i is in the inverted region:
    #   the content at lbl_start+i in new_x came from old position start+end-1-(lbl_start+i)
    lbl_start = flank
    lbl_end = flank + block_len

    inv_lbl_start = max(start, lbl_start)
    inv_lbl_end = min(end, lbl_end)

    if inv_lbl_start >= inv_lbl_end:
        return new_x, y                       # inversion entirely outside label window

    new_y = y.copy()
    bg = _bg_row(y)
    for p in range(inv_lbl_start, inv_lbl_end):
        i = p - lbl_start                     # index in y for new position
        old_p = start + end - 1 - p           # old position this came from

        if lbl_start <= old_p < lbl_end:
            new_y[i] = y[old_p - lbl_start]
        else:
            new_y[i] = bg                     # source was in flank → background

    return new_x, new_y


def reverse_complement(
    x: np.ndarray,
    y: np.ndarray,
    flank: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse-complement the entire sequence.

    Reverses position order and applies base complementation (C=4 only).
    The label window is mirrored to match: y[i] → y[block_len - 1 - i].

    This simulates presenting the minus-strand of the same genome.

    Parameters
    ----------
    x : (L, C)
    y : (block_len,)
    """
    L, C = x.shape
    new_x = x[::-1].copy()
    if C == 4:
        new_x = _complement(new_x)

    # Mirror the label window
    new_y = y[::-1].copy()

    return new_x, new_y


# ---------------------------------------------------------------------------
# Augmentor class
# ---------------------------------------------------------------------------

class Augmentor:
    """Randomly applies a mixture of sequence augmentations.

    Parameters
    ----------
    flank : int
        Flank size used when constructing the dataset (default 5000).
        Must match the flank baked into the data files.
    p_point_sub : float
        Probability of applying point substitution each call (default 0.5).
    point_sub_rate : float
        Per-nucleotide mutation rate for point substitution (default 0.005).
    p_deletion : float
        Probability of applying one or more deletions (default 0.3).
    deletion_size_range : tuple[int, int]
        (min, max) chunk size for deletion events (default (1, 30)).
    n_deletion_events : int
        Max number of deletion events per call (default 3).
    p_insertion : float
        Probability of applying one or more insertions (default 0.2).
    insertion_size_range : tuple[int, int]
        (min, max) chunk size for insertion events (default (1, 15)).
    n_insertion_events : int
        Max number of insertion events per call (default 2).
    p_inversion : float
        Probability of applying one or more inversions (default 0.2).
    inversion_size_range : tuple[int, int]
        (min, max) chunk size for inversion events (default (5, 200)).
    n_inversion_events : int
        Max number of inversion events per call (default 2).
    p_rc : float
        Probability of reverse-complementing the whole sequence (default 0.1).
    seed : int | None
        Random seed for reproducibility (default None = non-deterministic).

    Example
    -------
    >>> aug = Augmentor(flank=5000, seed=42)
    >>> x_aug, y_aug = aug(x, y)     # x: (L, 4) uint8, y: (block_len,) float32
    """

    def __init__(
        self,
        flank: int = 5000,
        p_point_sub: float = 0.5,
        point_sub_rate: float = 0.005,
        p_deletion: float = 0.3,
        deletion_size_range: tuple[int, int] = (1, 30),
        n_deletion_events: int = 3,
        p_insertion: float = 0.2,
        insertion_size_range: tuple[int, int] = (1, 15),
        n_insertion_events: int = 2,
        p_inversion: float = 0.2,
        inversion_size_range: tuple[int, int] = (5, 200),
        n_inversion_events: int = 2,
        p_rc: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.flank = flank
        self.p_point_sub = p_point_sub
        self.point_sub_rate = point_sub_rate
        self.p_deletion = p_deletion
        self.deletion_size_range = deletion_size_range
        self.n_deletion_events = n_deletion_events
        self.p_insertion = p_insertion
        self.insertion_size_range = insertion_size_range
        self.n_insertion_events = n_insertion_events
        self.p_inversion = p_inversion
        self.inversion_size_range = inversion_size_range
        self.n_inversion_events = n_inversion_events
        self.p_rc = p_rc
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def __call__(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a random subset of augmentations.

        Parameters
        ----------
        x : (L, C) uint8 — one-hot sequence (channels-last, as stored in pkl/h5)
        y : (block_len,) float32 — binary label array

        Returns
        -------
        x_aug : (L, C) same dtype as input
        y_aug : (block_len,) same dtype as input
        """
        rng = self._rng

        # ---- reverse complement (full sequence) ----
        if rng.random() < self.p_rc:
            x, y = reverse_complement(x, y, flank=self.flank)

        # ---- point substitution ----
        if rng.random() < self.p_point_sub:
            x, y = point_substitution(
                x, y,
                rate=self.point_sub_rate,
                flank=self.flank,
                rng=rng,
            )

        # ---- inversions (reverse-complement of a chunk) ----
        if rng.random() < self.p_inversion:
            n = int(rng.integers(1, self.n_inversion_events + 1))
            for _ in range(n):
                lo, hi = self.inversion_size_range
                chunk = int(rng.integers(lo, hi + 1))
                x, y = inversion(x, y, chunk_size=chunk, flank=self.flank, rng=rng)

        # ---- deletions ----
        if rng.random() < self.p_deletion:
            n = int(rng.integers(1, self.n_deletion_events + 1))
            for _ in range(n):
                lo, hi = self.deletion_size_range
                chunk = int(rng.integers(lo, hi + 1))
                x, y = deletion(x, y, chunk_size=chunk, flank=self.flank, rng=rng)

        # ---- insertions ----
        if rng.random() < self.p_insertion:
            n = int(rng.integers(1, self.n_insertion_events + 1))
            for _ in range(n):
                lo, hi = self.insertion_size_range
                chunk = int(rng.integers(lo, hi + 1))
                x, y = insertion(x, y, chunk_size=chunk, flank=self.flank, rng=rng)

        return x, y
