"""
Build and validate the canonical Bright2Nuc brightfield cache used by M1-M3.

Purpose
-------
Create a memory-mapped NumPy cache containing the raw per-nucleus brightfield
stacks in the exact order defined by the frozen culture-wise folds. The script
also writes an explicit cache index and verifies that cached crops are identical
to their source TIFF files for a deterministic set of samples.

Inputs
------
- ``data/processed/folds.csv``
- ``data/raw/TF_prediction/SingleNucleiBF/``

Outputs
-------
- ``data/processed/m1_bf16_uint8.npy``
- ``data/processed/m1_bf16_index.csv``

Reproducibility
---------------
The cache order is fixed by ``folds.csv``. Validation samples use random seed 42,
and the SHA256 digest of the generated index is printed at the end.

Notes
-----
The cache contains raw uint8 brightfield pixels only. No normalization is
applied here, and no fluorescence data are stored in the cache.
"""

from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/TF_prediction"
BF_DIR = RAW / "SingleNucleiBF"

FOLDS_PATH = ROOT / "data/processed/folds.csv"

CACHE_DIR = ROOT / "data/processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = CACHE_DIR / "m1_bf16_uint8.npy"
INDEX_PATH = CACHE_DIR / "m1_bf16_index.csv"


# ================================================================
# 1. CANONICAL INDEX = folds.csv
# ================================================================

folds = pd.read_csv(FOLDS_PATH)

required = {
    "filename",
    "culture",
    "dataset_tag",
    "TF_class",
    "fold",
}

missing = required - set(folds.columns)

if missing:
    raise RuntimeError(
        f"Missing columns in folds.csv: {missing}"
    )

if len(folds) != 30877:
    raise RuntimeError(
        f"Expected 30,877 nuclei, got {len(folds)}"
    )

if folds["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate identifiers in folds.csv"
    )


# ================================================================
# 2. CACHE CREATION
# ================================================================

shape = (
    len(folds),
    16,
    64,
    64,
)

print("=" * 80)
print("CACHE M1 — BRIGHTFIELD 16 x 64 x 64")
print("=" * 80)

print("Shape :", shape)
print("Dtype : uint8")
print(
    "Theoretical size: "
    f"{np.prod(shape) / 1024**3:.2f} GiB"
)

cache = np.lib.format.open_memmap(
    CACHE_PATH,
    mode="w+",
    dtype=np.uint8,
    shape=shape,
)

for i, name in enumerate(
    folds["filename"].astype(str)
):

    path = BF_DIR / name

    if not path.exists():
        raise FileNotFoundError(path)

    img = tifffile.imread(path)

    if img.shape != (16, 64, 64):
        raise RuntimeError(
            f"{name}: shape {img.shape}, "
            "expected (16, 64, 64)"
        )

    if img.dtype != np.uint8:
        raise RuntimeError(
            f"{name}: dtype {img.dtype}, "
            "expected uint8"
        )

    cache[i] = img

    if (i + 1) % 2500 == 0:
        print(
            f"{i+1:6d} / {len(folds)}"
        )

cache.flush()

del cache


# ================================================================
# 3. INDEX
# ================================================================

index = folds[
    [
        "filename",
        "culture",
        "dataset_tag",
        "TF_class",
        "fold",
    ]
].copy()

index.insert(
    0,
    "cache_index",
    np.arange(
        len(index),
        dtype=np.int64,
    ),
)

index.to_csv(
    INDEX_PATH,
    index=False,
)


# ================================================================
# 4. CACHE VERIFICATION
# ================================================================

cache = np.load(
    CACHE_PATH,
    mmap_mode="r",
)

print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

print("Actual shape :", cache.shape)
print("Actual dtype :", cache.dtype)

assert cache.shape == (
    30877,
    16,
    64,
    64,
)

assert cache.dtype == np.uint8


# Deterministic verification plus a few random samples
rng = random.Random(42)

test_indices = [
    0,
    1,
    len(index) // 2,
    len(index) - 1,
]

test_indices += rng.sample(
    range(len(index)),
    20,
)

test_indices = sorted(
    set(test_indices)
)

max_diff = 0

for i in test_indices:

    name = index.iloc[i]["filename"]

    original = tifffile.imread(
        BF_DIR / name
    )

    cached = np.asarray(
        cache[i]
    )

    diff = np.max(
        np.abs(
            original.astype(np.int16)
            - cached.astype(np.int16)
        )
    )

    max_diff = max(
        max_diff,
        int(diff),
    )

    if diff != 0:
        raise RuntimeError(
            f"Cache differs from TIFF for "
            f"{name}: max diff={diff}"
        )

print(
    "Files compared:",
    len(test_indices),
)

print(
    "Maximum difference:",
    max_diff,
)

print(
    "TIFF/cache equality:",
    "OK" if max_diff == 0 else "ERROR",
)


# ================================================================
# 5. QUICK DISTRIBUTION CHECK
# ================================================================

sample_idx = np.linspace(
    0,
    len(index) - 1,
    1000,
    dtype=int,
)

sample = np.asarray(
    cache[sample_idx],
    dtype=np.float32,
)

print("\nStatistics over 1,000 crops:")
print(
    f"min    : {sample.min():.1f}"
)
print(
    f"max    : {sample.max():.1f}"
)
print(
    f"mean   : {sample.mean():.3f}"
)
print(
    f"std    : {sample.std():.3f}"
)

for p in [
    1,
    5,
    50,
    95,
    99,
]:
    print(
        f"p{p:02d}    : "
        f"{np.percentile(sample, p):.3f}"
    )


# ================================================================
# 6. INDEX SHA256
# ================================================================

h = hashlib.sha256()

with open(
    INDEX_PATH,
    "rb",
) as f:

    for block in iter(
        lambda: f.read(1024 * 1024),
        b"",
    ):
        h.update(block)

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(CACHE_PATH)
print(INDEX_PATH)

print(
    "\nSHA256 index :",
    h.hexdigest(),
)

print("\nThe cache contains brightfield pixels only.")
print("No normalization has been applied yet.")
print("No fluorescence is used.")
print("\nDONE")
