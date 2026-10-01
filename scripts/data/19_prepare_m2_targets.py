"""
Prepare and freeze the auxiliary supervision table used by the M2/M3 models.

Purpose
-------
Merge the canonical brightfield cache index with the audited per-nucleus
transcription-factor measurements, verify alignment with ``TF_class``, assign
the validated TF channel mapping, and write the raw auxiliary-target table used
by downstream multitask training.

Inputs
------
- ``data/processed/m1_bf16_index.csv``
- ``outputs/data_audit/tf_expression_reconstruction.csv``

Output
------
- ``data/processed/m2_targets_raw.csv``

Reproducibility
---------------
The script checks the SHA256 digest of ``m1_bf16_index.csv`` before processing
and verifies that ``cache_index`` remains aligned with the brightfield cache.

Notes
-----
The marker values written here are raw fluorescence-derived supervision targets.
Their train-only percentile normalization is performed later by the training
scripts. ``dataset_tag`` remains descriptive and is not converted into
biological time.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = (
    ROOT
    / "data/processed/m1_bf16_index.csv"
)

EXPR_PATH = (
    ROOT
    / "outputs/data_audit/tf_expression_reconstruction.csv"
)

OUT_PATH = (
    ROOT
    / "data/processed/m2_targets_raw.csv"
)


EXPECTED_INDEX_SHA256 = (
    "55de2d34c1bba17e58ebd4ee3c60f5cd"
    "16efb4ea0e04582fc0166bb3835d6635"
)


# ================================================================
# HASH
# ================================================================

def sha256_file(path):
    """Return the SHA256 digest of a file using bounded-memory streaming."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


index_hash = sha256_file(
    INDEX_PATH
)

if index_hash != EXPECTED_INDEX_SHA256:
    raise RuntimeError(
        "m1_bf16_index.csv has changed!\n"
        f"Expected: {EXPECTED_INDEX_SHA256}\n"
        f"Found   : {index_hash}"
    )


print("=" * 88)
print("PREPARE M2 TARGETS")
print("=" * 88)

print(
    "Index SHA256 :",
    index_hash,
)


# ================================================================
# LOAD
# ================================================================

index = pd.read_csv(
    INDEX_PATH
)

expr = pd.read_csv(
    EXPR_PATH
)


print("\nIndex :", len(index))
print("TF    :", len(expr))


# ================================================================
# BASIC CHECKS
# ================================================================

if len(index) != 30877:
    raise RuntimeError(
        f"Index: expected 30877, got {len(index)}"
    )

if len(expr) != 30877:
    raise RuntimeError(
        f"TF: expected 30877, got {len(expr)}"
    )

if index["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filename in index"
    )

if expr["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filename in TF"
    )


required_expr = {
    "filename",
    "TF_class",
    "masked_ch0",
    "masked_ch1",
    "masked_ch2",
}

missing = (
    required_expr
    - set(expr.columns)
)

if missing:
    raise RuntimeError(
        f"Missing TF columns: {missing}"
    )


# ================================================================
# MERGE
# ================================================================

df = index.merge(
    expr[
        [
            "filename",
            "TF_class",
            "masked_ch0",
            "masked_ch1",
            "masked_ch2",
        ]
    ],
    on="filename",
    how="inner",
    suffixes=("_index", "_expr"),
    validate="one_to_one",
)


if len(df) != 30877:
    raise RuntimeError(
        "Incomplete merge."
    )


# ================================================================
# VERIFY DL
# ================================================================

max_diff = np.max(
    np.abs(
        df["TF_class_index"].to_numpy(
            dtype=np.float64
        )
        -
        df["TF_class_expr"].to_numpy(
            dtype=np.float64
        )
    )
)

print(
    "\nMaximum TF_class index/expression difference:",
    f"{max_diff:.12g}",
)

if max_diff > 1e-6:
    raise RuntimeError(
        "Inconsistent TF_class values across files."
    )


# ================================================================
# BIOLOGICAL CHANNEL MAPPING
#
# Experimental validation:
# ch0 = LUT cyan = OCT4
# ch1 = LUT green = FOXA2
# ch2 = LUT red = SOX17
# ================================================================

df = df.rename(
    columns={
        "TF_class_index": "DL",
        "masked_ch0": "OCT4_raw",
        "masked_ch1": "FOXA2_raw",
        "masked_ch2": "SOX17_raw",
    }
)

df = df.drop(
    columns=[
        "TF_class_expr",
    ]
)


# ================================================================
# CHECK FINITE VALUES
# ================================================================

targets = [
    "DL",
    "OCT4_raw",
    "FOXA2_raw",
    "SOX17_raw",
]

for col in targets:

    values = df[col].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Non-finite values in {col}"
        )


# ================================================================
# GLOBAL STATISTICS
# ================================================================

print("\n" + "=" * 88)
print("GLOBAL STATISTICS")
print("=" * 88)

for col in targets:

    x = df[col].to_numpy(
        dtype=np.float64
    )

    print(
        f"\n{col}"
    )

    print(
        f"  min  : {np.min(x):.6f}"
    )

    for p in [
        1,
        5,
        25,
        50,
        75,
        95,
        99,
    ]:
        print(
            f"  p{p:02d}  : "
            f"{np.percentile(x, p):.6f}"
        )

    print(
        f"  max  : {np.max(x):.6f}"
    )

    print(
        f"  mean : {np.mean(x):.6f}"
    )

    print(
        f"  std  : {np.std(x):.6f}"
    )


# ================================================================
# RELATION WITH DL
# ================================================================

print("\n" + "=" * 88)
print("MARKER CORRELATION WITH THE DIFFERENTIATION LABEL")
print("=" * 88)

for marker in [
    "OCT4_raw",
    "FOXA2_raw",
    "SOX17_raw",
]:

    pear = pearsonr(
        df["DL"],
        df[marker],
    ).statistic

    spear = spearmanr(
        df["DL"],
        df[marker],
    ).statistic

    print(
        f"{marker:12s} | "
        f"Pearson={pear:+.6f} | "
        f"Spearman={spear:+.6f}"
    )


# ================================================================
# PER DATASET TAG
#
# dataset_tag is descriptive only.
# We do NOT convert it to biological time.
# ================================================================

print("\n" + "=" * 88)
print("MEANS BY DATASET TAG")
print("=" * 88)

summary = (
    df
    .groupby("dataset_tag")
    .agg(
        n=("filename", "size"),
        cultures=("culture", "nunique"),
        DL=("DL", "mean"),
        OCT4=("OCT4_raw", "mean"),
        FOXA2=("FOXA2_raw", "mean"),
        SOX17=("SOX17_raw", "mean"),
    )
)

print(
    summary.round(6).to_string()
)


# ================================================================
# FOLD DISTRIBUTION
# ================================================================

print("\n" + "=" * 88)
print("TARGETS BY FOLD")
print("=" * 88)

fold_summary = (
    df
    .groupby("fold")
    .agg(
        n=("filename", "size"),
        cultures=("culture", "nunique"),
        DL=("DL", "mean"),
        OCT4=("OCT4_raw", "mean"),
        FOXA2=("FOXA2_raw", "mean"),
        SOX17=("SOX17_raw", "mean"),
    )
)

print(
    fold_summary.round(6).to_string()
)


# ================================================================
# FINAL TABLE
# ================================================================

keep = [
    "cache_index",
    "filename",
    "culture",
    "dataset_tag",
    "fold",
    "DL",
    "OCT4_raw",
    "FOXA2_raw",
    "SOX17_raw",
]

df = df[
    keep
].sort_values(
    "cache_index"
).reset_index(
    drop=True
)


if not np.array_equal(
    df["cache_index"].to_numpy(),
    np.arange(30877),
):
    raise RuntimeError(
        "cache_index is no longer aligned with the brightfield cache."
    )


df.to_csv(
    OUT_PATH,
    index=False,
)


# ================================================================
# OUTPUT HASH
# ================================================================

out_hash = sha256_file(
    OUT_PATH
)


print("\n" + "=" * 88)
print("FROZEN M2 TABLE")
print("=" * 88)

print(
    "Rows     :",
    len(df),
)

print(
    "Cultures :",
    df["culture"].nunique(),
)

print(
    "Folds    :",
    sorted(df["fold"].unique()),
)

print(
    "\nBiological mapping:"
)

print(
    "  ch0 -> OCT4"
)

print(
    "  ch1 -> FOXA2"
)

print(
    "  ch2 -> SOX17"
)

print(
    "\nNo biological time was inferred or fabricated."
)

print(
    "No marker normalization has "
    "been applied yet."
)

print(
    "\nFile:",
    OUT_PATH,
)

print(
    "SHA256  :",
    out_hash,
)

print("\nDONE")
