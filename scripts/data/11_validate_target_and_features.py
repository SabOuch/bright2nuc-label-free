"""
Audit Bright2Nuc static targets, handcrafted features, masks, and TF channels.

Purpose
-------
Validate the static Bright2Nuc inputs used during release preparation:
- count numeric morphology and brightfield-derived features;
- verify that ``SingleNuclei/`` can be used as a nuclear mask;
- reconstruct per-nucleus transcription-factor signals from ``SingleNucleiTF/``;
- test channel permutations against the published ``TF_class`` target;
- save diagnostic tables used by the M2 auxiliary-target preparation.

Inputs
------
- ``data/raw/TF_prediction/Positions_Classes.csv``
- ``data/raw/TF_prediction/Features.csv``
- ``data/raw/TF_prediction/Features_Brightfield.csv``
- ``data/raw/TF_prediction/SingleNuclei/``
- ``data/raw/TF_prediction/SingleNucleiTF/``

Outputs
-------
Written under ``outputs/data_audit/``:
- ``single_nuclei_mask_check.csv``
- ``tf_class_reconstruction_tests.csv``
- ``tf_expression_reconstruction.csv``

Notes
-----
This is an audit/preparation script, not a training script. Fluorescence-derived
signals are used only to reconstruct or audit supervision targets; they are
never used as model input at inference time.
"""

from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
TFROOT = ROOT / "data/raw/TF_prediction"

POS = TFROOT / "Positions_Classes.csv"
FEAT = TFROOT / "Features.csv"
FEAT_BF = TFROOT / "Features_Brightfield.csv"

NUC_DIR = TFROOT / "SingleNuclei"
TF_DIR = TFROOT / "SingleNucleiTF"

OUT = ROOT / "outputs/data_audit"
OUT.mkdir(parents=True, exist_ok=True)

pos = pd.read_csv(POS)
feat = pd.read_csv(FEAT)
bf = pd.read_csv(FEAT_BF)

print("=" * 80)
print(" VALIDATION OF THE TF_class TARGET AND THE 120 FEATURES")
print("=" * 80)

# ------------------------------------------------------------------
# 1. FEATURES
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("1. NUMBER OF FEATURES")
print("=" * 80)

feat_numeric = [
    c for c in feat.columns
    if c != "filename" and pd.api.types.is_numeric_dtype(feat[c])
]

bf_numeric = [
    c for c in bf.columns
    if c != "filename" and pd.api.types.is_numeric_dtype(bf[c])
]

print("Numeric Features.csv columns         :", len(feat_numeric))
print("Numeric Features_Brightfield columns :", len(bf_numeric))
print("Sum                                  :", len(feat_numeric) + len(bf_numeric))

if "boarder_distance" in pos.columns:
    print("+ boarder_distance                   : 1")
    print("Total including boarder_distance     :",
          len(feat_numeric) + len(bf_numeric) + 1)

print("\nFeatures.csv columns:")
for c in feat_numeric:
    print(" -", c)

print("\nFeatures_Brightfield.csv columns:")
for c in bf_numeric:
    print(" -", c)

# ------------------------------------------------------------------
# 2. IS THE SingleNuclei CROP A VALID USABLE MASK?
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("2. SingleNuclei MASK VERIFICATION")
print("=" * 80)

merged = pos[["filename"]].merge(
    feat[["filename", "volume"]],
    on="filename",
    how="inner"
)

# deterministic sample
sample = merged.iloc[
    np.linspace(0, len(merged)-1, 300, dtype=int)
].copy()

mask_rows = []

for _, row in sample.iterrows():

    name = row["filename"]
    img = tifffile.imread(NUC_DIR / name)

    n_nonzero = int(np.count_nonzero(img))
    volume = float(row["volume"])

    mask_rows.append((name, volume, n_nonzero))

mask_df = pd.DataFrame(
    mask_rows,
    columns=["filename", "feature_volume", "nonzero_voxels"]
)

corr = np.corrcoef(
    mask_df["feature_volume"],
    mask_df["nonzero_voxels"]
)[0, 1]

exact = np.isclose(
    mask_df["feature_volume"],
    mask_df["nonzero_voxels"]
).mean()

print(f"Volume / nonzero-voxel correlation    : {corr:.10f}")
print(f"Exact/numeric equality                : {exact*100:.2f} %")

print("\nExamples:")
print(mask_df.head(15).to_string(index=False))

mask_df.to_csv(
    OUT / "single_nuclei_mask_check.csv",
    index=False
)

# ------------------------------------------------------------------
# 3. EXTRACT THE THREE TF SIGNALS FOR ALL NUCLEI
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("3. OCT4 / FOXA2 / SOX17 EXTRACTION — THREE-CHANNEL TEST")
print("=" * 80)

print("Reading 30,877 crops...")
print("This may take a few minutes.")

names = pos["filename"].astype(str).tolist()

signals_masked = np.zeros((len(names), 3), dtype=np.float64)
signals_full = np.zeros((len(names), 3), dtype=np.float64)

for i, name in enumerate(names):

    nuc = tifffile.imread(NUC_DIR / name)
    tf = tifffile.imread(TF_DIR / name)

    if tf.shape != (16, 64, 64, 3):
        raise RuntimeError(
            f"Unexpected TF shape for {name}: {tf.shape}"
        )

    mask = nuc > 0

    # mean over the entire crop
    signals_full[i] = tf.reshape(-1, 3).mean(axis=0)

    # mean within the nucleus only
    if mask.any():
        signals_masked[i] = tf[mask].mean(axis=0)
    else:
        signals_masked[i] = np.nan

    if (i + 1) % 2500 == 0:
        print(f"  {i+1:6d} / {len(names)}")

target = pos["TF_class"].to_numpy(dtype=np.float64)

# ------------------------------------------------------------------
# 4. CHANNEL-PERMUTATION TEST
# ------------------------------------------------------------------

def normalize_percentile(x):
    """Scale one signal with global p1/p99 limits for this audit only."""
    lo = np.nanpercentile(x, 1)
    hi = np.nanpercentile(x, 99)

    y = (x - lo) / (hi - lo)

    # "between 0 and 1"
    y = np.clip(y, 0.0, 1.0)

    return y, lo, hi


def evaluate(signals, source_name):
    """Evaluate all TF-channel permutations against the reference target."""

    normalized = np.zeros_like(signals)
    limits = []

    for c in range(3):
        normalized[:, c], lo, hi = normalize_percentile(signals[:, c])
        limits.append((lo, hi))

    rows = []

    # permutation = OCT4, FOXA2, SOX17
    for perm in permutations(range(3)):

        oct4 = normalized[:, perm[0]]
        foxa2 = normalized[:, perm[1]]
        sox17 = normalized[:, perm[2]]

        denom = oct4 + (foxa2 + sox17) / 2

        dl = np.divide(
            (foxa2 + sox17) / 2,
            denom,
            out=np.full_like(denom, np.nan),
            where=denom != 0
        )

        valid = np.isfinite(dl) & np.isfinite(target)

        r = pearsonr(
            dl[valid],
            target[valid]
        ).statistic

        mae = np.mean(
            np.abs(dl[valid] - target[valid])
        )

        max_err = np.max(
            np.abs(dl[valid] - target[valid])
        )

        rows.append({
            "source": source_name,
            "OCT4_channel": perm[0],
            "FOXA2_channel": perm[1],
            "SOX17_channel": perm[2],
            "pearson_r": r,
            "MAE": mae,
            "max_abs_error": max_err,
            "valid_n": int(valid.sum()),
        })

    result = pd.DataFrame(rows).sort_values(
        ["pearson_r", "MAE"],
        ascending=[False, True]
    )

    return result, limits


masked_result, masked_limits = evaluate(
    signals_masked,
    "masked_nucleus"
)

full_result, full_limits = evaluate(
    signals_full,
    "whole_crop"
)

all_results = pd.concat(
    [masked_result, full_result],
    ignore_index=True
).sort_values(
    ["pearson_r", "MAE"],
    ascending=[False, True]
)

print("\n" + "=" * 80)
print("RESULTS — BEST MATCHES")
print("=" * 80)

print(
    all_results.head(12).to_string(
        index=False,
        float_format=lambda x: f"{x:.10f}"
    )
)

print("\nPercentiles masked nucleus :")
for c, (lo, hi) in enumerate(masked_limits):
    print(f" channel {c}: p1={lo:.6f}  p99={hi:.6f}")

print("\nPercentiles whole crop :")
for c, (lo, hi) in enumerate(full_limits):
    print(f" channel {c}: p1={lo:.6f}  p99={hi:.6f}")

all_results.to_csv(
    OUT / "tf_class_reconstruction_tests.csv",
    index=False
)

# ------------------------------------------------------------------
# 5. SAVE RAW EXPRESSION VALUES
# ------------------------------------------------------------------

expr = pd.DataFrame({
    "filename": names,
    "TF_class": target,

    "masked_ch0": signals_masked[:, 0],
    "masked_ch1": signals_masked[:, 1],
    "masked_ch2": signals_masked[:, 2],

    "full_ch0": signals_full[:, 0],
    "full_ch1": signals_full[:, 1],
    "full_ch2": signals_full[:, 2],
})

expr.to_csv(
    OUT / "tf_expression_reconstruction.csv",
    index=False
)

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(OUT / "single_nuclei_mask_check.csv")
print(OUT / "tf_class_reconstruction_tests.csv")
print(OUT / "tf_expression_reconstruction.csv")

print("\nDONE")
