"""
Restrict the M3 longitudinal analysis to a common geometric support.

Purpose
-------
Remove spots whose crop centers fall outside the spatial region that is valid
for every longitudinal acquisition, then recompute track- and culture-level M3
and PC1 summaries on this common support.

Input
-----
- ``outputs/longitudinal/m3_longitudinal_spot_predictions.csv``

Output
------
- ``outputs/longitudinal/common_support_culture_day.csv``

Statistics
----------
For each longitudinal contrast, the script reports exact Wilcoxon, sign-test,
and exhaustive sign-flip results across the six cultures.

Notes
-----
The common crop-center bounds are fixed from the smallest observed acquisition
geometry after XY downsampling.
"""

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from scipy.stats import (
    wilcoxon,
    binomtest,
)


ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "outputs"
    / "longitudinal"
)

OUT.mkdir(parents=True, exist_ok=True)

SPOT_PATH = (
    OUT
    / "m3_longitudinal_spot_predictions.csv"
)


# ============================================================
# LOAD
# ============================================================

spots = pd.read_csv(
    SPOT_PATH
)

cultures = sorted(
    spots["culture"].unique()
)

print("=" * 100)
print("INITIAL DATA")
print("=" * 100)

print(
    "Spots :",
    len(spots),
)

print(
    "Cultures :",
    cultures,
)


# ============================================================
# COMMON GEOMETRIC SUPPORT
#
# Smallest volume after XY downsampling:
#
# Day00: approximately 51 × 430 × 430
# Day01/02: approximately 57 × 413 × 413
#
# For a 16 × 64 × 64 crop:
#
# common z: center 8 ... 43
# common y: center 32 ... 381
# common x: center 32 ... 381
#
# Use the centers already computed during inference.
# ============================================================

common = spots[
    (spots["crop_center_z"] >= 8)
    &
    (spots["crop_center_z"] <= 43)
    &
    (spots["crop_center_y"] >= 32)
    &
    (spots["crop_center_y"] <= 381)
    &
    (spots["crop_center_x"] >= 32)
    &
    (spots["crop_center_x"] <= 381)
].copy()


print()
print("=" * 100)
print("AFTER APPLYING COMMON SUPPORT")
print("=" * 100)

print(
    "Retained spots:",
    len(common),
)

print(
    "Fraction        :",
    f"{len(common)/len(spots):.2%}",
)


counts = (
    common
    .groupby(
        [
            "culture",
            "day",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reindex(
        columns=[
            "Day00",
            "Day01",
            "Day02",
        ]
    )
)

print()
print("Spots by culture/day:")
print(
    counts.to_string()
)


# ============================================================
# TRACK LEVEL
# ============================================================

tracks = (
    common
    .groupby(
        [
            "culture",
            "day",
            "track_id",
        ],
        as_index=False,
    )
    .agg(
        n_spots=(
            "m3_dl_mean",
            "size",
        ),

        m3=(
            "m3_dl_mean",
            "mean",
        ),

        pc1=(
            "pc1_z_mean",
            "mean",
        ),

        brightness=(
            "crop_mean",
            "mean",
        ),
    )
)


culture_day = (
    tracks
    .groupby(
        [
            "culture",
            "day",
        ],
        as_index=False,
    )
    .agg(
        n_tracks=(
            "track_id",
            "nunique",
        ),

        m3=(
            "m3",
            "mean",
        ),

        pc1=(
            "pc1",
            "mean",
        ),

        brightness=(
            "brightness",
            "mean",
        ),
    )
)


print()
print("=" * 100)
print("CULTURE × DAY — COMMON SUPPORT")
print("=" * 100)

print(
    culture_day.to_string(
        index=False
    )
)


# ============================================================
# HELPERS
# ============================================================

def get_delta(
    column,
    day_a,
    day_b,
):
    """Return per-culture change between two acquisition days."""

    p = (
        culture_day
        .pivot(
            index="culture",
            columns="day",
            values=column,
        )
        .loc[
            cultures,
            [
                day_a,
                day_b,
            ],
        ]
    )

    return (
        p[day_b]
        - p[day_a]
    ).to_numpy(
        dtype=float
    )


def signflip_p(delta):
    """Compute the exact two-sided sign-flip p-value for the mean delta."""

    delta = np.asarray(
        delta,
        dtype=float,
    )

    observed = abs(
        delta.mean()
    )

    values = []

    for signs in product(
        [-1.0, 1.0],
        repeat=len(delta),
    ):

        values.append(
            abs(
                np.mean(
                    delta
                    * np.asarray(signs)
                )
            )
        )

    return float(
        np.mean(
            np.asarray(values)
            >= observed - 1e-12
        )
    )


def report(
    name,
    delta,
):
    """Print exact paired small-sample statistics for one day contrast."""

    delta = np.asarray(
        delta
    )

    w = wilcoxon(
        delta,
        method="exact",
        alternative="two-sided",
    )

    sign = binomtest(
        int(
            np.sum(
                delta > 0
            )
        ),
        n=len(delta),
        p=0.5,
        alternative="two-sided",
    )


    print()
    print(name)

    print(
        " deltas :",
        [
            round(
                float(x),
                6,
            )
            for x in delta
        ]
    )

    print(
        " mean    :",
        f"{delta.mean():+.6f}",
    )

    print(
        " signs   :",
        f"{np.sum(delta < 0)} neg / "
        f"{np.sum(delta > 0)} pos",
    )

    print(
        " Wilcoxon exact :",
        f"p={w.pvalue:.6f}",
    )

    print(
        " Sign test      :",
        f"p={sign.pvalue:.6f}",
    )

    print(
        " Sign-flip mean :",
        f"p={signflip_p(delta):.6f}",
    )


# ============================================================
# M3
# ============================================================

print()
print("=" * 100)
print("M3 — COMMON SUPPORT")
print("=" * 100)

for a, b in [
    ("Day00", "Day01"),
    ("Day01", "Day02"),
    ("Day00", "Day02"),
]:

    report(
        f"{b}-{a}",
        get_delta(
            "m3",
            a,
            b,
        ),
    )


# ============================================================
# PC1
# ============================================================

print()
print("=" * 100)
print("PC1 — COMMON SUPPORT")
print("=" * 100)

for a, b in [
    ("Day00", "Day01"),
    ("Day01", "Day02"),
    ("Day00", "Day02"),
]:

    report(
        f"{b}-{a}",
        get_delta(
            "pc1",
            a,
            b,
        ),
    )


# ============================================================
# SAVE
# ============================================================

out_path = (
    OUT
    / "common_support_culture_day.csv"
)

culture_day.to_csv(
    out_path,
    index=False,
)

print()
print("=" * 100)
print("SAVED OUTPUT")
print("=" * 100)

print(
    out_path
)

print()
print("DONE")
