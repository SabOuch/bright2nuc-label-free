"""
Validate longitudinal crop construction against the static training domain.

Purpose
-------
Reconstruct longitudinal 16 × 64 × 64 brightfield crops for one representative
acquisition (05D, Day01), quantify border exclusions and intensity statistics,
and compare their pixel distribution with crops from the frozen M1-M3 training
cache.

External data
-------------
The longitudinal dataset root is read from ``BRIGHT2NUC_DATA_ROOT``; the
repository-local ``data/raw/`` directory is the fallback.

Inputs
------
- longitudinal brightfield TIFFs;
- TrackMate statistics for ``05D_Live5min_Day01``;
- ``data/processed/m1_bf16_uint8.npy``.

Outputs
-------
This is a diagnostic script: it prints validation statistics and does not write
a result table.

Notes
-----
XY is downsampled by a factor of two, Z is unchanged, no artificial padding is
introduced, and only exact 16 × 64 × 64 crops are retained.
"""

from pathlib import Path
import os

import cv2
import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[2]

ZENODO = Path(
    os.environ.get(
        "BRIGHT2NUC_DATA_ROOT",
        ROOT / "data" / "raw",
    )
)

BF_DIR = (
    ZENODO
    / "Single_cell_velocities_Brightfield_3D"
    / "Brightfield_3D"
    / "test"
    / "image"
)

TRACK_DIR = (
    ZENODO
    / "Single_cell_velocities_Tracking"
    / "Tracking"
)

CSV_PATH = (
    TRACK_DIR
    / "05D_Live5min_Day01_Spots in tracks statistics.csv"
)

TRAIN_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "m1_bf16_uint8.npy"
)


df = pd.read_csv(CSV_PATH)

print("=" * 88)
print("SOURCE")
print("=" * 88)

print("Tracking rows:", len(df))
print("Tracks          :", df["TRACK_ID"].nunique())
print(
    "Frames          :",
    int(df["FRAME"].min()),
    "->",
    int(df["FRAME"].max()),
)


kept = 0
excluded_border = 0

crop_means = []
crop_stds = []

pixel_samples = []

example_shapes = set()
example_dtypes = set()

centers = []


for frame, frame_df in df.groupby("FRAME"):

    frame = int(frame)

    bf_path = (
        BF_DIR
        / f"05D_Live5min_Day01_t{frame}.tif"
    )

    if not bf_path.exists():
        raise FileNotFoundError(
            bf_path
        )

    bf = tifffile.imread(
        bf_path
    )

    if bf.ndim != 3:
        raise RuntimeError(
            f"Unexpected BF shape: {bf.shape}"
        )

    zdim, height, width = bf.shape

    # --------------------------------------------------------
    # 2× XY downsampling:
    # 0.25 µm/px -> 0.5 µm/px.
    #
    # Z is NOT resized.
    # --------------------------------------------------------

    new_w = width // 2
    new_h = height // 2

    bf_half = np.empty(
        (
            zdim,
            new_h,
            new_w,
        ),
        dtype=np.uint8,
    )

    for z in range(zdim):

        bf_half[z] = cv2.resize(
            bf[z],
            (
                new_w,
                new_h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )


    for _, row in frame_df.iterrows():

        # TrackMate :
        # POSITION_X = X
        # POSITION_Y = Y
        # POSITION_Z = Z
        #
        # XY coordinates must follow the 2× downsampling.

        cx = int(
            np.rint(
                float(row["POSITION_X"])
                / 2.0
            )
        )

        cy = int(
            np.rint(
                float(row["POSITION_Y"])
                / 2.0
            )
        )

        cz = int(
            np.rint(
                float(row["POSITION_Z"])
            )
        )


        z0 = cz - 8
        z1 = cz + 8

        y0 = cy - 32
        y1 = cy + 32

        x0 = cx - 32
        x1 = cx + 32


        # ----------------------------------------------------
        # IMPORTANT :
        # No artificial padding.
        #
        # Reproduce the domain of the published crops.
        # ----------------------------------------------------

        if (
            z0 < 0
            or y0 < 0
            or x0 < 0
            or z1 > bf_half.shape[0]
            or y1 > bf_half.shape[1]
            or x1 > bf_half.shape[2]
        ):

            excluded_border += 1
            continue


        crop = bf_half[
            z0:z1,
            y0:y1,
            x0:x1,
        ]


        if crop.shape != (
            16,
            64,
            64,
        ):

            raise RuntimeError(
                f"Invalid crop: {crop.shape}"
            )


        kept += 1

        example_shapes.add(
            crop.shape
        )

        example_dtypes.add(
            str(crop.dtype)
        )

        crop_means.append(
            float(crop.mean())
        )

        crop_stds.append(
            float(crop.std())
        )

        centers.append(
            (
                cz,
                cy,
                cx,
            )
        )


        # A few crops are sufficient to
        # compare the pixel distribution.

        if len(pixel_samples) < 500:

            pixel_samples.append(
                crop.ravel()
            )


print()
print("=" * 88)
print("CROPS LONGITUDINAUX")
print("=" * 88)

print(
    "Total spots       :",
    len(df),
)

print(
    "Retained crops    :",
    kept,
)

print(
    "Border exclusions :",
    excluded_border,
)

print(
    "Retained fraction :",
    f"{kept / len(df):.3%}",
)

print(
    "Observed shapes   :",
    example_shapes,
)

print(
    "Observed dtypes   :",
    example_dtypes,
)


crop_means = np.asarray(
    crop_means,
    dtype=np.float64,
)

crop_stds = np.asarray(
    crop_stds,
    dtype=np.float64,
)


print()
print("Mean intensity per crop:")

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
        f"  p{p:02d} :",
        f"{np.percentile(crop_means, p):.3f}",
    )


print()
print("Standard deviation per crop:")

for p in [
    1,
    5,
    50,
    95,
    99,
]:

    print(
        f"  p{p:02d} :",
        f"{np.percentile(crop_stds, p):.3f}",
    )


pixels = np.concatenate(
    pixel_samples
)

print()
print("Longitudinal pixels (up to 500 crops):")

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
        f"  p{p:02d} :",
        f"{np.percentile(pixels, p):.3f}",
    )


# ============================================================
# Comparison with the actual crops used to train M3
# ============================================================

print()
print("=" * 88)
print("REFERENCE — M3 TRAINING CACHE")
print("=" * 88)


cache = np.load(
    TRAIN_CACHE,
    mmap_mode="r",
)

print(
    "Cache shape :",
    cache.shape,
)

print(
    "Cache dtype :",
    cache.dtype,
)


rng = np.random.default_rng(
    42
)

n_sample = min(
    500,
    len(cache),
)

idx = rng.choice(
    len(cache),
    size=n_sample,
    replace=False,
)


train_means = []
train_pixels = []


for i in idx:

    crop = np.asarray(
        cache[int(i)]
    )

    train_means.append(
        float(crop.mean())
    )

    train_pixels.append(
        crop.ravel()
    )


train_means = np.asarray(
    train_means
)

train_pixels = np.concatenate(
    train_pixels
)


print()
print("Mean per training crop:")

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
        f"  p{p:02d} :",
        f"{np.percentile(train_means, p):.3f}",
    )


print()
print("Training pixels:")

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
        f"  p{p:02d} :",
        f"{np.percentile(train_pixels, p):.3f}",
    )


print()
print("=" * 88)
print("VALIDATION COMPLETE")
print("=" * 88)
