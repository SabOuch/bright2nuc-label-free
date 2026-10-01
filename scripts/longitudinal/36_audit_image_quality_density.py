"""
Audit acquisition quality and cell-density proxies in the longitudinal data.

Purpose
-------
For each of the 18 culture × day acquisitions, quantify image brightness,
contrast, robust dynamic range, normalized focus metrics, spots per frame, and
XY nearest-neighbour distance. The script then compares changes in these
acquisition characteristics with changes in common-support M3 predictions.

Inputs
------
- ``outputs/longitudinal/frame_to_bf_mapping.csv``
- ``outputs/longitudinal/common_support_culture_day.csv``
- longitudinal brightfield TIFFs and tracking tables.

Outputs
-------
Written under ``outputs/longitudinal/``:
- ``longitudinal_image_quality_density.csv``
- ``longitudinal_qc_delta_correlations.csv``

Interpretation
--------------
Change-vs-change correlations use culture as the unit (n=6) and are descriptive
confounder diagnostics only.
"""

from pathlib import Path
import os

import cv2
import numpy as np
import pandas as pd
import tifffile

from scipy.spatial import cKDTree
from scipy.stats import spearmanr


# ============================================================
# PATHS
# ============================================================

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

MAP_PATH = (
    ROOT
    / "outputs"
    / "longitudinal"
    / "frame_to_bf_mapping.csv"
)

M3_PATH = (
    ROOT
    / "outputs"
    / "longitudinal"
    / "common_support_culture_day.csv"
)

OUT = (
    ROOT
    / "outputs"
    / "longitudinal"
)

OUT.mkdir(parents=True, exist_ok=True)

CULTURES = [
    "02A",
    "05D",
    "08A",
    "11D",
    "14A",
    "17D",
]

DAYS = [
    "Day00",
    "Day01",
    "Day02",
]


# ============================================================
# IMAGE QUALITY
# ============================================================

def slice_metrics(img):
    """Compute brightness, contrast, range, and normalized focus metrics."""

    # XY downsampling exactly as for the longitudinal crops.
    h, w = img.shape

    small = cv2.resize(
        img,
        (
            w // 2,
            h // 2,
        ),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    mean = float(
        small.mean()
    )

    std = float(
        small.std()
    )

    p05 = float(
        np.percentile(
            small,
            5,
        )
    )

    p95 = float(
        np.percentile(
            small,
            95,
        )
    )

    robust_range = (
        p95 - p05
    )


    # --------------------------------------------------------
    # Sharpness after z-scoring the plane.
    #
    # This greatly limits the effect of a simple difference
    # in brightness or contrast.
    # --------------------------------------------------------

    normalized = (
        small - mean
    ) / (
        std + 1e-6
    )


    gx = cv2.Sobel(
        normalized,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        normalized,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    tenengrad = float(
        np.mean(
            gx * gx
            + gy * gy
        )
    )


    lap = cv2.Laplacian(
        normalized,
        cv2.CV_32F,
        ksize=3,
    )

    lap_var = float(
        np.var(
            lap
        )
    )


    return {
        "mean":
            mean,

        "std":
            std,

        "range_p95_p05":
            robust_range,

        "tenengrad_norm":
            tenengrad,

        "laplacian_var_norm":
            lap_var,
    }


# ============================================================
# LOAD FRAME MAPPING + M3
# ============================================================

mapping = pd.read_csv(
    MAP_PATH
)

m3 = pd.read_csv(
    M3_PATH
)


rows = []


print("=" * 108)
print("AUDIT OF 18 ACQUISITIONS")
print("=" * 108)


for culture in CULTURES:

    for day in DAYS:

        print()
        print(
            f"--- {culture} {day} ---"
        )


        # ----------------------------------------------------
        # TRACKING
        # ----------------------------------------------------

        track_path = (
            TRACK_DIR
            / (
                f"{culture}_Live5min_{day}"
                "_Spots in tracks statistics.csv"
            )
        )

        tracks = pd.read_csv(
            track_path
        )


        group_map = (
            mapping[
                (mapping["culture"] == culture)
                &
                (mapping["day"] == day)
            ]
            .set_index(
                "frame"
            )
        )


        # ----------------------------------------------------
        # IMAGE QC
        # ----------------------------------------------------

        frame_mean = []
        frame_std = []
        frame_range = []
        frame_tenengrad = []
        frame_lap = []


        # ----------------------------------------------------
        # DENSITY QC
        # ----------------------------------------------------

        frame_nn_xy = []
        frame_n_spots = []


        for frame, frame_df in tracks.groupby(
            "FRAME"
        ):

            frame = int(
                frame
            )

            bf_t = int(
                group_map.loc[
                    frame,
                    "bf_t",
                ]
            )


            bf_path = (
                BF_DIR
                / (
                    f"{culture}_Live5min_"
                    f"{day}_t{bf_t}.tif"
                )
            )


            bf = tifffile.imread(
                bf_path
            )


            if bf.ndim != 3:

                raise RuntimeError(
                    f"{bf_path.name}: "
                    f"unexpected shape {bf.shape}"
                )


            zdim = bf.shape[0]


            # ------------------------------------------------
            # Seven evenly spaced Z planes.
            #
            # Sufficient to compare optical quality without
            # recomputing metrics over >20,000 planes.
            # ------------------------------------------------

            if zdim >= 12:

                z_indices = np.unique(
                    np.rint(
                        np.linspace(
                            4,
                            zdim - 5,
                            7,
                        )
                    ).astype(int)
                )

            else:

                z_indices = np.arange(
                    zdim
                )


            slice_rows = []


            for z in z_indices:

                q = slice_metrics(
                    bf[int(z)]
                )

                slice_rows.append(
                    q
                )


            qdf = pd.DataFrame(
                slice_rows
            )


            frame_mean.append(
                float(
                    qdf[
                        "mean"
                    ].median()
                )
            )

            frame_std.append(
                float(
                    qdf[
                        "std"
                    ].median()
                )
            )

            frame_range.append(
                float(
                    qdf[
                        "range_p95_p05"
                    ].median()
                )
            )

            frame_tenengrad.append(
                float(
                    qdf[
                        "tenengrad_norm"
                    ].median()
                )
            )

            frame_lap.append(
                float(
                    qdf[
                        "laplacian_var_norm"
                    ].median()
                )
            )


            # ------------------------------------------------
            # XY DENSITY
            #
            # Nearest-neighbor distance in the XY plane,
            # in original TrackMate pixels.
            #
            # Avoid introducing an assumption about
            # the Z scale here.
            # ------------------------------------------------

            xy = (
                frame_df[
                    [
                        "POSITION_X",
                        "POSITION_Y",
                    ]
                ]
                .dropna()
                .to_numpy(
                    dtype=np.float64
                )
            )


            frame_n_spots.append(
                len(xy)
            )


            if len(xy) >= 2:

                tree = cKDTree(
                    xy
                )

                distances, _ = (
                    tree.query(
                        xy,
                        k=2,
                    )
                )

                nn = (
                    distances[
                        :,
                        1,
                    ]
                )

                frame_nn_xy.append(
                    float(
                        np.median(
                            nn
                        )
                    )
                )


        row = {
            "culture":
                culture,

            "day":
                day,

            "n_frames":
                len(
                    frame_mean
                ),

            "n_spots":
                len(
                    tracks
                ),

            "n_tracks":
                int(
                    tracks[
                        "TRACK_ID"
                    ].nunique()
                ),

            "brightness_median":
                float(
                    np.median(
                        frame_mean
                    )
                ),

            "contrast_std_median":
                float(
                    np.median(
                        frame_std
                    )
                ),

            "dynamic_range_median":
                float(
                    np.median(
                        frame_range
                    )
                ),

            "tenengrad_norm_median":
                float(
                    np.median(
                        frame_tenengrad
                    )
                ),

            "laplacian_norm_median":
                float(
                    np.median(
                        frame_lap
                    )
                ),

            "spots_per_frame_median":
                float(
                    np.median(
                        frame_n_spots
                    )
                ),

            "nn_xy_px_median":
                float(
                    np.median(
                        frame_nn_xy
                    )
                ),
        }


        rows.append(
            row
        )


        print(
            "brightness =",
            f"{row['brightness_median']:.2f}",
            "| contrast =",
            f"{row['contrast_std_median']:.2f}",
            "| Tenengrad =",
            f"{row['tenengrad_norm_median']:.4f}",
            "| LapVar =",
            f"{row['laplacian_norm_median']:.4f}",
            "| NN xy =",
            f"{row['nn_xy_px_median']:.2f}",
        )


qc = pd.DataFrame(
    rows
)


# ============================================================
# ADD M3
# ============================================================

m3_small = (
    m3[
        [
            "culture",
            "day",
            "m3",
        ]
    ]
    .copy()
)


qc = qc.merge(
    m3_small,
    on=[
        "culture",
        "day",
    ],
    how="left",
    validate="one_to_one",
)


if qc["m3"].isna().any():

    raise RuntimeError(
        "M3 is missing after the merge."
    )


# ============================================================
# PRINT TABLE
# ============================================================

print()
print("=" * 108)
print("COMPLETE QC TABLE")
print("=" * 108)

print(
    qc[
        [
            "culture",
            "day",
            "m3",
            "brightness_median",
            "contrast_std_median",
            "dynamic_range_median",
            "tenengrad_norm_median",
            "laplacian_norm_median",
            "spots_per_frame_median",
            "nn_xy_px_median",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# CHANGE-vs-CHANGE CORRELATIONS
#
# Culture = unit.
# Descriptive only, n=6.
# ============================================================

metrics = [
    "brightness_median",
    "contrast_std_median",
    "dynamic_range_median",
    "tenengrad_norm_median",
    "laplacian_norm_median",
    "spots_per_frame_median",
    "nn_xy_px_median",
]


def delta(
    column,
    day_a,
    day_b,
):
    """Return per-culture acquisition change between two days."""

    p = (
        qc
        .pivot(
            index="culture",
            columns="day",
            values=column,
        )
        .loc[
            CULTURES,
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
        dtype=np.float64
    )


corr_rows = []


print()
print("=" * 108)
print("CORRELATIONS BETWEEN QC CHANGES AND ΔM3")
print("=" * 108)

print(
    "WARNING: n=6 cultures -> descriptive confounder analyses."
)


for day_a, day_b in [
    (
        "Day00",
        "Day01",
    ),
    (
        "Day01",
        "Day02",
    ),
    (
        "Day00",
        "Day02",
    ),
]:

    dm3 = delta(
        "m3",
        day_a,
        day_b,
    )


    print()
    print(
        f"{day_b} - {day_a}"
    )

    print(
        "ΔM3 :",
        [
            round(
                float(x),
                4,
            )
            for x in dm3
        ],
    )


    for metric in metrics:

        dqc = delta(
            metric,
            day_a,
            day_b,
        )


        result = spearmanr(
            dqc,
            dm3,
        )


        rho = float(
            result.statistic
        )

        p = float(
            result.pvalue
        )


        corr_rows.append({
            "comparison":
                f"{day_b}-{day_a}",

            "metric":
                metric,

            "rho":
                rho,

            "p_descriptive":
                p,
        })


        print(
            f"  {metric:26s} "
            f"rho={rho:+.4f} "
            f"p={p:.4f}"
        )


# ============================================================
# DAY PATTERNS
# ============================================================

print()
print("=" * 108)
print("MEDIAN ACROSS 6 CULTURES BY DAY")
print("=" * 108)

day_summary = (
    qc
    .groupby(
        "day",
        sort=False,
    )[
        [
            "m3",
            "brightness_median",
            "contrast_std_median",
            "dynamic_range_median",
            "tenengrad_norm_median",
            "laplacian_norm_median",
            "spots_per_frame_median",
            "nn_xy_px_median",
        ]
    ]
    .median()
)

print(
    day_summary.to_string()
)


# ============================================================
# SAVE
# ============================================================

qc_path = (
    OUT
    / "longitudinal_image_quality_density.csv"
)

corr_path = (
    OUT
    / "longitudinal_qc_delta_correlations.csv"
)

qc.to_csv(
    qc_path,
    index=False,
)

pd.DataFrame(
    corr_rows
).to_csv(
    corr_path,
    index=False,
)


print()
print("=" * 108)
print("FILES")
print("=" * 108)

print(
    qc_path
)

print(
    corr_path
)

print()
print("DONE")
