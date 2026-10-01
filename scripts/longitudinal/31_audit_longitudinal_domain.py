"""
Audit longitudinal crop retention and domain shift across all 18 acquisitions.

Purpose
-------
Rebuild valid longitudinal crops for six cultures × three days using the frozen
frame mapping, summarize crop intensity and contrast statistics, quantify border
retention, and compare brightness with the static M1-M3 training cache.

Inputs
------
- ``outputs/longitudinal/frame_to_bf_mapping.csv``
- longitudinal brightfield TIFFs and tracking tables;
- ``data/processed/m1_bf16_uint8.npy``.

Output
------
- ``outputs/longitudinal/longitudinal_domain_summary.csv``

Interpretation
--------------
The resulting brightness ratios are descriptive domain-comparison diagnostics;
they are not used to recalibrate the predictive models.
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

MAP_PATH = (
    ROOT
    / "outputs"
    / "longitudinal"
    / "frame_to_bf_mapping.csv"
)

CACHE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "m1_bf16_uint8.npy"
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

DAY_NUMBER = {
    "Day00": 0,
    "Day01": 1,
    "Day02": 2,
}


mapping = pd.read_csv(
    MAP_PATH
)

rows = []


print("=" * 100)
print("AUDIT OF 6 CULTURES × 3 DAYS")
print("=" * 100)


for culture in CULTURES:

    for day in DAYS:

        print()
        print(
            f"--- {culture} {day} ---"
        )

        csv_path = (
            TRACK_DIR
            / (
                f"{culture}_Live5min_{day}"
                "_Spots in tracks statistics.csv"
            )
        )

        df = pd.read_csv(
            csv_path
        )

        group_map = (
            mapping[
                (mapping["culture"] == culture)
                &
                (mapping["day"] == day)
            ]
            .set_index("frame")
        )


        n_total = len(df)

        valid_count = 0

        valid_tracks = set()

        crop_means = []

        crop_stds = []

        frame_means = []


        for frame, frame_df in df.groupby(
            "FRAME"
        ):

            frame = int(frame)

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

            zdim, height, width = (
                bf.shape
            )

            new_h = height // 2
            new_w = width // 2

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


            frame_means.append(
                float(
                    bf_half.mean()
                )
            )


            for _, spot in frame_df.iterrows():

                cx = int(
                    np.rint(
                        float(
                            spot[
                                "POSITION_X"
                            ]
                        )
                        / 2.0
                    )
                )

                cy = int(
                    np.rint(
                        float(
                            spot[
                                "POSITION_Y"
                            ]
                        )
                        / 2.0
                    )
                )

                cz = int(
                    np.rint(
                        float(
                            spot[
                                "POSITION_Z"
                            ]
                        )
                    )
                )


                z0 = cz - 8
                z1 = cz + 8

                y0 = cy - 32
                y1 = cy + 32

                x0 = cx - 32
                x1 = cx + 32


                if (
                    z0 < 0
                    or y0 < 0
                    or x0 < 0
                    or z1 > bf_half.shape[0]
                    or y1 > bf_half.shape[1]
                    or x1 > bf_half.shape[2]
                ):
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
                        f"Invalid crop: "
                        f"{crop.shape}"
                    )


                valid_count += 1

                valid_tracks.add(
                    int(
                        spot[
                            "TRACK_ID"
                        ]
                    )
                )

                crop_means.append(
                    float(
                        crop.mean()
                    )
                )

                crop_stds.append(
                    float(
                        crop.std()
                    )
                )


        crop_means = np.asarray(
            crop_means,
            dtype=np.float64,
        )

        crop_stds = np.asarray(
            crop_stds,
            dtype=np.float64,
        )

        frame_means = np.asarray(
            frame_means,
            dtype=np.float64,
        )


        row = {
            "culture":
                culture,

            "day":
                day,

            "day_number":
                DAY_NUMBER[day],

            "n_spots":
                n_total,

            "n_valid":
                valid_count,

            "retention":
                valid_count
                / n_total,

            "n_tracks":
                int(
                    df[
                        "TRACK_ID"
                    ].nunique()
                ),

            "n_tracks_with_valid_crop":
                len(
                    valid_tracks
                ),

            "frame_mean_median":
                float(
                    np.median(
                        frame_means
                    )
                ),

            "crop_mean_p05":
                float(
                    np.percentile(
                        crop_means,
                        5,
                    )
                ),

            "crop_mean_median":
                float(
                    np.median(
                        crop_means
                    )
                ),

            "crop_mean_p95":
                float(
                    np.percentile(
                        crop_means,
                        95,
                    )
                ),

            "crop_std_median":
                float(
                    np.median(
                        crop_stds
                    )
                ),
        }

        rows.append(
            row
        )

        print(
            "spots =",
            n_total,
            "| valid =",
            valid_count,
            f"({row['retention']:.1%})",
            "| crop mean median =",
            f"{row['crop_mean_median']:.2f}",
        )


summary = pd.DataFrame(
    rows
)


# ============================================================
# TRAINING REFERENCE
# ============================================================

cache = np.load(
    CACHE_PATH,
    mmap_mode="r",
)

rng = np.random.default_rng(
    42
)

idx = rng.choice(
    len(cache),
    size=min(
        1000,
        len(cache),
    ),
    replace=False,
)

train_means = np.asarray(
    [
        float(
            np.asarray(
                cache[int(i)]
            ).mean()
        )
        for i in idx
    ],
    dtype=np.float64,
)

train_median = float(
    np.median(
        train_means
    )
)

summary[
    "brightness_ratio_vs_train"
] = (
    summary[
        "crop_mean_median"
    ]
    / train_median
)


out_path = (
    OUT
    / "longitudinal_domain_summary.csv"
)

summary.to_csv(
    out_path,
    index=False,
)


print()
print("=" * 100)
print("COMPLETE TABLE")
print("=" * 100)

print(
    summary[
        [
            "culture",
            "day",
            "n_spots",
            "n_valid",
            "retention",
            "n_tracks",
            "n_tracks_with_valid_crop",
            "crop_mean_median",
            "crop_std_median",
            "brightness_ratio_vs_train",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("TRAINING REFERENCE")
print("=" * 100)

print(
    "Median mean/crop:",
    f"{train_median:.3f}",
)

print(
    "p05 :",
    f"{np.percentile(train_means, 5):.3f}",
)

print(
    "p95 :",
    f"{np.percentile(train_means, 95):.3f}",
)


print()
print("=" * 100)
print("BY DAY — MEDIAN ACROSS 6 CULTURES")
print("=" * 100)

day_summary = (
    summary
    .groupby(
        "day",
        sort=False,
    )
    .agg(
        crop_mean_median=(
            "crop_mean_median",
            "median",
        ),
        brightness_ratio=(
            "brightness_ratio_vs_train",
            "median",
        ),
        retention=(
            "retention",
            "mean",
        ),
    )
)

print(
    day_summary.to_string()
)


print()
print("=" * 100)
print("INTENSITY VARIATION WITHIN EACH CULTURE")
print("=" * 100)

pivot = summary.pivot(
    index="culture",
    columns="day",
    values="crop_mean_median",
)

pivot[
    "D01-D00"
] = (
    pivot["Day01"]
    - pivot["Day00"]
)

pivot[
    "D02-D01"
] = (
    pivot["Day02"]
    - pivot["Day01"]
)

print(
    pivot.to_string()
)


print()
print(
    "Saved:",
    out_path,
)

print()
print("=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)
