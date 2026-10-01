"""
Run the frozen M4 DINOv2 + Ridge model on the common longitudinal support.

Purpose
-------
Apply the frozen DINOv2 representation and the five fold-specific Ridge probes
to the same 88,458 longitudinal spots used for M1-M3, then aggregate
predictions at spot, track, and culture × day levels.

The script first verifies that the existing spot→track→culture aggregation
reproduces the M1 longitudinal table before computing M4 predictions.

Inputs
------
- ``outputs/longitudinal/m1_longitudinal_spot_predictions.csv``
- ``outputs/longitudinal/m1_longitudinal_culture_day.csv``
- ``outputs/m4/fold{0..4}_ridge_probe.joblib``
- longitudinal brightfield TIFFs under ``BRIGHT2NUC_DATA_ROOT``.

Outputs
-------
Written under ``outputs/longitudinal/``:
- ``m4_longitudinal_spot_predictions.csv``
- ``m4_longitudinal_track_predictions.csv``
- ``m4_longitudinal_culture_day.csv``
- ``m4_longitudinal_delta_comparison.csv``

Methodological notes
--------------------
The longitudinal M4 sensitivity analysis uses slices 7, 8, and 9 as pseudo-RGB,
a frozen DINOv2 ViT-S backbone, and the previously fitted fold-specific Ridge
probes. No TTA or prediction clipping is applied.

The frozen longitudinal preprocessing is preserved here so that this script
reproduces the reported sensitivity-analysis results.
"""

from pathlib import Path
import os
import time

import cv2
import joblib
import numpy as np
import pandas as pd
import tifffile

import torch
import torch.nn.functional as F
import timm

from scipy.stats import spearmanr


# ======================================================================
# PATHS / CONFIG
# ======================================================================

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

M4_DIR = (
    ROOT
    / "outputs/m4"
)

OUT = (
    ROOT
    / "outputs/longitudinal"
)

OUT.mkdir(parents=True, exist_ok=True)

SOURCE_SPOTS = (
    OUT
    / "m1_longitudinal_spot_predictions.csv"
)

SPOT_OUT = (
    OUT
    / "m4_longitudinal_spot_predictions.csv"
)

TRACK_OUT = (
    OUT
    / "m4_longitudinal_track_predictions.csv"
)

CULTURE_OUT = (
    OUT
    / "m4_longitudinal_culture_day.csv"
)

COMPARE_OUT = (
    OUT
    / "m4_longitudinal_delta_comparison.csv"
)

MODEL_NAME = (
    "vit_small_patch14_dinov2.lvd142m"
)

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

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# LOAD COMMON SUPPORT
# ======================================================================

spots = pd.read_csv(
    SOURCE_SPOTS
)

if len(spots) != 88458:
    raise RuntimeError(
        f"Expected support: 88458, got: {len(spots)}"
    )


required = {
    "culture",
    "day",
    "day_number",
    "frame",
    "bf_t",
    "track_id",
    "crop_center_x",
    "crop_center_y",
    "crop_center_z",
    "crop_mean",
    "position_z",
    "m1_mean",
    "m2_mean",
    "m3_dl_mean",
}

missing = required - set(
    spots.columns
)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


spots = spots.reset_index(
    drop=True
)

print("=" * 100)
print("M4 LONGITUDINAL — FROZEN DINOv2 + RIDGE")
print("=" * 100)

print()
print(
    "Common-support spots:",
    len(spots),
)

print(
    "Cultures             :",
    spots["culture"].nunique(),
)

print(
    "Culture × day        :",
    spots[
        ["culture", "day"]
    ]
    .drop_duplicates()
    .shape[0],
)

print(
    "Device               :",
    DEVICE,
)

if DEVICE.type == "cuda":
    print(
        "GPU                  :",
        torch.cuda.get_device_name(0),
    )


# ======================================================================
# VERIFY EXISTING AGGREGATION
# ======================================================================

def track_then_culture(
    df,
    pred_col,
):
    """Aggregate one spot-level prediction column to track and culture × day."""

    track = (
        df
        .groupby(
            [
                "culture",
                "day",
                "day_number",
                "track_id",
            ],
            as_index=False,
        )
        .agg(
            prediction=(
                pred_col,
                "mean",
            ),
        )
    )

    culture = (
        track
        .groupby(
            [
                "culture",
                "day",
                "day_number",
            ],
            as_index=False,
        )
        .agg(
            prediction=(
                "prediction",
                "mean",
            ),
        )
    )

    return track, culture


_, check_m1 = (
    track_then_culture(
        spots,
        "m1_mean",
    )
)

m1_existing = pd.read_csv(
    OUT
    / "m1_longitudinal_culture_day.csv"
)

check = (
    check_m1
    .merge(
        m1_existing[
            [
                "culture",
                "day",
                "m1_mean",
            ]
        ],
        on=[
            "culture",
            "day",
        ],
        validate="one_to_one",
    )
)

aggregation_diff = float(
    np.max(
        np.abs(
            check["prediction"]
            - check["m1_mean"]
        )
    )
)


print()
print("=" * 100)
print("VALIDATION OF EXISTING AGGREGATION")
print("=" * 100)

print(
    "Max diff M1 "
    "spot→track→culture vs M1 file:",
    f"{aggregation_diff:.10f}",
)


if aggregation_diff > 1e-6:
    raise RuntimeError(
        "The reconstructed aggregation "
        "does not reproduce M1."
    )


print(
    "✅ Longitudinal aggregation confirmed."
)


# ======================================================================
# LOAD M4 PROBES
# ======================================================================

probes = []

for fold in range(5):

    path = (
        M4_DIR
        / f"fold{fold}_ridge_probe.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    probe = joblib.load(
        path
    )

    probes.append(
        probe
    )

    print(
        f"Fold {fold} loaded "
        f"| alpha={probe['alpha']}"
    )


# ======================================================================
# LOAD DINOv2
# ======================================================================

print()
print(
    "Loading DINOv2..."
)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
)

model.eval()

for p in model.parameters():
    p.requires_grad = False

model = model.to(
    DEVICE
)


mean = torch.tensor(
    [
        0.485,
        0.456,
        0.406,
    ],
    dtype=torch.float32,
    device=DEVICE,
).view(
    1,
    3,
    1,
    1,
)

std = torch.tensor(
    [
        0.229,
        0.224,
        0.225,
    ],
    dtype=torch.float32,
    device=DEVICE,
).view(
    1,
    3,
    1,
    1,
)


# ======================================================================
# STORAGE
# ======================================================================

N = len(spots)

pred_folds = np.full(
    (
        N,
        5,
    ),
    np.nan,
    dtype=np.float32,
)

crop_mean_rebuilt = np.full(
    N,
    np.nan,
    dtype=np.float32,
)


# ======================================================================
# EMBEDDING + RIDGE
# ======================================================================

def predict_batch(
    crops_rgb,
):
    """Embed one pseudo-RGB batch and apply the five frozen Ridge probes."""

    x = torch.from_numpy(
        np.asarray(
            crops_rgb,
            dtype=np.float32,
        )
    )

    x = (
        x
        / 255.0
    )

    x = x.to(
        DEVICE,
        non_blocking=True,
    )

    x = F.interpolate(
        x,
        size=(
            518,
            518,
        ),
        mode="bicubic",
        align_corners=False,
    )

    x = (
        x - mean
    ) / std


    with torch.inference_mode():

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                DEVICE.type == "cuda"
            ),
        ):

            feat = model(
                x
            )


    feat = (
        feat
        .float()
        .cpu()
        .numpy()
    )


    if feat.shape[1] != 384:
        raise RuntimeError(
            f"Unexpected embedding shape: {feat.shape}"
        )


    if not np.isfinite(
        feat
    ).all():
        raise RuntimeError(
            "NaN/Inf in embeddings."
        )


    preds = np.empty(
        (
            len(feat),
            5,
        ),
        dtype=np.float32,
    )


    for fold, probe in enumerate(
        probes
    ):

        z = (
            probe["scaler"]
            .transform(
                feat
            )
        )

        preds[
            :,
            fold
        ] = (
            probe["ridge"]
            .predict(
                z
            )
            .astype(
                np.float32
            )
        )


    return preds


# ======================================================================
# LONGITUDINAL INFERENCE
# ======================================================================

start_time = time.time()

groups_done = 0
total_groups = (
    spots
    .groupby(
        [
            "culture",
            "day",
            "frame",
        ]
    )
    .ngroups
)


for culture in CULTURES:

    for day in DAYS:

        sub = spots[
            (
                spots["culture"]
                == culture
            )
            &
            (
                spots["day"]
                == day
            )
        ]


        print()
        print("=" * 100)
        print(
            culture,
            day,
        )
        print("=" * 100)

        print(
            "Crops :",
            len(sub),
        )


        for frame, g in sub.groupby(
            "frame",
            sort=True,
        ):

            bf_t_values = (
                g["bf_t"]
                .unique()
            )


            if len(
                bf_t_values
            ) != 1:
                raise RuntimeError(
                    "Inconsistent bf_t."
                )


            bf_t = int(
                bf_t_values[0]
            )


            bf_path = (
                BF_DIR
                / (
                    f"{culture}_Live5min_"
                    f"{day}_t{bf_t}.tif"
                )
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
                    f"Unexpected BF shape: "
                    f"{bf.shape}"
                )


            zdim, h, w = bf.shape


            bf_half = np.empty(
                (
                    zdim,
                    h // 2,
                    w // 2,
                ),
                dtype=bf.dtype,
            )


            for z in range(
                zdim
            ):

                bf_half[z] = (
                    cv2.resize(
                        bf[z],
                        (
                            w // 2,
                            h // 2,
                        ),
                        interpolation=(
                            cv2.INTER_LINEAR
                        ),
                    )
                )


            group_indices = []
            group_rgb = []


            for i, row in g.iterrows():

                cz = int(
                    row[
                        "crop_center_z"
                    ]
                )

                cy = int(
                    row[
                        "crop_center_y"
                    ]
                )

                cx = int(
                    row[
                        "crop_center_x"
                    ]
                )


                crop = bf_half[
                    cz - 8:
                    cz + 8,

                    cy - 32:
                    cy + 32,

                    cx - 32:
                    cx + 32,
                ]


                if crop.shape != (
                    16,
                    64,
                    64,
                ):
                    raise RuntimeError(
                        f"Invalid crop "
                        f"{culture} {day} "
                        f"frame={frame} "
                        f"index={i} "
                        f"shape={crop.shape}"
                    )


                crop_mean_rebuilt[
                    i
                ] = float(
                    crop.mean()
                )


                rgb = np.asarray(
                    crop[
                        [7, 8, 9],
                        :,
                        :
                    ],
                    dtype=np.uint8,
                )


                group_indices.append(
                    int(i)
                )

                group_rgb.append(
                    rgb
                )


            group_indices = np.asarray(
                group_indices,
                dtype=np.int64,
            )

            group_rgb = np.stack(
                group_rgb,
                axis=0,
            )


            for start in range(
                0,
                len(group_indices),
                BATCH_SIZE,
            ):

                end = min(
                    start
                    + BATCH_SIZE,
                    len(group_indices),
                )

                ids = (
                    group_indices[
                        start:end
                    ]
                )

                batch = (
                    group_rgb[
                        start:end
                    ]
                )


                pred_folds[
                    ids
                ] = predict_batch(
                    batch
                )


            groups_done += 1


            if (
                groups_done % 25
                == 0
                or groups_done
                == total_groups
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                fraction = (
                    groups_done
                    / total_groups
                )

                eta = (
                    elapsed
                    * (
                        1.0
                        - fraction
                    )
                    / max(
                        fraction,
                        1e-9,
                    )
                )


                print(
                    f"  groups "
                    f"{groups_done}/"
                    f"{total_groups} "
                    f"| ETA "
                    f"{eta/60:.1f} min"
                )


        day_ids = (
            sub.index
            .to_numpy(
                dtype=np.int64
            )
        )

        day_pred = (
            pred_folds[
                day_ids
            ]
            .mean(
                axis=1
            )
        )

        print(
            "M4 median:",
            f"{np.median(day_pred):.6f}",
        )


# ======================================================================
# VALIDATION PREDICTIONS
# ======================================================================

if not np.isfinite(
    pred_folds
).all():
    raise RuntimeError(
        "Incomplete M4 predictions."
    )


if not np.isfinite(
    crop_mean_rebuilt
).all():
    raise RuntimeError(
        "Incomplete reconstructed crop_mean."
    )


crop_diff = np.abs(
    crop_mean_rebuilt
    - spots[
        "crop_mean"
    ].to_numpy(
        dtype=np.float32
    )
)


print()
print("=" * 100)
print("CROP VALIDATION")
print("=" * 100)

print(
    "Max diff crop_mean :",
    f"{crop_diff.max():.10f}",
)

print(
    "Mean diff crop_mean:",
    f"{crop_diff.mean():.10f}",
)


if crop_diff.max() > 1e-4:
    raise RuntimeError(
        "The M4 crops do not reproduce "
        "the M1/M2/M3 support."
    )


print(
    "✅ Longitudinal crops reproduced."
)


# ======================================================================
# SAVE SPOT PREDICTIONS
# ======================================================================

for fold in range(5):

    spots[
        f"m4_fold{fold}"
    ] = (
        pred_folds[
            :,
            fold
        ]
    )


spots[
    "m4_mean"
] = pred_folds.mean(
    axis=1
)

spots[
    "m4_sd_folds"
] = pred_folds.std(
    axis=1
)


spots.to_csv(
    SPOT_OUT,
    index=False,
)


# ======================================================================
# TRACK LEVEL
# ======================================================================

agg_dict = {
    "n_spots":
        (
            "frame",
            "size",
        ),

    "position_z_median":
        (
            "position_z",
            "median",
        ),

    "m1_mean":
        (
            "m1_mean",
            "mean",
        ),

    "m2_mean":
        (
            "m2_mean",
            "mean",
        ),

    "m3_mean":
        (
            "m3_dl_mean",
            "mean",
        ),

    "m4_mean":
        (
            "m4_mean",
            "mean",
        ),
}


for fold in range(5):

    agg_dict[
        f"m4_fold{fold}"
    ] = (
        f"m4_fold{fold}",
        "mean",
    )


tracks = (
    spots
    .groupby(
        [
            "culture",
            "day",
            "day_number",
            "track_id",
        ],
        as_index=False,
    )
    .agg(
        **agg_dict
    )
)


tracks.to_csv(
    TRACK_OUT,
    index=False,
)


# ======================================================================
# CULTURE × DAY
# ======================================================================

culture_day = (
    tracks
    .groupby(
        [
            "culture",
            "day",
            "day_number",
        ],
        as_index=False,
    )
    .agg(
        n_tracks=(
            "track_id",
            "size",
        ),

        m1=(
            "m1_mean",
            "mean",
        ),

        m2=(
            "m2_mean",
            "mean",
        ),

        m3=(
            "m3_mean",
            "mean",
        ),

        m4=(
            "m4_mean",
            "mean",
        ),
    )
    .sort_values(
        [
            "culture",
            "day_number",
        ]
    )
)


culture_day.to_csv(
    CULTURE_OUT,
    index=False,
)


print()
print("=" * 100)
print("M4 — CULTURE × DAY")
print("=" * 100)

print(
    culture_day[
        [
            "culture",
            "day",
            "n_tracks",
            "m4",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ======================================================================
# TRAJECTORIES
# ======================================================================

def trajectory_table(
    value_col,
):
    """Return per-culture day values together with the three day-to-day deltas."""

    p = (
        culture_day
        .pivot(
            index="culture",
            columns="day",
            values=value_col,
        )
        .loc[
            CULTURES,
            DAYS,
        ]
        .copy()
    )

    p[
        "D01-D00"
    ] = (
        p["Day01"]
        - p["Day00"]
    )

    p[
        "D02-D01"
    ] = (
        p["Day02"]
        - p["Day01"]
    )

    p[
        "D02-D00"
    ] = (
        p["Day02"]
        - p["Day00"]
    )

    return p


traj_m1 = trajectory_table(
    "m1"
)

traj_m2 = trajectory_table(
    "m2"
)

traj_m3 = trajectory_table(
    "m3"
)

traj_m4 = trajectory_table(
    "m4"
)


print()
print("=" * 100)
print("M4 TRAJECTORIES")
print("=" * 100)

print(
    traj_m4
    .round(6)
    .to_string()
)


# ======================================================================
# DELTA SUMMARY
# ======================================================================

comparisons = [
    "D01-D00",
    "D02-D01",
    "D02-D00",
]


rows = []


print()
print("=" * 100)
print("DELTAS M4")
print("=" * 100)


for comp in comparisons:

    d = (
        traj_m4[
            comp
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    n_neg = int(
        np.sum(
            d < 0
        )
    )

    n_pos = int(
        np.sum(
            d > 0
        )
    )

    n_zero = int(
        np.sum(
            d == 0
        )
    )


    print()
    print(comp)

    print(
        " Δ :",
        np.round(
            d,
            6,
        ).tolist(),
    )

    print(
        " mean    :",
        f"{d.mean():+.6f}",
    )

    print(
        " signs   :",
        f"{n_neg} neg / "
        f"{n_pos} pos / "
        f"{n_zero} zero",
    )


# ======================================================================
# CONCORDANCE OF M4 vs M1/M2/M3 CHANGES
# ======================================================================

model_traj = {
    "M1":
        traj_m1,

    "M2":
        traj_m2,

    "M3":
        traj_m3,
}


print()
print("=" * 100)
print("CONCORDANCE OF CHANGES — M4 vs M1/M2/M3")
print("=" * 100)


for comp in comparisons:

    print()
    print(comp)

    m4_delta = (
        traj_m4[
            comp
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    for name, table in (
        model_traj.items()
    ):

        other = (
            table[
                comp
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        r = spearmanr(
            m4_delta,
            other,
        )


        print(
            f" M4 vs {name} : "
            f"rho={r.statistic:+.4f} "
            f"p_descriptive="
            f"{r.pvalue:.4f}"
        )


        rows.append({
            "comparison":
                comp,

            "other_model":
                name,

            "rho":
                r.statistic,

            "p_descriptive":
                r.pvalue,
        })


pd.DataFrame(
    rows
).to_csv(
    COMPARE_OUT,
    index=False,
)


# ======================================================================
# FINAL FILES
# ======================================================================

elapsed = (
    time.time()
    - start_time
)


print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(
    SPOT_OUT
)

print(
    TRACK_OUT
)

print(
    CULTURE_OUT
)

print(
    COMPARE_OUT
)


print()
print(
    "Total time:",
    f"{elapsed/60:.2f} min",
)

print()
print(
    "✅ M4 LONGITUDINAL COMPLETE"
)

print(
    "Same support: 88458 spots"
)

print(
    "Same longitudinal crop as M1/M2/M3"
)

print(
    "No clipping, no TTA"
)

print()
print("DONE")
