"""
Run frozen M3 models on the longitudinal Bright2Nuc acquisitions.

Purpose
-------
Reconstruct the exact longitudinal 16 × 64 × 64 brightfield crops, apply the
five frozen M3 fold models with TTA8, aggregate predictions across folds, and
summarize predictions at spot, track, and culture × day levels.

The script also projects M3 embeddings onto the frozen per-fold PC1 calibration
used in the longitudinal sensitivity analysis.

Inputs
------
- ``outputs/longitudinal/frame_to_bf_mapping.csv``
- ``outputs/m3/fold{0..4}/best_model.pt``
- ``outputs/m3/latent_pc1/fold{0..4}_pc1_calibrator.npz``
- longitudinal brightfield TIFFs and tracking tables.

Outputs
-------
Written under ``outputs/longitudinal/``:
- per-acquisition M3 prediction CSVs;
- ``m3_longitudinal_spot_predictions.csv``
- ``m3_longitudinal_track_predictions.csv``
- ``m3_longitudinal_culture_day.csv``
- ``m3_longitudinal_brightness_sensitivity.csv``

Notes
-----
Predictions are averaged across the five frozen culture-wise fold models.
Track-level aggregation gives each track equal weight before culture × day
summaries are computed.
"""

from pathlib import Path
import os
import gc

import cv2
import numpy as np
import pandas as pd
import tifffile

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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

M3_DIR = (
    ROOT
    / "outputs"
    / "m3"
)

CAL_DIR = (
    M3_DIR
    / "latent_pc1"
)

OUT = (
    ROOT
    / "outputs"
    / "longitudinal"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
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

DAY_NUMBER = {
    "Day00": 0,
    "Day01": 1,
    "Day02": 2,
}

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

class M3ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny architecture matching the frozen M3 checkpoints."""

    def __init__(self):

        super().__init__()

        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            in_chans=16,
            num_classes=0,
            global_pool="avg",
        )

        self.n_features = int(
            self.encoder.num_features
        )

        self.head_dl = nn.Linear(
            self.n_features,
            1,
        )

        self.head_markers = nn.Linear(
            self.n_features,
            3,
        )


    def encode(self, x):
        """Resize a 16-channel crop and return the pooled encoder embedding."""

        x = F.interpolate(
            x,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )

        return self.encoder(x)


# ============================================================
# TTA ×8
# ============================================================

@torch.no_grad()
def infer_tta8(
    model,
    crops,
    bf_mean,
    bf_std,
):
    """Return TTA8-averaged differentiation predictions and embeddings."""

    model.eval()

    all_dl = []
    all_emb = []

    n = len(crops)

    for start in range(
        0,
        n,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            n,
        )

        x = torch.from_numpy(
            crops[start:end]
        )

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        x = (
            x.float()
            / 255.0
        )

        x = (
            x - bf_mean
        ) / bf_std


        dl_tta = []
        emb_tta = []


        for k in range(4):

            xr = torch.rot90(
                x,
                k=k,
                dims=(-2, -1),
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    DEVICE.type == "cuda"
                ),
            ):

                emb = model.encode(
                    xr
                )

                dl = (
                    model.head_dl(
                        emb
                    )
                    .squeeze(1)
                )


            emb_tta.append(
                emb.float()
            )

            dl_tta.append(
                dl.float()
            )


            xf = torch.flip(
                xr,
                dims=(-1,),
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    DEVICE.type == "cuda"
                ),
            ):

                emb = model.encode(
                    xf
                )

                dl = (
                    model.head_dl(
                        emb
                    )
                    .squeeze(1)
                )


            emb_tta.append(
                emb.float()
            )

            dl_tta.append(
                dl.float()
            )


        dl_mean = (
            torch.stack(
                dl_tta,
                dim=0,
            )
            .mean(dim=0)
        )

        emb_mean = (
            torch.stack(
                emb_tta,
                dim=0,
            )
            .mean(dim=0)
        )


        all_dl.append(
            dl_mean
            .cpu()
            .numpy()
        )

        all_emb.append(
            emb_mean
            .cpu()
            .numpy()
        )


    return (
        np.concatenate(
            all_dl
        ),
        np.concatenate(
            all_emb
        ),
    )


# ============================================================
# LOAD THE FIVE FROZEN MODELS + PC1 CALIBRATORS
# ============================================================

print("=" * 100)
print("M3 LONGITUDINAL INFERENCE")
print("=" * 100)

print(
    "Device :",
    DEVICE,
)

if DEVICE.type == "cuda":

    print(
        "GPU    :",
        torch.cuda.get_device_name(0),
    )


models = []
calibrators = []


for fold in range(5):

    checkpoint_path = (
        M3_DIR
        / f"fold{fold}"
        / "best_model.pt"
    )

    calibrator_path = (
        CAL_DIR
        / f"fold{fold}_pc1_calibrator.npz"
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    if not calibrator_path.exists():

        raise FileNotFoundError(
            calibrator_path
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = (
        M3ConvNeXtTiny()
        .to(DEVICE)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()


    bf_mean = float(
        checkpoint[
            "bf_mean"
        ]
    )

    bf_std = float(
        checkpoint[
            "bf_std"
        ]
    )


    cal = np.load(
        calibrator_path
    )

    calibrator = {
        "center":
            cal[
                "center"
            ].astype(
                np.float32
            ),

        "component":
            cal[
                "component"
            ].astype(
                np.float32
            ),

        "score_mean":
            float(
                cal[
                    "score_mean"
                ]
            ),

        "score_std":
            float(
                cal[
                    "score_std"
                ]
            ),

        "bf_mean":
            bf_mean,

        "bf_std":
            bf_std,
    }


    models.append(
        model
    )

    calibrators.append(
        calibrator
    )


    print(
        f"Fold {fold} loaded | "
        f"BF mean={bf_mean:.6f} "
        f"std={bf_std:.6f}"
    )


# ============================================================
# FRAME MAPPING
# ============================================================

mapping = pd.read_csv(
    MAP_PATH
)


# ============================================================
# BUILD ONE ACQUISITION
# ============================================================

def build_acquisition(
    culture,
    day,
):
    """Build valid crops and metadata for one culture × day acquisition."""

    csv_path = (
        TRACK_DIR
        / (
            f"{culture}_Live5min_{day}"
            "_Spots in tracks statistics.csv"
        )
    )

    tracks = pd.read_csv(
        csv_path
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


    crops = []

    metadata = []


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

        zdim, height, width = (
            bf.shape
        )


        new_h = (
            height // 2
        )

        new_w = (
            width // 2
        )


        bf_half = np.empty(
            (
                zdim,
                new_h,
                new_w,
            ),
            dtype=np.uint8,
        )


        for z in range(
            zdim
        ):

            bf_half[z] = cv2.resize(
                bf[z],
                (
                    new_w,
                    new_h,
                ),
                interpolation=cv2.INTER_LINEAR,
            )


        frame_mean = float(
            bf_half.mean()
        )


        for _, spot in frame_df.iterrows():

            x_original = float(
                spot[
                    "POSITION_X"
                ]
            )

            y_original = float(
                spot[
                    "POSITION_Y"
                ]
            )

            z_original = float(
                spot[
                    "POSITION_Z"
                ]
            )


            cx = int(
                np.rint(
                    x_original
                    / 2.0
                )
            )

            cy = int(
                np.rint(
                    y_original
                    / 2.0
                )
            )

            cz = int(
                np.rint(
                    z_original
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


            crops.append(
                crop.copy()
            )


            metadata.append({
                "culture":
                    culture,

                "day":
                    day,

                "day_number":
                    DAY_NUMBER[
                        day
                    ],

                "frame":
                    frame,

                "bf_t":
                    bf_t,

                "track_id":
                    int(
                        spot[
                            "TRACK_ID"
                        ]
                    ),

                "position_x":
                    x_original,

                "position_y":
                    y_original,

                "position_z":
                    z_original,

                "crop_center_x":
                    cx,

                "crop_center_y":
                    cy,

                "crop_center_z":
                    cz,

                "frame_mean":
                    frame_mean,

                "crop_mean":
                    float(
                        crop.mean()
                    ),

                "crop_std":
                    float(
                        crop.std()
                    ),
            })


    if len(crops) == 0:

        raise RuntimeError(
            f"No crop for "
            f"{culture} {day}"
        )


    return (
        np.stack(
            crops,
            axis=0,
        ),
        pd.DataFrame(
            metadata
        ),
        len(tracks),
    )


# ============================================================
# FULL INFERENCE
# ============================================================

all_parts = []


for culture in CULTURES:

    for day in DAYS:

        print()
        print("=" * 100)
        print(
            culture,
            day,
        )
        print("=" * 100)


        crops, meta, n_total = (
            build_acquisition(
                culture,
                day,
            )
        )


        print(
            "Tracking spots:",
            n_total,
        )

        print(
            "Valid crops   :",
            len(crops),
        )

        print(
            "Retention     :",
            f"{len(crops)/n_total:.1%}",
        )

        print(
            "Median crop mean:",
            f"{meta['crop_mean'].median():.3f}",
        )


        fold_dl = []
        fold_pc1 = []


        for fold in range(5):

            print(
                f"  Fold {fold}..."
            )


            cal = (
                calibrators[
                    fold
                ]
            )


            dl, emb = infer_tta8(
                models[
                    fold
                ],
                crops,
                cal[
                    "bf_mean"
                ],
                cal[
                    "bf_std"
                ],
            )


            if emb.shape[1] != 768:

                raise RuntimeError(
                    f"Embedding fold "
                    f"{fold}: "
                    f"{emb.shape}"
                )


            pc1 = (
                (
                    emb
                    - cal[
                        "center"
                    ]
                )
                @ cal[
                    "component"
                ]
            )


            pc1_z = (
                pc1
                - cal[
                    "score_mean"
                ]
            ) / cal[
                "score_std"
            ]


            meta[
                f"m3_dl_fold{fold}"
            ] = dl

            meta[
                f"pc1_z_fold{fold}"
            ] = pc1_z


            fold_dl.append(
                dl
            )

            fold_pc1.append(
                pc1_z
            )


            del dl
            del emb
            del pc1
            del pc1_z

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


        fold_dl = np.stack(
            fold_dl,
            axis=1,
        )

        fold_pc1 = np.stack(
            fold_pc1,
            axis=1,
        )


        meta[
            "m3_dl_mean"
        ] = fold_dl.mean(
            axis=1
        )

        meta[
            "m3_dl_sd"
        ] = fold_dl.std(
            axis=1,
            ddof=0,
        )

        meta[
            "pc1_z_mean"
        ] = fold_pc1.mean(
            axis=1
        )

        meta[
            "pc1_z_sd"
        ] = fold_pc1.std(
            axis=1,
            ddof=0,
        )


        acquisition_path = (
            OUT
            / (
                f"{culture}_{day}"
                "_m3_predictions.csv"
            )
        )

        meta.to_csv(
            acquisition_path,
            index=False,
        )


        print(
            "Median M3 DL  :",
            f"{meta['m3_dl_mean'].median():.4f}",
        )

        print(
            "Median PC1 z  :",
            f"{meta['pc1_z_mean'].median():.4f}",
        )


        all_parts.append(
            meta
        )


        del crops
        del meta
        del fold_dl
        del fold_pc1

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# COMBINE SPOTS
# ============================================================

spots = pd.concat(
    all_parts,
    ignore_index=True,
)

spot_path = (
    OUT
    / "m3_longitudinal_spot_predictions.csv"
)

spots.to_csv(
    spot_path,
    index=False,
)


# ============================================================
# TRACK LEVEL
#
# Each track is counted once.
# This prevents a long track from having artificially more
# weight than a short track.
# ============================================================

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
        n_valid_spots=(
            "m3_dl_mean",
            "size",
        ),

        m3_dl=(
            "m3_dl_mean",
            "mean",
        ),

        m3_dl_sd_folds=(
            "m3_dl_sd",
            "mean",
        ),

        pc1_z=(
            "pc1_z_mean",
            "mean",
        ),

        pc1_z_sd_folds=(
            "pc1_z_sd",
            "mean",
        ),

        crop_mean=(
            "crop_mean",
            "mean",
        ),

        crop_std=(
            "crop_std",
            "mean",
        ),
    )
)


track_path = (
    OUT
    / "m3_longitudinal_track_predictions.csv"
)

tracks.to_csv(
    track_path,
    index=False,
)


# ============================================================
# CULTURE × DAY
# ============================================================

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
            "nunique",
        ),

        m3_dl_mean=(
            "m3_dl",
            "mean",
        ),

        m3_dl_median=(
            "m3_dl",
            "median",
        ),

        pc1_z_mean=(
            "pc1_z",
            "mean",
        ),

        pc1_z_median=(
            "pc1_z",
            "median",
        ),

        crop_mean=(
            "crop_mean",
            "mean",
        ),
    )
)


culture_day_path = (
    OUT
    / "m3_longitudinal_culture_day.csv"
)

culture_day.to_csv(
    culture_day_path,
    index=False,
)


# ============================================================
# BRIGHTNESS SENSITIVITY
# ============================================================

brightness_rows = []


for (
    culture,
    day
), group in tracks.groupby(
    [
        "culture",
        "day",
    ]
):

    if len(group) >= 3:

        r_dl = spearmanr(
            group[
                "crop_mean"
            ],
            group[
                "m3_dl"
            ],
        ).statistic

        r_pc1 = spearmanr(
            group[
                "crop_mean"
            ],
            group[
                "pc1_z"
            ],
        ).statistic

    else:

        r_dl = np.nan
        r_pc1 = np.nan


    brightness_rows.append({
        "culture":
            culture,

        "day":
            day,

        "n_tracks":
            len(group),

        "spearman_brightness_m3":
            r_dl,

        "spearman_brightness_pc1":
            r_pc1,
    })


brightness = pd.DataFrame(
    brightness_rows
)

brightness_path = (
    OUT
    / "m3_longitudinal_brightness_sensitivity.csv"
)

brightness.to_csv(
    brightness_path,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 100)
print("CULTURE × DAY RESULTS")
print("=" * 100)

print(
    culture_day[
        [
            "culture",
            "day",
            "n_tracks",
            "m3_dl_mean",
            "m3_dl_median",
            "pc1_z_mean",
            "pc1_z_median",
            "crop_mean",
        ]
    ]
    .sort_values(
        [
            "culture",
            "day_number",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("M3 TRAJECTORIES — MEAN BY TRACK")
print("=" * 100)

print(
    culture_day
    .pivot(
        index="culture",
        columns="day",
        values="m3_dl_mean",
    )
    .to_string()
)


print()
print("=" * 100)
print("PC1 TRAJECTORIES — MEAN BY TRACK")
print("=" * 100)

print(
    culture_day
    .pivot(
        index="culture",
        columns="day",
        values="pc1_z_mean",
    )
    .to_string()
)


print()
print("=" * 100)
print("BRIGHTNESS SENSITIVITY")
print("=" * 100)

print(
    brightness.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(
    spot_path
)

print(
    track_path
)

print(
    culture_day_path
)

print(
    brightness_path
)

print()
print("DONE")
