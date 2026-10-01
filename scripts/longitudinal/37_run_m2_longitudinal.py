"""
Run frozen M2 models on the common-support longitudinal crops.

Purpose
-------
Rebuild exactly the common-support crops previously used by M3, verify crop
identity through stored crop means, apply the five frozen M2 fold models with
TTA8, and aggregate predictions at spot, track, and culture × day levels.

Inputs
------
- ``outputs/longitudinal/m3_longitudinal_spot_predictions.csv``
- ``outputs/longitudinal/common_support_culture_day.csv``
- ``outputs/m2/fold{0..4}/best_model.pt``
- longitudinal brightfield TIFFs.

Outputs
-------
Written under ``outputs/longitudinal/``:
- ``m2_longitudinal_spot_predictions.csv``
- ``m2_longitudinal_track_predictions.csv``
- ``m2_longitudinal_culture_day.csv``

Notes
-----
The longitudinal statistics are sensitivity analyses over six cultures. M2
uses only brightfield data at inference; marker supervision was training-only.
"""

from pathlib import Path
import os
from itertools import product, permutations
import gc

import cv2
import numpy as np
import pandas as pd
import tifffile

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from scipy.stats import rankdata, wilcoxon, spearmanr


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

M2_DIR = (
    ROOT
    / "outputs"
    / "m2"
)

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

M3_COMMON_PATH = (
    OUT
    / "common_support_culture_day.csv"
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

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 128


# ============================================================
# MODEL
# ============================================================

class M2ResNet18(nn.Module):
    """ResNet18 multitask architecture matching the frozen M2 checkpoints."""

    def __init__(self):

        super().__init__()

        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            in_chans=16,
            num_classes=0,
            global_pool="avg",
        )

        n_features = int(
            self.encoder.num_features
        )

        self.head_dl = nn.Linear(
            n_features,
            1,
        )

        self.head_markers = nn.Linear(
            n_features,
            3,
        )


    def forward(self, x):
        """Return differentiation and auxiliary-marker predictions."""

        x = F.interpolate(
            x,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )

        features = self.encoder(
            x
        )

        dl = (
            self.head_dl(
                features
            )
            .squeeze(1)
        )

        markers = self.head_markers(
            features
        )

        return dl, markers


# ============================================================
# TTA x8
# ============================================================

@torch.no_grad()
def infer_tta8(
    model,
    crops,
    bf_mean,
    bf_std,
):
    """Return the TTA8-averaged differentiation prediction for each crop."""

    model.eval()

    outputs = []

    for start in range(
        0,
        len(crops),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(crops),
        )

        x = torch.from_numpy(
            crops[start:end]
        ).to(
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

        preds = []

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
                dl, _ = model(
                    xr
                )

            preds.append(
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
                dl, _ = model(
                    xf
                )

            preds.append(
                dl.float()
            )

        pred = (
            torch.stack(
                preds,
                dim=0,
            )
            .mean(dim=0)
        )

        outputs.append(
            pred.cpu().numpy()
        )

    return np.concatenate(
        outputs
    )


# ============================================================
# LOAD COMMON-SUPPORT SPOTS
# ============================================================

spots = pd.read_csv(
    SPOT_PATH
)

spots = spots[
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

print("=" * 100)
print("M2 LONGITUDINAL")
print("=" * 100)

print(
    "Common-support spots:",
    len(spots),
)

print(
    "Device :",
    DEVICE,
)

if DEVICE.type == "cuda":

    print(
        "GPU :",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# LOAD MODELS
# ============================================================

models = []
norms = []

for fold in range(5):

    ckpt_path = (
        M2_DIR
        / f"fold{fold}"
        / "best_model.pt"
    )

    if not ckpt_path.exists():

        raise FileNotFoundError(
            ckpt_path
        )

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    model = M2ResNet18().to(
        DEVICE
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

    models.append(
        model
    )

    norms.append(
        (
            bf_mean,
            bf_std,
        )
    )

    print(
        f"Fold {fold} loaded | "
        f"mean={bf_mean:.6f} "
        f"std={bf_std:.6f}"
    )


# ============================================================
# REBUILD CROPS EXACTLY
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

        sub = (
            spots[
                (spots["culture"] == culture)
                &
                (spots["day"] == day)
            ]
            .copy()
            .reset_index(drop=True)
        )

        crops = np.empty(
            (
                len(sub),
                16,
                64,
                64,
            ),
            dtype=np.uint8,
        )

        for frame, g in sub.groupby(
            "frame"
        ):

            frame = int(
                frame
            )

            bf_t_values = (
                g[
                    "bf_t"
                ]
                .unique()
            )

            if len(bf_t_values) != 1:

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

            bf = tifffile.imread(
                bf_path
            )

            zdim, h, w = bf.shape

            bf_half = np.empty(
                (
                    zdim,
                    h // 2,
                    w // 2,
                ),
                dtype=np.uint8,
            )

            for z in range(
                zdim
            ):

                bf_half[z] = cv2.resize(
                    bf[z],
                    (
                        w // 2,
                        h // 2,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )

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
                        f"{crop.shape}"
                    )

                crops[i] = crop


        # ----------------------------------------------------
        # Sanity check against the crop means already saved
        # during M3 inference.
        # ----------------------------------------------------

        rebuilt_means = crops.mean(
            axis=(
                1,
                2,
                3,
            )
        )

        max_mean_diff = float(
            np.max(
                np.abs(
                    rebuilt_means
                    - sub[
                        "crop_mean"
                    ].to_numpy()
                )
            )
        )

        print(
            "Crops :",
            len(crops),
        )

        print(
            "Max diff crop_mean vs M3 :",
            f"{max_mean_diff:.8f}",
        )

        if max_mean_diff > 1e-5:

            raise RuntimeError(
                "Reconstruction crop "
                "differs from M3."
            )


        fold_preds = []

        for fold in range(5):

            print(
                f"  Fold {fold}..."
            )

            bf_mean, bf_std = (
                norms[
                    fold
                ]
            )

            pred = infer_tta8(
                models[
                    fold
                ],
                crops,
                bf_mean,
                bf_std,
            )

            sub[
                f"m2_fold{fold}"
            ] = pred

            fold_preds.append(
                pred
            )


        fold_preds = np.stack(
            fold_preds,
            axis=1,
        )

        sub[
            "m2_mean"
        ] = fold_preds.mean(
            axis=1
        )

        sub[
            "m2_sd_folds"
        ] = fold_preds.std(
            axis=1,
            ddof=0,
        )


        print(
            "M2 median:",
            f"{sub['m2_mean'].median():.6f}",
        )

        all_parts.append(
            sub
        )

        del crops
        del fold_preds

        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# COMBINE
# ============================================================

pred = pd.concat(
    all_parts,
    ignore_index=True,
)

spot_out = (
    OUT
    / "m2_longitudinal_spot_predictions.csv"
)

pred.to_csv(
    spot_out,
    index=False,
)


# ============================================================
# TRACK LEVEL
# ============================================================

agg = {
    "m2_mean":
        "mean",

    "m2_sd_folds":
        "mean",

    "crop_mean":
        "mean",
}

for fold in range(5):

    agg[
        f"m2_fold{fold}"
    ] = "mean"


tracks = (
    pred
    .groupby(
        [
            "culture",
            "day",
            "track_id",
        ],
        as_index=False,
    )
    .agg(
        agg
    )
)

track_out = (
    OUT
    / "m2_longitudinal_track_predictions.csv"
)

tracks.to_csv(
    track_out,
    index=False,
)


# ============================================================
# CULTURE LEVEL
# ============================================================

culture_day = (
    tracks
    .groupby(
        [
            "culture",
            "day",
        ],
        as_index=False,
    )
    .mean(
        numeric_only=True
    )
)

culture_out = (
    OUT
    / "m2_longitudinal_culture_day.csv"
)

culture_day.to_csv(
    culture_out,
    index=False,
)


print()
print("=" * 100)
print("M2 — CULTURE × DAY")
print("=" * 100)

print(
    culture_day[
        [
            "culture",
            "day",
            "m2_mean",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# TRAJECTORIES
# ============================================================

pivot = (
    culture_day
    .pivot(
        index="culture",
        columns="day",
        values="m2_mean",
    )
    .loc[
        CULTURES,
        DAYS,
    ]
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

pivot[
    "D02-D00"
] = (
    pivot["Day02"]
    - pivot["Day00"]
)


print()
print("=" * 100)
print("TRAJECTOIRES M2")
print("=" * 100)

print(
    pivot.to_string()
)


# ============================================================
# EXACT FRIEDMAN
# ============================================================

def friedman_q(x):
    """Compute the Friedman rank statistic for the six-culture trajectories."""

    ranks = np.stack(
        [
            rankdata(
                row
            )
            for row in x
        ]
    )

    n, k = ranks.shape

    rank_sums = ranks.sum(
        axis=0
    )

    return float(
        12
        / (
            n
            * k
            * (k + 1)
        )
        * np.sum(
            rank_sums ** 2
        )
        -
        3
        * n
        * (k + 1)
    )


def exact_friedman_p(x):
    """Enumerate within-culture day permutations for an exact p-value."""

    observed = friedman_q(
        x
    )

    n, k = x.shape

    ranks = np.stack(
        [
            rankdata(
                row
            )
            for row in x
        ]
    )

    perms = list(
        permutations(
            range(k)
        )
    )

    extreme = 0
    total = 0

    for choices in product(
        perms,
        repeat=n,
    ):

        sums = np.zeros(
            k,
            dtype=float,
        )

        for i, perm in enumerate(
            choices
        ):

            sums += ranks[
                i,
                list(perm),
            ]

        q = (
            12
            / (
                n
                * k
                * (k + 1)
            )
            * np.sum(
                sums ** 2
            )
            -
            3
            * n
            * (k + 1)
        )

        if q >= observed - 1e-12:
            extreme += 1

        total += 1

    return (
        observed,
        extreme / total,
    )


x = (
    pivot[
        DAYS
    ]
    .to_numpy(
        dtype=float
    )
)

q, p_exact = (
    exact_friedman_p(
        x
    )
)


print()
print("=" * 100)
print("M2 STATISTICS")
print("=" * 100)

print(
    "Friedman Q :",
    f"{q:.6f}",
)

print(
    "exact p    :",
    f"{p_exact:.8f}",
)


for a, b in [
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

    delta = (
        pivot[b]
        - pivot[a]
    ).to_numpy()

    w = wilcoxon(
        delta,
        method="exact",
        alternative="two-sided",
    )

    print()
    print(
        f"{b}-{a}"
    )

    print(
        " Δ :",
        [
            round(
                float(v),
                6,
            )
            for v in delta
        ]
    )

    print(
        " mean    :",
        f"{delta.mean():+.6f}",
    )

    print(
        " signs  :",
        f"{np.sum(delta < 0)} neg / "
        f"{np.sum(delta > 0)} pos",
    )

    print(
        " Raw Wilcoxon  :",
        f"{w.pvalue:.6f}",
    )


# ============================================================
# M2 vs M3 CHANGES
# ============================================================

m3 = pd.read_csv(
    M3_COMMON_PATH
)

m3_pivot = (
    m3
    .pivot(
        index="culture",
        columns="day",
        values="m3",
    )
    .loc[
        CULTURES,
        DAYS,
    ]
)


print()
print("=" * 100)
print("CONCORDANCE OF M2 vs M3 CHANGES")
print("=" * 100)


for a, b in [
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

    dm2 = (
        pivot[b]
        - pivot[a]
    ).to_numpy()

    dm3 = (
        m3_pivot[b]
        - m3_pivot[a]
    ).to_numpy()

    r = spearmanr(
        dm2,
        dm3,
    )

    print()
    print(
        f"{b}-{a}"
    )

    print(
        " Spearman ΔM2 vs ΔM3 :",
        f"rho={r.statistic:+.4f}",
        f"p={r.pvalue:.4f}",
    )


print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(
    spot_out
)

print(
    track_out
)

print(
    culture_out
)

print()
print("DONE")
