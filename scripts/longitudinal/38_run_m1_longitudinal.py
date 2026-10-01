"""
Run frozen M1 models on the same common-support longitudinal crops as M2/M3.

Purpose
-------
Reuse the common-support spot set, reconstruct the corresponding brightfield
crops, verify crop identity, apply the five frozen M1 fold models with TTA8,
and aggregate predictions at spot, track, and culture × day levels.

Inputs
------
- ``outputs/longitudinal/m2_longitudinal_spot_predictions.csv``
- ``outputs/longitudinal/m2_longitudinal_culture_day.csv``
- ``outputs/longitudinal/common_support_culture_day.csv``
- ``outputs/m1/fold{0..4}/best_model.pt``
- matching M2 checkpoints for the frozen fold-specific brightfield
  normalization constants;
- longitudinal brightfield TIFFs.

Outputs
-------
Written under ``outputs/longitudinal/``:
- ``m1_longitudinal_spot_predictions.csv``
- ``m1_longitudinal_track_predictions.csv``
- ``m1_longitudinal_culture_day.csv``

Notes
-----
Historical M1 checkpoints do not contain the brightfield normalization
constants. The script therefore reads the corresponding M2 fold checkpoint,
which used the same culture split and train-only normalization procedure.
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

M1_DIR = ROOT / "outputs" / "m1"
OUT = ROOT / "outputs" / "longitudinal"

OUT.mkdir(parents=True, exist_ok=True)

BASE_PATH = (
    OUT
    / "m2_longitudinal_spot_predictions.csv"
)

M2_CULTURE_PATH = (
    OUT
    / "m2_longitudinal_culture_day.csv"
)

M3_CULTURE_PATH = (
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

class M1ResNet18(nn.Module):
    """ResNet18 regression architecture matching the frozen M1 checkpoints."""

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


    def forward(self, x):
        """Return the differentiation-score prediction for each crop."""

        x = F.interpolate(
            x,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )

        features = self.encoder(
            x
        )

        return (
            self.head_dl(
                features
            )
            .squeeze(1)
        )


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
                p = model(
                    xr
                )

            preds.append(
                p.float()
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
                p = model(
                    xf
                )

            preds.append(
                p.float()
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
# BASE SPOTS
# ============================================================

spots = pd.read_csv(
    BASE_PATH
)

print("=" * 100)
print("M1 LONGITUDINAL")
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
# LOAD 5 CHECKPOINTS
# ============================================================

models = []
norms = []


for fold in range(5):

    path = (
        M1_DIR
        / f"fold{fold}"
        / "best_model.pt"
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint[
        "model_state_dict"
    ]

    # Small safeguard in case the old script
    # used "head" instead of "head_dl".
    if (
        "head.weight" in state
        and "head_dl.weight" not in state
    ):

        state = dict(
            state
        )

        state[
            "head_dl.weight"
        ] = state.pop(
            "head.weight"
        )

        state[
            "head_dl.bias"
        ] = state.pop(
            "head.bias"
        )


    model = M1ResNet18().to(
        DEVICE
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()


    # Old M1 checkpoints do not store bf_mean/bf_std.
    # M2 uses exactly the same splits and the same train-only
    # normalization procedure as M1. Therefore, retrieve the
    # constants from the corresponding M2 fold.
    norm_checkpoint = torch.load(
        ROOT
        / "outputs"
        / "m2"
        / f"fold{fold}"
        / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    bf_mean = float(
        norm_checkpoint[
            "bf_mean"
        ]
    )

    bf_std = float(
        norm_checkpoint[
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
# INFERENCE
# ============================================================

parts = []


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
                        f"Invalid crop: "
                        f"{crop.shape}"
                    )


                crops[i] = crop


        rebuilt_mean = crops.mean(
            axis=(
                1,
                2,
                3,
            )
        )

        max_diff = float(
            np.max(
                np.abs(
                    rebuilt_mean
                    -
                    sub[
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
            "Max diff crop_mean :",
            f"{max_diff:.8f}",
        )


        if max_diff > 1e-5:

            raise RuntimeError(
                "Reconstruction differs."
            )


        fold_preds = []


        for fold in range(5):

            print(
                f"  Fold {fold}..."
            )

            mean, std = norms[
                fold
            ]

            p = infer_tta8(
                models[
                    fold
                ],
                crops,
                mean,
                std,
            )

            sub[
                f"m1_fold{fold}"
            ] = p

            fold_preds.append(
                p
            )


        fold_preds = np.stack(
            fold_preds,
            axis=1,
        )


        sub[
            "m1_mean"
        ] = fold_preds.mean(
            axis=1
        )

        sub[
            "m1_sd_folds"
        ] = fold_preds.std(
            axis=1,
            ddof=0,
        )


        print(
            "M1 median:",
            f"{sub['m1_mean'].median():.6f}",
        )


        parts.append(
            sub
        )


        del crops
        del fold_preds

        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# SPOT + TRACK + CULTURE
# ============================================================

pred = pd.concat(
    parts,
    ignore_index=True,
)


spot_out = (
    OUT
    / "m1_longitudinal_spot_predictions.csv"
)

pred.to_csv(
    spot_out,
    index=False,
)


agg = {
    "m1_mean":
        "mean",

    "m1_sd_folds":
        "mean",

    "crop_mean":
        "mean",
}


for fold in range(5):

    agg[
        f"m1_fold{fold}"
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
    / "m1_longitudinal_track_predictions.csv"
)

tracks.to_csv(
    track_out,
    index=False,
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
    .mean(
        numeric_only=True
    )
)


culture_out = (
    OUT
    / "m1_longitudinal_culture_day.csv"
)

culture_day.to_csv(
    culture_out,
    index=False,
)


print()
print("=" * 100)
print("M1 — CULTURE × DAY")
print("=" * 100)

print(
    culture_day[
        [
            "culture",
            "day",
            "m1_mean",
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
        values="m1_mean",
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
print("TRAJECTOIRES M1")
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
            rankdata(row)
            for row in x
        ]
    )

    n, k = ranks.shape

    sums = ranks.sum(
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
            sums ** 2
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
            rankdata(row)
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
print("M1 STATISTICS")
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
    ("Day00", "Day01"),
    ("Day01", "Day02"),
    ("Day00", "Day02"),
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
# COMPARE M1 / M2 / M3
# ============================================================

m2 = pd.read_csv(
    M2_CULTURE_PATH
)

m3 = pd.read_csv(
    M3_CULTURE_PATH
)


p2 = (
    m2.pivot(
        index="culture",
        columns="day",
        values="m2_mean",
    )
    .loc[
        CULTURES,
        DAYS,
    ]
)


p3 = (
    m3.pivot(
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
print("CONCORDANCE OF CHANGES")
print("=" * 100)


for a, b in [
    ("Day00", "Day01"),
    ("Day01", "Day02"),
    ("Day00", "Day02"),
]:

    d1 = (
        pivot[b]
        - pivot[a]
    ).to_numpy()

    d2 = (
        p2[b]
        - p2[a]
    ).to_numpy()

    d3 = (
        p3[b]
        - p3[a]
    ).to_numpy()


    r12 = spearmanr(
        d1,
        d2,
    )

    r13 = spearmanr(
        d1,
        d3,
    )


    print()
    print(
        f"{b}-{a}"
    )

    print(
        " M1 vs M2 :",
        f"rho={r12.statistic:+.4f}",
        f"p={r12.pvalue:.4f}",
    )

    print(
        " M1 vs M3 :",
        f"rho={r13.statistic:+.4f}",
        f"p={r13.pvalue:.4f}",
    )


print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(spot_out)
print(track_out)
print(culture_out)

print()
print("DONE")
