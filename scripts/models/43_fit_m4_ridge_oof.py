from pathlib import Path
import hashlib

import joblib
import numpy as np
import pandas as pd
import torch

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path(__file__).resolve().parents[2]

EMBED_PATH = (
    ROOT
    / "outputs/m4/dinov2_static_embeddings.npy"
)

META_PATH = (
    ROOT
    / "outputs/m4/dinov2_static_metadata.csv"
)

OUT = (
    ROOT
    / "outputs/m4"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

ALPHAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

EXPECTED_N = 30877
EXPECTED_DIM = 384


# ======================================================================
# UTILS
# ======================================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    return {
        "r2":
            r2_score(
                y_true,
                y_pred,
            ),

        "mae":
            mean_absolute_error(
                y_true,
                y_pred,
            ),

        "rmse":
            mean_squared_error(
                y_true,
                y_pred,
            ) ** 0.5,

        "pearson":
            pearsonr(
                y_true,
                y_pred,
            ).statistic,

        "spearman":
            spearmanr(
                y_true,
                y_pred,
            ).statistic,
    }


# ======================================================================
# LOAD
# ======================================================================

X = np.load(
    EMBED_PATH,
    mmap_mode="r",
)

meta = pd.read_csv(
    META_PATH
)


if X.shape != (
    EXPECTED_N,
    EXPECTED_DIM,
):
    raise RuntimeError(
        f"Unexpected embeddings: {X.shape}"
    )


if len(meta) != EXPECTED_N:
    raise RuntimeError(
        f"Metadata: {len(meta)} rows"
    )


required = {
    "cache_index",
    "culture",
    "fold",
    "DL",
}

missing = (
    required
    - set(meta.columns)
)

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


if not np.array_equal(
    meta["cache_index"].to_numpy(),
    np.arange(EXPECTED_N),
):
    raise RuntimeError(
        "Misaligned cache_index."
    )


if not np.isfinite(
    np.asarray(X)
).all():
    raise RuntimeError(
        "Non-finite embeddings."
    )


print("=" * 96)
print("M4 — FROZEN DINOv2 + RIDGE — 5-FOLD CULTURE-WISE")
print("=" * 96)

print()
print("Embeddings :", X.shape)
print("Cultures   :", meta["culture"].nunique())
print("Nuclei     :", len(meta))
print("Alpha grid :", ALPHAS)

print()
print(
    "SHA256 embeddings :",
    sha256_file(
        EMBED_PATH
    ),
)


# ======================================================================
# OOF STORAGE
# ======================================================================

oof_prediction = np.full(
    EXPECTED_N,
    np.nan,
    dtype=np.float64,
)

oof_fold = np.full(
    EXPECTED_N,
    -1,
    dtype=np.int64,
)

fold_rows = []
alpha_rows = []


# ======================================================================
# 5 EXTERNAL FOLDS
# ======================================================================

for fold in range(5):

    print()
    print("=" * 96)
    print(
        f"FOLD {fold}"
    )
    print("=" * 96)


    # --------------------------------------------------------------
    # EXACT M1 SPLIT
    # --------------------------------------------------------------

    checkpoint_path = (
        ROOT
        / f"outputs/m1/fold{fold}/best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    train_cultures = sorted(
        checkpoint[
            "train_cultures"
        ]
    )

    val_cultures = sorted(
        checkpoint[
            "val_cultures"
        ]
    )

    test_cultures = sorted(
        checkpoint[
            "test_cultures"
        ]
    )


    # --------------------------------------------------------------
    # VERIFY DISJOINT
    # --------------------------------------------------------------

    train_set = set(
        train_cultures
    )

    val_set = set(
        val_cultures
    )

    test_set = set(
        test_cultures
    )


    if (
        train_set & val_set
        or train_set & test_set
        or val_set & test_set
    ):
        raise RuntimeError(
            f"Fold {fold}: culture leakage."
        )


    official_test = set(
        meta.loc[
            meta["fold"] == fold,
            "culture",
        ].unique()
    )


    if official_test != test_set:
        raise RuntimeError(
            f"Fold {fold}: test cultures "
            "!= fold metadata."
        )


    train_mask = (
        meta["culture"]
        .isin(train_cultures)
        .to_numpy()
    )

    val_mask = (
        meta["culture"]
        .isin(val_cultures)
        .to_numpy()
    )

    test_mask = (
        meta["culture"]
        .isin(test_cultures)
        .to_numpy()
    )


    print(
        f"Train : "
        f"{train_mask.sum():6d} nuclei / "
        f"{len(train_cultures):2d} cultures"
    )

    print(
        f"Val   : "
        f"{val_mask.sum():6d} nuclei / "
        f"{len(val_cultures):2d} cultures"
    )

    print(
        f"Test  : "
        f"{test_mask.sum():6d} nuclei / "
        f"{len(test_cultures):2d} cultures"
    )


    X_train = np.asarray(
        X[train_mask],
        dtype=np.float32,
    )

    X_val = np.asarray(
        X[val_mask],
        dtype=np.float32,
    )

    X_test = np.asarray(
        X[test_mask],
        dtype=np.float32,
    )


    y_train = (
        meta.loc[
            train_mask,
            "DL",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y_val = (
        meta.loc[
            val_mask,
            "DL",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y_test = (
        meta.loc[
            test_mask,
            "DL",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    # --------------------------------------------------------------
    # STANDARDIZATION — TRAIN ONLY
    # --------------------------------------------------------------

    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(
        X_train
    )

    X_val_std = scaler.transform(
        X_val
    )

    X_test_std = scaler.transform(
        X_test
    )


    # --------------------------------------------------------------
    # ALPHA SELECTION — EXISTING VALIDATION ONLY
    # --------------------------------------------------------------

    candidates = []


    print()
    print(
        "Alpha validation:"
    )


    for alpha in ALPHAS:

        model = Ridge(
            alpha=alpha,
        )

        model.fit(
            X_train_std,
            y_train,
        )

        pred_val = model.predict(
            X_val_std
        )

        val_m = metrics(
            y_val,
            pred_val,
        )

        candidates.append(
            (
                val_m["mae"],
                alpha,
                val_m,
            )
        )


        alpha_rows.append({
            "fold":
                fold,

            "alpha":
                alpha,

            "val_r2":
                val_m["r2"],

            "val_mae":
                val_m["mae"],

            "val_rmse":
                val_m["rmse"],

            "val_pearson":
                val_m["pearson"],

            "val_spearman":
                val_m["spearman"],
        })


        print(
            f"alpha={alpha:8g} | "
            f"MAE={val_m['mae']:.6f} | "
            f"R2={val_m['r2']:+.6f}"
        )


    # Tie-break:
    # smallest validation MAE,
    # then smallest alpha.
    candidates.sort(
        key=lambda z: (
            z[0],
            z[1],
        )
    )

    best_val_mae, best_alpha, best_val_metrics = (
        candidates[0]
    )


    print()
    print(
        f"✅ selected alpha: {best_alpha:g}"
    )

    print(
        f"   Val MAE      : "
        f"{best_val_mae:.6f}"
    )


    # --------------------------------------------------------------
    # FINAL RIDGE
    #
    # IMPORTANT:
    # train only — no train+val refit.
    # --------------------------------------------------------------

    final_model = Ridge(
        alpha=best_alpha,
    )

    final_model.fit(
        X_train_std,
        y_train,
    )


    pred_test = final_model.predict(
        X_test_std
    )


    test_m = metrics(
        y_test,
        pred_test,
    )


    print()
    print(
        "EXTERNAL TEST"
    )

    for k, v in test_m.items():

        print(
            f"{k:10s}: "
            f"{v:.6f}"
        )


    # --------------------------------------------------------------
    # STORE OOF
    # --------------------------------------------------------------

    test_indices = (
        meta.loc[
            test_mask,
            "cache_index",
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    if np.isfinite(
        oof_prediction[
            test_indices
        ]
    ).any():
        raise RuntimeError(
            f"Fold {fold}: OOF already filled."
        )


    oof_prediction[
        test_indices
    ] = pred_test

    oof_fold[
        test_indices
    ] = fold


    # --------------------------------------------------------------
    # SAVE PROBE
    # --------------------------------------------------------------

    probe_path = (
        OUT
        / f"fold{fold}_ridge_probe.joblib"
    )


    joblib.dump(
        {
            "fold":
                fold,

            "alpha":
                best_alpha,

            "scaler":
                scaler,

            "ridge":
                final_model,

            "train_cultures":
                train_cultures,

            "val_cultures":
                val_cultures,

            "test_cultures":
                test_cultures,

            "input_slices":
                [7, 8, 9],

            "embedding_dim":
                EXPECTED_DIM,

            "backbone":
                "vit_small_patch14_dinov2.lvd142m",

            "input_size":
                518,

            "interpolation":
                "bicubic",

            "imagenet_mean":
                [0.485, 0.456, 0.406],

            "imagenet_std":
                [0.229, 0.224, 0.225],
        },
        probe_path,
    )


    fold_rows.append({
        "fold":
            fold,

        "best_alpha":
            best_alpha,

        "n_train":
            int(
                train_mask.sum()
            ),

        "n_val":
            int(
                val_mask.sum()
            ),

        "n_test":
            int(
                test_mask.sum()
            ),

        "val_mae":
            best_val_metrics["mae"],

        "test_r2":
            test_m["r2"],

        "test_mae":
            test_m["mae"],

        "test_rmse":
            test_m["rmse"],

        "test_pearson":
            test_m["pearson"],

        "test_spearman":
            test_m["spearman"],
    })


# ======================================================================
# VERIFY COMPLETE OOF
# ======================================================================

if not np.isfinite(
    oof_prediction
).all():
    missing = np.where(
        ~np.isfinite(
            oof_prediction
        )
    )[0]

    raise RuntimeError(
        f"Incomplete OOF: "
        f"{len(missing)} missing predictions."
    )


if not np.array_equal(
    oof_fold,
    meta[
        "fold"
    ].to_numpy(
        dtype=np.int64
    ),
):
    raise RuntimeError(
        "OOF fold assignment incorrect."
    )


# ======================================================================
# GLOBAL NUCLEUS OOF
# ======================================================================

y = meta[
    "DL"
].to_numpy(
    dtype=np.float64
)

global_metrics = metrics(
    y,
    oof_prediction,
)


# ======================================================================
# CULTURE-LEVEL OOF
# ======================================================================

oof = meta.copy()

oof[
    "prediction_m4"
] = oof_prediction


culture = (
    oof
    .groupby(
        "culture",
        as_index=False,
    )
    .agg(
        target_DL=(
            "DL",
            "mean",
        ),

        prediction_m4=(
            "prediction_m4",
            "mean",
        ),

        n_nuclei=(
            "cache_index",
            "size",
        ),
    )
)


culture_metrics = metrics(
    culture[
        "target_DL"
    ],
    culture[
        "prediction_m4"
    ],
)


# ======================================================================
# SAVE
# ======================================================================

oof_path = (
    OUT
    / "m4_oof_predictions.csv"
)

culture_path = (
    OUT
    / "m4_oof_culture_predictions.csv"
)

fold_path = (
    OUT
    / "m4_fold_summary.csv"
)

alpha_path = (
    OUT
    / "m4_alpha_validation.csv"
)


oof.to_csv(
    oof_path,
    index=False,
)

culture.to_csv(
    culture_path,
    index=False,
)

pd.DataFrame(
    fold_rows
).to_csv(
    fold_path,
    index=False,
)

pd.DataFrame(
    alpha_rows
).to_csv(
    alpha_path,
    index=False,
)


# ======================================================================
# FINAL REPORT
# ======================================================================

print()
print("=" * 96)
print("M4 — GLOBAL OOF — NUCLEUS LEVEL")
print("=" * 96)

for k, v in global_metrics.items():

    print(
        f"{k:10s}: "
        f"{v:.6f}"
    )


print()
print("=" * 96)
print("M4 — GLOBAL OOF — CULTURE LEVEL")
print("=" * 96)

print(
    "Cultures   :",
    len(culture),
)

for k, v in culture_metrics.items():

    print(
        f"{k:10s}: "
        f"{v:.6f}"
    )


print()
print("=" * 96)
print("ALPHA BY FOLD")
print("=" * 96)

print(
    pd.DataFrame(
        fold_rows
    )[
        [
            "fold",
            "best_alpha",
            "val_mae",
            "test_r2",
            "test_mae",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


summary_path = (
    OUT
    / "m4_oof_summary.txt"
)

lines = []

lines.append(
    "M4 DINOv2 ViT-S frozen + Ridge"
)

lines.append(
    "=" * 72
)

lines.append("")

lines.append(
    "GLOBAL OOF — NUCLEUS"
)

for k, v in global_metrics.items():
    lines.append(
        f"{k}: {v:.8f}"
    )

lines.append("")

lines.append(
    "GLOBAL OOF — CULTURE"
)

for k, v in culture_metrics.items():
    lines.append(
        f"{k}: {v:.8f}"
    )

lines.append("")

lines.append(
    "No TTA. No clipping. "
    "Same external culture folds as M1/M2/M3."
)

lines.append(
    "Alpha selected on the pre-existing "
    "culture-wise validation split of each fold."
)

lines.append(
    "No longitudinal M4 data were used."
)


summary_path.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print("=" * 96)
print("FILES")
print("=" * 96)

print(oof_path)
print(culture_path)
print(fold_path)
print(alpha_path)
print(summary_path)

print()
print("✅ STATIC M4 COMPLETE")
print("NO LONGITUDINAL DATA USED")
print()
print("DONE")
