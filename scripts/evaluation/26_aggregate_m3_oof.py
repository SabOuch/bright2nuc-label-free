"""
Aggregate M3 out-of-fold predictions and compare M3 with M0-M2.

Purpose
-------
Collect the five held-out M3 fold predictions, align them with the frozen
M0/M1/M2 OOF table, verify differentiation and auxiliary-target consistency,
and compute global, fold-level, culture-level, and auxiliary-marker metrics.

The script additionally performs:
- a paired per-culture M3-vs-M2 MAE comparison;
- a two-sided Wilcoxon signed-rank test across cultures;
- a 1,000-resample culture-cluster bootstrap.

Inputs
------
- ``outputs/m2/m2_oof_predictions.csv``
- ``outputs/m3/fold{0..4}/test_predictions.csv``

Outputs
-------
Written under ``outputs/m3/``:
- ``m3_oof_predictions.csv``
- ``m3_vs_m2_vs_m1_vs_m0_global_metrics.csv``
- ``m3_vs_m2_vs_m1_vs_m0_per_fold.csv``
- ``m3_fold_summary.csv``
- ``m3_vs_m2_per_culture.csv``
- ``m3_culture_metrics.csv``
- ``m3_marker_oof_metrics.csv``
- ``m3_cluster_bootstrap.csv``
- ``m3_aggregate_summary.txt``

Notes
-----
M3 is the ConvNeXt-Tiny multitask model. Fluorescence-derived marker values are
used only as auxiliary training targets; biological time is neither an input
nor a target.

Reproducibility
---------------
The cluster bootstrap resamples whole cultures with NumPy seed 42.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    wilcoxon,
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

M2_OOF_PATH = (
    ROOT
    / "outputs/m2/m2_oof_predictions.csv"
)

M3_DIR = (
    ROOT
    / "outputs/m3"
)

OUT = M3_DIR

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# METRICS
# ================================================================

def metrics(y, pred):
    """Return regression and correlation metrics for one prediction vector."""

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    return {
        "r2": r2_score(
            y,
            pred,
        ),

        "mae": mean_absolute_error(
            y,
            pred,
        ),

        "rmse": np.sqrt(
            mean_squared_error(
                y,
                pred,
            )
        ),

        "pearson": pearsonr(
            y,
            pred,
        ).statistic,

        "spearman": spearmanr(
            y,
            pred,
        ).statistic,
    }


# ================================================================
# 1. LOAD M2 OOF REFERENCE
# ================================================================

if not M2_OOF_PATH.exists():
    raise FileNotFoundError(
        M2_OOF_PATH
    )


base = pd.read_csv(
    M2_OOF_PATH
)


required_base = {
    "filename",
    "culture",
    "fold",
    "DL",
    "pred_M0_119",
    "m1_tta8",
    "m2_tta8",
    "target_OCT4_norm",
    "target_FOXA2_norm",
    "target_SOX17_norm",
}


missing = (
    required_base
    - set(base.columns)
)


if missing:
    raise RuntimeError(
        f"Missing columns in M2 OOF: "
        f"{missing}"
    )


if len(base) != 30877:
    raise RuntimeError(
        f"M2 OOF: expected 30,877 rows, "
        f"got {len(base)}"
    )


if base["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filename in M2 OOF."
    )


print("=" * 88)
print("M3 OOF AGGREGATION")
print("=" * 88)

print(
    "M2 OOF reference:",
    len(base),
    "nuclei /",
    base["culture"].nunique(),
    "cultures",
)


# ================================================================
# 2. LOAD 5 M3 TEST FOLDS
# ================================================================

parts = []


for fold in range(5):

    path = (
        M3_DIR
        / f"fold{fold}"
        / "test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )


    x = pd.read_csv(
        path
    )


    required = {
        "cache_index",
        "filename",
        "culture",
        "fold",
        "target_DL",
        "prediction_single",
        "prediction_tta8",
        "target_OCT4_norm",
        "target_FOXA2_norm",
        "target_SOX17_norm",
        "pred_OCT4_norm",
        "pred_FOXA2_norm",
        "pred_SOX17_norm",
    }


    missing = (
        required
        - set(x.columns)
    )


    if missing:
        raise RuntimeError(
            f"Fold {fold}: "
            f"missing columns {missing}"
        )


    if not (
        x["fold"] == fold
    ).all():

        raise RuntimeError(
            f"Fold {fold}: "
            "inconsistent fold number."
        )


    if x["filename"].duplicated().any():

        raise RuntimeError(
            f"Fold {fold}: "
            "duplicate filename."
        )


    print(
        f"Fold {fold}: "
        f"{len(x):5d} nuclei / "
        f"{x['culture'].nunique():2d} cultures"
    )


    parts.append(
        x
    )


m3 = pd.concat(
    parts,
    ignore_index=True,
)


print()
print(
    "M3 total:",
    len(m3),
    "nuclei",
)


if len(m3) != 30877:
    raise RuntimeError(
        f"Expected 30,877 M3 predictions, "
        f"got {len(m3)}."
    )


if m3["filename"].duplicated().any():
    raise RuntimeError(
        "A nucleus appears in multiple M3 folds."
    )


if m3["culture"].nunique() != 48:
    raise RuntimeError(
        f"Expected 48 cultures, "
        f"got {m3['culture'].nunique()}."
    )


print(
    "M3 cultures:",
    m3["culture"].nunique(),
)


# ================================================================
# 3. PREPARE M3 COLUMNS
# ================================================================

m3small = m3[
    [
        "filename",
        "culture",
        "fold",
        "target_DL",
        "prediction_single",
        "prediction_tta8",
        "target_OCT4_norm",
        "target_FOXA2_norm",
        "target_SOX17_norm",
        "pred_OCT4_norm",
        "pred_FOXA2_norm",
        "pred_SOX17_norm",
    ]
].rename(
    columns={
        "culture":
            "culture_m3",

        "fold":
            "fold_m3",

        "target_DL":
            "target_DL_m3",

        "prediction_single":
            "m3_single",

        "prediction_tta8":
            "m3_tta8",

        "target_OCT4_norm":
            "m3_target_OCT4_norm",

        "target_FOXA2_norm":
            "m3_target_FOXA2_norm",

        "target_SOX17_norm":
            "m3_target_SOX17_norm",

        "pred_OCT4_norm":
            "m3_pred_OCT4_norm",

        "pred_FOXA2_norm":
            "m3_pred_FOXA2_norm",

        "pred_SOX17_norm":
            "m3_pred_SOX17_norm",
    }
)


# ================================================================
# 4. MERGE M0/M1/M2 + M3
# ================================================================

oof = base.merge(
    m3small,
    on="filename",
    how="inner",
    validate="one_to_one",
)


if len(oof) != 30877:
    raise RuntimeError(
        "Incomplete M2/M3 merge."
    )


if not (
    oof["culture"]
    == oof["culture_m3"]
).all():

    raise RuntimeError(
        "Culture mismatch M2/M3."
    )


if not (
    oof["fold"]
    == oof["fold_m3"]
).all():

    raise RuntimeError(
        "Fold mismatch M2/M3."
    )


target_diff = float(
    np.max(
        np.abs(
            oof["DL"].to_numpy(
                dtype=np.float64
            )
            -
            oof["target_DL_m3"].to_numpy(
                dtype=np.float64
            )
        )
    )
)


print()
print(
    "Maximum DL target difference between M2 and M3:",
    f"{target_diff:.12g}",
)


if target_diff > 1e-6:
    raise RuntimeError(
        "Target DL mismatch M2/M3."
    )


# Verify the marker targets.
for marker in [
    "OCT4",
    "FOXA2",
    "SOX17",
]:

    old = (
        oof[
            f"target_{marker}_norm"
        ].to_numpy(
            dtype=np.float64
        )
    )

    new = (
        oof[
            f"m3_target_{marker}_norm"
        ].to_numpy(
            dtype=np.float64
        )
    )

    diff = float(
        np.max(
            np.abs(
                old - new
            )
        )
    )

    print(
        f"Maximum {marker:5s} target difference between M2 and M3: "
        f"{diff:.12g}"
    )

    if diff > 1e-6:
        raise RuntimeError(
            f"Target {marker} mismatch M2/M3."
        )


oof = oof.drop(
    columns=[
        "culture_m3",
        "fold_m3",
        "target_DL_m3",
    ]
)


# ================================================================
# 5. GLOBAL OOF METRICS
# ================================================================

models = {
    "M0_RF":
        "pred_M0_119",

    "M1_ResNet18_DL":
        "m1_tta8",

    "M2_ResNet18_markers":
        "m2_tta8",

    "M3_ConvNeXt_markers":
        "m3_tta8",
}


global_rows = []


print()
print("=" * 88)
print("GLOBAL OOF RESULTS — 30,877 NUCLEI")
print("=" * 88)


for name, col in models.items():

    r = metrics(
        oof["DL"],
        oof[col],
    )

    global_rows.append({
        "model": name,
        **r,
    })

    print(
        f"\n{name}"
    )

    for k, v in r.items():

        print(
            f"{k:10s}: "
            f"{v:.6f}"
        )


global_df = pd.DataFrame(
    global_rows
)


# ================================================================
# 6. PER-FOLD METRICS
# ================================================================

fold_rows = []


for fold in range(5):

    sub = oof[
        oof["fold"] == fold
    ]

    for name, col in models.items():

        r = metrics(
            sub["DL"],
            sub[col],
        )

        fold_rows.append({
            "fold":
                fold,

            "model":
                name,

            "n":
                len(sub),

            **r,
        })


fold_df = pd.DataFrame(
    fold_rows
)


print()
print("=" * 88)
print("R2 BY FOLD")
print("=" * 88)

print(
    fold_df.pivot(
        index="fold",
        columns="model",
        values="r2",
    )
    .round(6)
    .to_string()
)


print()
print("=" * 88)
print("MAE BY FOLD")
print("=" * 88)

print(
    fold_df.pivot(
        index="fold",
        columns="model",
        values="mae",
    )
    .round(6)
    .to_string()
)


# ================================================================
# 7. FOLD MEAN +/- SD
# ================================================================

fold_summary = (
    fold_df
    .groupby("model")
    .agg(
        r2_mean=(
            "r2",
            "mean",
        ),

        r2_std=(
            "r2",
            "std",
        ),

        mae_mean=(
            "mae",
            "mean",
        ),

        mae_std=(
            "mae",
            "std",
        ),

        pearson_mean=(
            "pearson",
            "mean",
        ),

        spearman_mean=(
            "spearman",
            "mean",
        ),
    )
    .reset_index()
)


print()
print("=" * 88)
print("MEAN +/- SD ACROSS 5 FOLDS")
print("=" * 88)

print(
    fold_summary
    .round(6)
    .to_string(
        index=False
    )
)


# ================================================================
# 8. CULTURE LEVEL
# ================================================================

culture = (
    oof
    .groupby("culture")
    .agg(
        n=(
            "filename",
            "size",
        ),

        DL=(
            "DL",
            "mean",
        ),

        M0=(
            "pred_M0_119",
            "mean",
        ),

        M1=(
            "m1_tta8",
            "mean",
        ),

        M2=(
            "m2_tta8",
            "mean",
        ),

        M3=(
            "m3_tta8",
            "mean",
        ),
    )
    .reset_index()
)


culture_models = {
    "M0_RF":
        "M0",

    "M1_ResNet18_DL":
        "M1",

    "M2_ResNet18_markers":
        "M2",

    "M3_ConvNeXt_markers":
        "M3",
}


culture_rows = []


print()
print("=" * 88)
print("CULTURE-LEVEL RESULTS — 48 CULTURES")
print("=" * 88)


for name, col in culture_models.items():

    r = metrics(
        culture["DL"],
        culture[col],
    )

    culture_rows.append({
        "model": name,
        **r,
    })

    print(
        f"\n{name}"
    )

    for k, v in r.items():

        print(
            f"{k:10s}: "
            f"{v:.6f}"
        )


culture_metrics_df = pd.DataFrame(
    culture_rows
)


# ================================================================
# 9. PER-CULTURE NUCLEUS MAE
# ================================================================

culture_errors = []


for culture_name, sub in oof.groupby(
    "culture",
    sort=True,
):

    culture_errors.append({
        "culture":
            culture_name,

        "n":
            len(sub),

        "M0_mae":
            np.mean(
                np.abs(
                    sub["DL"]
                    - sub["pred_M0_119"]
                )
            ),

        "M1_mae":
            np.mean(
                np.abs(
                    sub["DL"]
                    - sub["m1_tta8"]
                )
            ),

        "M2_mae":
            np.mean(
                np.abs(
                    sub["DL"]
                    - sub["m2_tta8"]
                )
            ),

        "M3_mae":
            np.mean(
                np.abs(
                    sub["DL"]
                    - sub["m3_tta8"]
                )
            ),
    })


culture_errors = pd.DataFrame(
    culture_errors
)


culture = culture.merge(
    culture_errors,
    on=[
        "culture",
        "n",
    ],
    validate="one_to_one",
)


# ================================================================
# 10. M3 VS M2 — PAIRED CULTURE TEST
# ================================================================

delta = (
    culture["M3_mae"]
    - culture["M2_mae"]
)


n_better = int(
    (delta < 0).sum()
)

n_worse = int(
    (delta > 0).sum()
)

n_equal = int(
    (delta == 0).sum()
)


wilcox = wilcoxon(
    culture["M3_mae"],
    culture["M2_mae"],
    alternative="two-sided",
)


print()
print("=" * 88)
print("M3 vs M2 — MEAN NUCLEUS MAE BY CULTURE")
print("=" * 88)

print(
    "Cultures improved by M3:",
    f"{n_better}/48",
)

print(
    "Cultures degraded by M3:",
    f"{n_worse}/48",
)

print(
    "Ties                       :",
    n_equal,
)

print(
    "Mean M3-M2 delta           :",
    f"{delta.mean():+.6f}",
)

print(
    "Median M3-M2 delta         :",
    f"{delta.median():+.6f}",
)

print(
    "Wilcoxon statistic         :",
    f"{wilcox.statistic:.6f}",
)

print(
    "Wilcoxon p two-sided       :",
    f"{wilcox.pvalue:.8g}",
)


# ================================================================
# 11. M3 AUXILIARY MARKERS
# ================================================================

marker_rows = []


print()
print("=" * 88)
print("AUXILIARY MARKER PERFORMANCE — M3 OOF")
print("=" * 88)


for marker in [
    "OCT4",
    "FOXA2",
    "SOX17",
]:

    r = metrics(
        oof[
            f"m3_target_{marker}_norm"
        ],
        oof[
            f"m3_pred_{marker}_norm"
        ],
    )

    marker_rows.append({
        "marker":
            marker,
        **r,
    })

    print(
        f"\n{marker}"
    )

    for k, v in r.items():

        print(
            f"{k:10s}: "
            f"{v:.6f}"
        )


marker_df = pd.DataFrame(
    marker_rows
)


# ================================================================
# 12. CLUSTER BOOTSTRAP BY CULTURE
# ================================================================

print()
print("=" * 88)
print("BOOTSTRAP BY CULTURE — 1,000 RESAMPLES")
print("=" * 88)


rng = np.random.default_rng(
    42
)


culture_names = np.array(
    sorted(
        oof[
            "culture"
        ].unique()
    )
)


indices_by_culture = {
    c:
        np.flatnonzero(
            oof[
                "culture"
            ].to_numpy()
            == c
        )

    for c in culture_names
}


y_all = oof[
    "DL"
].to_numpy()


m2_all = oof[
    "m2_tta8"
].to_numpy()


m3_all = oof[
    "m3_tta8"
].to_numpy()


bootstrap_rows = []


for b in range(1000):

    draw = rng.choice(
        culture_names,
        size=len(
            culture_names
        ),
        replace=True,
    )


    idx = np.concatenate(
        [
            indices_by_culture[c]
            for c in draw
        ]
    )


    yy = y_all[idx]

    p2 = m2_all[idx]
    p3 = m3_all[idx]


    r2_2 = r2_score(
        yy,
        p2,
    )

    r2_3 = r2_score(
        yy,
        p3,
    )


    mae_2 = mean_absolute_error(
        yy,
        p2,
    )

    mae_3 = mean_absolute_error(
        yy,
        p3,
    )


    bootstrap_rows.append({
        "bootstrap":
            b,

        "M2_r2":
            r2_2,

        "M3_r2":
            r2_3,

        "delta_r2_M3_M2":
            r2_3 - r2_2,

        "M2_mae":
            mae_2,

        "M3_mae":
            mae_3,

        "delta_mae_M3_M2":
            mae_3 - mae_2,
    })


boot = pd.DataFrame(
    bootstrap_rows
)


for col in [
    "delta_r2_M3_M2",
    "delta_mae_M3_M2",
]:

    values = boot[
        col
    ].to_numpy()

    low, high = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    print(
        f"{col:22s}: "
        f"{values.mean():+.6f} "
        f"95% CI "
        f"[{low:+.6f}, "
        f"{high:+.6f}]"
    )


# ================================================================
# 13. SAVE
# ================================================================

OOF_PATH = (
    OUT
    / "m3_oof_predictions.csv"
)

GLOBAL_PATH = (
    OUT
    / "m3_vs_m2_vs_m1_vs_m0_global_metrics.csv"
)

FOLD_PATH = (
    OUT
    / "m3_vs_m2_vs_m1_vs_m0_per_fold.csv"
)

FOLD_SUMMARY_PATH = (
    OUT
    / "m3_fold_summary.csv"
)

CULTURE_PATH = (
    OUT
    / "m3_vs_m2_per_culture.csv"
)

CULTURE_METRICS_PATH = (
    OUT
    / "m3_culture_metrics.csv"
)

MARKER_PATH = (
    OUT
    / "m3_marker_oof_metrics.csv"
)

BOOT_PATH = (
    OUT
    / "m3_cluster_bootstrap.csv"
)

SUMMARY_PATH = (
    OUT
    / "m3_aggregate_summary.txt"
)


oof.to_csv(
    OOF_PATH,
    index=False,
)

global_df.to_csv(
    GLOBAL_PATH,
    index=False,
)

fold_df.to_csv(
    FOLD_PATH,
    index=False,
)

fold_summary.to_csv(
    FOLD_SUMMARY_PATH,
    index=False,
)

culture.to_csv(
    CULTURE_PATH,
    index=False,
)

culture_metrics_df.to_csv(
    CULTURE_METRICS_PATH,
    index=False,
)

marker_df.to_csv(
    MARKER_PATH,
    index=False,
)

boot.to_csv(
    BOOT_PATH,
    index=False,
)


m2_global = metrics(
    oof["DL"],
    oof["m2_tta8"],
)

m3_global = metrics(
    oof["DL"],
    oof["m3_tta8"],
)


ci_r2 = np.percentile(
    boot["delta_r2_M3_M2"],
    [
        2.5,
        97.5,
    ],
)

ci_mae = np.percentile(
    boot["delta_mae_M3_M2"],
    [
        2.5,
        97.5,
    ],
)


summary_lines = [
    "=" * 88,
    "M3 OOF — FINAL RESULTS",
    "=" * 88,
    "",
    "30,877 nuclei / 48 cultures / 5 folds",
    "",
    "M2 — GLOBAL OOF",
    f"R2        : {m2_global['r2']:.6f}",
    f"MAE       : {m2_global['mae']:.6f}",
    f"Pearson   : {m2_global['pearson']:.6f}",
    f"Spearman  : {m2_global['spearman']:.6f}",
    "",
    "M3 CONVNEXT-TINY — GLOBAL OOF",
    f"R2        : {m3_global['r2']:.6f}",
    f"MAE       : {m3_global['mae']:.6f}",
    f"Pearson   : {m3_global['pearson']:.6f}",
    f"Spearman  : {m3_global['spearman']:.6f}",
    "",
    "DELTA M3-M2",
    f"R2        : {m3_global['r2'] - m2_global['r2']:+.6f}",
    f"MAE       : {m3_global['mae'] - m2_global['mae']:+.6f}",
    "",
    "CULTURE BOOTSTRAP — 95% CI",
    (
        "Delta R2  : "
        f"{boot['delta_r2_M3_M2'].mean():+.6f} "
        f"[{ci_r2[0]:+.6f}, {ci_r2[1]:+.6f}]"
    ),
    (
        "Delta MAE : "
        f"{boot['delta_mae_M3_M2'].mean():+.6f} "
        f"[{ci_mae[0]:+.6f}, {ci_mae[1]:+.6f}]"
    ),
    "",
    "PAIRED TEST BY CULTURE",
    f"M3 better: {n_better}/48 cultures",
    f"M3 worse : {n_worse}/48 cultures",
    f"Wilcoxon p  : {wilcox.pvalue:.8g}",
    "",
    "IMPORTANT",
    "M3 = ConvNeXt-Tiny multitask DL + markers.",
    "Fluorescence is never an input.",
    "Biological time is neither an input nor a target.",
]


SUMMARY_PATH.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)


print()
print("=" * 88)
print("FINAL SUMMARY")
print("=" * 88)

print(
    SUMMARY_PATH.read_text(
        encoding="utf-8"
    )
)


print()
print("=" * 88)
print("OUTPUT FILES")
print("=" * 88)


for p in [
    OOF_PATH,
    GLOBAL_PATH,
    FOLD_PATH,
    FOLD_SUMMARY_PATH,
    CULTURE_PATH,
    CULTURE_METRICS_PATH,
    MARKER_PATH,
    BOOT_PATH,
    SUMMARY_PATH,
]:

    print(
        p
    )


print()
print("DONE")
