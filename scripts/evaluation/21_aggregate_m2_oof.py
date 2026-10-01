"""
Aggregate M2 out-of-fold predictions and compare M2 with M1 and M0.

Purpose
-------
Collect the five held-out M2 fold prediction files, align them with the existing
M0/M1 OOF table, verify target/culture/fold consistency, and compute
nucleus-, fold-, culture-, and auxiliary-marker metrics.

The script also performs a paired per-culture M2-vs-M1 MAE analysis and a
culture-cluster bootstrap with 1,000 resamples.

Inputs
------
- ``outputs/m1/m1_oof_predictions.csv``
- ``outputs/m2/fold{0..4}/test_predictions.csv``

Outputs
-------
Written under ``outputs/m2/``:
- ``m2_oof_predictions.csv``
- ``m2_vs_m1_vs_m0_global_metrics.csv``
- ``m2_vs_m1_vs_m0_per_fold.csv``
- ``m2_vs_m1_per_culture.csv``
- ``m2_culture_metrics.csv``
- ``m2_marker_oof_metrics.csv``
- ``m2_cluster_bootstrap.csv``

Notes
-----
OCT4, FOXA2, and SOX17 are auxiliary supervision targets only. They are not
model inputs at inference time.

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


ROOT = Path(__file__).resolve().parents[2]

M1_PATH = (
    ROOT
    / "outputs/m1/m1_oof_predictions.csv"
)

M2_DIR = (
    ROOT
    / "outputs/m2"
)

OUT = M2_DIR
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
# 1. LOAD M2 FOLDS
# ================================================================

parts = []

for fold in range(5):

    path = (
        M2_DIR
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
            f"missing columns "
            f"{missing}"
        )

    if not (
        x["fold"] == fold
    ).all():
        raise RuntimeError(
            f"Fold {fold}: "
            "inconsistent fold number."
        )

    if x[
        "filename"
    ].duplicated().any():

        raise RuntimeError(
            f"Fold {fold}: "
            "duplicate filename."
        )

    parts.append(
        x
    )


m2 = pd.concat(
    parts,
    ignore_index=True,
)


print("=" * 88)
print("M2 OOF AGGREGATION")
print("=" * 88)

print(
    "Rows      :",
    len(m2),
)

print(
    "Nuclei     :",
    m2["filename"].nunique(),
)

print(
    "Cultures  :",
    m2["culture"].nunique(),
)


if len(m2) != 30877:
    raise RuntimeError(
        f"Expected 30,877 rows, "
        f"got {len(m2)}."
    )


if m2[
    "filename"
].duplicated().any():

    raise RuntimeError(
        "A nucleus appears in "
        "multiple M2 folds."
    )


# ================================================================
# 2. LOAD PREVIOUS OOF TABLE
# ================================================================

m1 = pd.read_csv(
    M1_PATH
)


required_m1 = {
    "filename",
    "culture",
    "fold",
    "target",
    "pred_M0_119",
    "prediction_single",
    "prediction_tta8",
}

missing = (
    required_m1
    - set(m1.columns)
)

if missing:
    raise RuntimeError(
        f"Missing M1 columns: "
        f"{missing}"
    )


if len(m1) != 30877:
    raise RuntimeError(
        "M1 OOF does not contain "
        "30,877 rows."
    )


m1 = m1.rename(
    columns={
        "prediction_single":
            "m1_single",

        "prediction_tta8":
            "m1_tta8",

        "target":
            "DL",
    }
)


# ================================================================
# 3. MERGE M1 + M2
# ================================================================

m2small = m2[
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
        "prediction_single":
            "m2_single",

        "prediction_tta8":
            "m2_tta8",
    }
)


oof = m1.merge(
    m2small,
    on="filename",
    suffixes=(
        "_m1",
        "_m2",
    ),
    validate="one_to_one",
)


if len(oof) != 30877:
    raise RuntimeError(
        "Incomplete M1/M2 merge."
    )


if not (
    oof["culture_m1"]
    == oof["culture_m2"]
).all():

    raise RuntimeError(
        "Culture mismatch "
        "between M1 and M2."
    )


if not (
    oof["fold_m1"]
    == oof["fold_m2"]
).all():

    raise RuntimeError(
        "Fold mismatch "
        "between M1 and M2."
    )


target_diff = np.max(
    np.abs(
        oof["DL"].to_numpy(
            dtype=np.float64
        )
        -
        oof["target_DL"].to_numpy(
            dtype=np.float64
        )
    )
)


print(
    "Maximum M1/M2 target difference:",
    f"{target_diff:.12g}",
)


if target_diff > 1e-6:
    raise RuntimeError(
        "Target DL mismatch."
    )


oof = oof.rename(
    columns={
        "culture_m1":
            "culture",

        "fold_m1":
            "fold",
    }
)


oof = oof.drop(
    columns=[
        "culture_m2",
        "fold_m2",
        "target_DL",
    ]
)


# ================================================================
# 4. GLOBAL OOF METRICS
# ================================================================

y = oof[
    "DL"
].to_numpy()


models = {
    "M0_119":
        "pred_M0_119",

    "M1_TTA8":
        "m1_tta8",

    "M2_markers_TTA8":
        "m2_tta8",
}


global_rows = []


print("\n" + "=" * 88)
print("GLOBAL OOF RESULTS — 30,877 NUCLEI")
print("=" * 88)


for name, col in models.items():

    r = metrics(
        y,
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
# 5. PER-FOLD
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


print("\n" + "=" * 88)
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


print("\n" + "=" * 88)
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
# 6. FOLD MEAN +/- SD
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
    )
)


print("\n" + "=" * 88)
print("MEAN +/- SD ACROSS THE FIVE FOLDS")
print("=" * 88)

print(
    fold_summary
    .round(6)
    .to_string()
)


# ================================================================
# 7. CULTURE-LEVEL DL
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
    )
    .reset_index()
)


culture_metrics = {
    "M0_119":
        metrics(
            culture["DL"],
            culture["M0"],
        ),

    "M1_TTA8":
        metrics(
            culture["DL"],
            culture["M1"],
        ),

    "M2_markers_TTA8":
        metrics(
            culture["DL"],
            culture["M2"],
        ),
}


print("\n" + "=" * 88)
print("CULTURE-LEVEL RESULTS — 48 CULTURES")
print("=" * 88)


for name, r in culture_metrics.items():

    print(
        f"\n{name}"
    )

    for k, v in r.items():

        print(
            f"{k:10s}: "
            f"{v:.6f}"
        )


# ================================================================
# 8. PER-CULTURE NUCLEUS MAE
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
                    - sub[
                        "pred_M0_119"
                    ]
                )
            ),

        "M1_mae":
            np.mean(
                np.abs(
                    sub["DL"]
                    - sub[
                        "m1_tta8"
                    ]
                )
            ),

        "M2_mae":
            np.mean(
                np.abs(
                    sub["DL"]
                    - sub[
                        "m2_tta8"
                    ]
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
# 9. M2 VS M1 — PAIRED CULTURE TEST
# ================================================================

delta = (
    culture["M2_mae"]
    - culture["M1_mae"]
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
    culture["M2_mae"],
    culture["M1_mae"],
    alternative="two-sided",
)


print("\n" + "=" * 88)
print("M2 vs M1 — MEAN NUCLEUS-LEVEL MAE BY CULTURE")
print("=" * 88)

print(
    "Cultures improved by M2:",
    f"{n_better}/48",
)

print(
    "Cultures degraded by M2:",
    f"{n_worse}/48",
)

print(
    "Ties                       :",
    n_equal,
)

print(
    "Mean M2-M1 delta           :",
    f"{delta.mean():+.6f}",
)

print(
    "Median M2-M1 delta         :",
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
# 10. MARKER PERFORMANCE — M2
# ================================================================

print("\n" + "=" * 88)
print("AUXILIARY-MARKER PERFORMANCE — OOF")
print("=" * 88)


marker_rows = []


for marker in [
    "OCT4",
    "FOXA2",
    "SOX17",
]:

    target_col = (
        f"target_{marker}_norm"
    )

    pred_col = (
        f"pred_{marker}_norm"
    )

    r = metrics(
        oof[target_col],
        oof[pred_col],
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
# 11. CLUSTER BOOTSTRAP BY CULTURE
# ================================================================

print("\n" + "=" * 88)
print("CULTURE-CLUSTER BOOTSTRAP — 1,000 RESAMPLES")
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


m0_all = oof[
    "pred_M0_119"
].to_numpy()


m1_all = oof[
    "m1_tta8"
].to_numpy()


m2_all = oof[
    "m2_tta8"
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

    p0 = m0_all[idx]
    p1 = m1_all[idx]
    p2 = m2_all[idx]


    r2_0 = r2_score(
        yy,
        p0,
    )

    r2_1 = r2_score(
        yy,
        p1,
    )

    r2_2 = r2_score(
        yy,
        p2,
    )


    mae_0 = mean_absolute_error(
        yy,
        p0,
    )

    mae_1 = mean_absolute_error(
        yy,
        p1,
    )

    mae_2 = mean_absolute_error(
        yy,
        p2,
    )


    bootstrap_rows.append({
        "bootstrap":
            b,

        "M0_r2":
            r2_0,

        "M1_r2":
            r2_1,

        "M2_r2":
            r2_2,

        "delta_r2_M2_M1":
            r2_2 - r2_1,

        "delta_r2_M2_M0":
            r2_2 - r2_0,

        "M0_mae":
            mae_0,

        "M1_mae":
            mae_1,

        "M2_mae":
            mae_2,

        "delta_mae_M2_M1":
            mae_2 - mae_1,

        "delta_mae_M2_M0":
            mae_2 - mae_0,
    })


boot = pd.DataFrame(
    bootstrap_rows
)


def report_delta(
    col,
):
    """Print the mean bootstrap delta and its percentile 95% interval."""

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


report_delta(
    "delta_r2_M2_M1"
)

report_delta(
    "delta_mae_M2_M1"
)

report_delta(
    "delta_r2_M2_M0"
)

report_delta(
    "delta_mae_M2_M0"
)


# ================================================================
# 12. SAVE
# ================================================================

culture_global_df = pd.DataFrame(
    [
        {
            "model":
                name,
            **r,
        }

        for name, r
        in culture_metrics.items()
    ]
)


oof.to_csv(
    OUT
    / "m2_oof_predictions.csv",
    index=False,
)


global_df.to_csv(
    OUT
    / "m2_vs_m1_vs_m0_global_metrics.csv",
    index=False,
)


fold_df.to_csv(
    OUT
    / "m2_vs_m1_vs_m0_per_fold.csv",
    index=False,
)


culture.to_csv(
    OUT
    / "m2_vs_m1_per_culture.csv",
    index=False,
)


culture_global_df.to_csv(
    OUT
    / "m2_culture_metrics.csv",
    index=False,
)


marker_df.to_csv(
    OUT
    / "m2_marker_oof_metrics.csv",
    index=False,
)


boot.to_csv(
    OUT
    / "m2_cluster_bootstrap.csv",
    index=False,
)


print("\n" + "=" * 88)
print("OUTPUT FILES")
print("=" * 88)


for p in [
    OUT
    / "m2_oof_predictions.csv",

    OUT
    / "m2_vs_m1_vs_m0_global_metrics.csv",

    OUT
    / "m2_vs_m1_vs_m0_per_fold.csv",

    OUT
    / "m2_vs_m1_per_culture.csv",

    OUT
    / "m2_culture_metrics.csv",

    OUT
    / "m2_marker_oof_metrics.csv",

    OUT
    / "m2_cluster_bootstrap.csv",
]:

    print(
        p
    )


print("\nDONE")
