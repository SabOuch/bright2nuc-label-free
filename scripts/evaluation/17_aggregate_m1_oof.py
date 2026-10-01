"""
Aggregate M1 out-of-fold predictions and compare them with the M0 baseline.

Purpose
-------
Collect the five held-out M1 fold prediction files, verify their alignment with
the frozen culture-wise split, merge the corresponding M0 predictions, and
compute nucleus-, fold-, and culture-level metrics.

The script also performs:
- a paired per-culture MAE comparison between M1 and M0;
- a two-sided Wilcoxon signed-rank test across cultures;
- a culture-cluster bootstrap with 1,000 resamples;
- a comparison of single-view and TTA8 M1 predictions.

Inputs
------
- ``outputs/m1/fold{0..4}/test_predictions.csv``
- ``outputs/m0/m0_oof_predictions.csv``
- ``data/processed/folds.csv``

Outputs
-------
Written under ``outputs/m1/``:
- ``m1_oof_predictions.csv``
- ``m1_vs_m0_global_metrics.csv``
- ``m1_vs_m0_per_fold.csv``
- ``m1_vs_m0_per_culture.csv``
- ``m1_vs_m0_culture_metrics.csv``
- ``m1_vs_m0_cluster_bootstrap.csv``

Reproducibility
---------------
The bootstrap uses NumPy seed 42. All evaluations are out-of-fold and preserve
the frozen culture-wise split.
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

M1_DIR = ROOT / "outputs/m1"
M0_PATH = ROOT / "outputs/m0/m0_oof_predictions.csv"
FOLDS_PATH = ROOT / "data/processed/folds.csv"

OUT = ROOT / "outputs/m1"
OUT.mkdir(parents=True, exist_ok=True)


# ================================================================
# METRICS
# ================================================================

def metrics(y, pred):
    """Return regression and correlation metrics for one prediction vector."""

    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    return {
        "r2": r2_score(y, pred),
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(
            mean_squared_error(y, pred)
        ),
        "pearson": pearsonr(y, pred).statistic,
        "spearman": spearmanr(y, pred).statistic,
    }


# ================================================================
# 1. LOAD M1 FOLDS
# ================================================================

parts = []

for fold in range(5):

    path = (
        M1_DIR
        / f"fold{fold}"
        / "test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    x = pd.read_csv(path)

    required = {
        "filename",
        "culture",
        "fold",
        "target",
        "prediction_single",
        "prediction_tta8",
    }

    missing = required - set(x.columns)

    if missing:
        raise RuntimeError(
            f"Fold {fold}: missing columns "
            f"{missing}"
        )

    if not (x["fold"] == fold).all():
        raise RuntimeError(
            f"Fold {fold}: inconsistent fold number."
        )

    if x["filename"].duplicated().any():
        raise RuntimeError(
            f"Fold {fold}: duplicate filename."
        )

    parts.append(x)


m1 = pd.concat(
    parts,
    ignore_index=True,
)


print("=" * 88)
print("M1 OOF AGGREGATION")
print("=" * 88)

print("Rows      :", len(m1))
print(
    "Nuclei    :",
    m1["filename"].nunique(),
)
print(
    "Cultures :",
    m1["culture"].nunique(),
)


if len(m1) != 30877:
    raise RuntimeError(
        f"Expected 30,877 rows, got {len(m1)}"
    )

if m1["filename"].duplicated().any():
    raise RuntimeError(
        "A nucleus appears in multiple M1 folds."
    )


# ================================================================
# 2. VERIFY AGAINST FROZEN FOLDS
# ================================================================

folds = pd.read_csv(
    FOLDS_PATH
)

check = (
    m1[
        [
            "filename",
            "culture",
            "fold",
            "target",
        ]
    ]
    .merge(
        folds[
            [
                "filename",
                "culture",
                "fold",
                "TF_class",
            ]
        ],
        on="filename",
        suffixes=("_m1", "_frozen"),
        validate="one_to_one",
    )
)

if len(check) != 30877:
    raise RuntimeError(
        "M1 does not exactly match folds.csv."
    )

if not (
    check["culture_m1"]
    == check["culture_frozen"]
).all():
    raise RuntimeError(
        "Culture mismatch."
    )

if not (
    check["fold_m1"]
    == check["fold_frozen"]
).all():
    raise RuntimeError(
        "Fold mismatch."
    )

max_target_diff = np.max(
    np.abs(
        check["target"].to_numpy()
        - check["TF_class"].to_numpy()
    )
)

print(
    "Max target difference vs folds.csv :",
    f"{max_target_diff:.12g}",
)

if max_target_diff > 1e-6:
    raise RuntimeError(
        "Target mismatch."
    )


# ================================================================
# 3. LOAD M0
# ================================================================

m0 = pd.read_csv(
    M0_PATH
)

required_m0 = {
    "filename",
    "culture",
    "TF_class",
    "fold",
    "pred_M0_119",
}

missing = required_m0 - set(m0.columns)

if missing:
    raise RuntimeError(
        f"Missing M0 columns: {missing}"
    )

if len(m0) != 30877:
    raise RuntimeError(
        "M0 does not contain 30,877 rows."
    )


# ================================================================
# 4. MERGE M0 + M1
# ================================================================

oof = (
    m1[
        [
            "filename",
            "culture",
            "fold",
            "target",
            "prediction_single",
            "prediction_tta8",
        ]
    ]
    .merge(
        m0[
            [
                "filename",
                "pred_M0_119",
            ]
        ],
        on="filename",
        validate="one_to_one",
    )
)


if len(oof) != 30877:
    raise RuntimeError(
        "Incomplete M0/M1 merge."
    )


# ================================================================
# 5. GLOBAL OOF METRICS
# ================================================================

y = oof["target"].to_numpy()

results = {}

results["M0_119"] = metrics(
    y,
    oof["pred_M0_119"],
)

results["M1_single"] = metrics(
    y,
    oof["prediction_single"],
)

results["M1_TTA8"] = metrics(
    y,
    oof["prediction_tta8"],
)


print("\n" + "=" * 88)
print("GLOBAL OOF RESULTS — 30,877 NUCLEI")
print("=" * 88)

global_rows = []

for model_name, r in results.items():

    global_rows.append({
        "model": model_name,
        **r,
    })

    print(
        f"\n{model_name}"
    )

    for key, value in r.items():
        print(
            f"{key:10s}: {value:.6f}"
        )


# ================================================================
# 6. PER-FOLD COMPARISON
# ================================================================

fold_rows = []

for fold in range(5):

    sub = oof[
        oof["fold"] == fold
    ]

    for model_name, col in [
        ("M0_119", "pred_M0_119"),
        ("M1_single", "prediction_single"),
        ("M1_TTA8", "prediction_tta8"),
    ]:

        r = metrics(
            sub["target"],
            sub[col],
        )

        fold_rows.append({
            "fold": fold,
            "model": model_name,
            "n": len(sub),
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
# 7. MEAN +/- SD ACROSS FOLDS
# ================================================================

fold_summary = (
    fold_df
    .groupby("model")
    .agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        pearson_mean=("pearson", "mean"),
        pearson_std=("pearson", "std"),
    )
)


print("\n" + "=" * 88)
print("MEAN +/- SD ACROSS THE FIVE FOLDS")
print("=" * 88)

print(
    fold_summary.round(6).to_string()
)


# ================================================================
# 8. CULTURE-LEVEL RESULTS
# ================================================================

culture = (
    oof
    .groupby("culture")
    .agg(
        n=("filename", "size"),
        target_mean=("target", "mean"),
        m0_mean=("pred_M0_119", "mean"),
        m1_single_mean=("prediction_single", "mean"),
        m1_tta_mean=("prediction_tta8", "mean"),
        m0_nucleus_mae=(
            "pred_M0_119",
            lambda x: 0.0,
        ),
    )
    .drop(columns=["m0_nucleus_mae"])
    .reset_index()
)


# mean absolute nucleus error inside each culture
culture_errors = []

for culture_name, sub in oof.groupby(
    "culture",
    sort=True,
):

    culture_errors.append({
        "culture": culture_name,

        "n": len(sub),

        "m0_mae": np.mean(
            np.abs(
                sub["target"]
                - sub["pred_M0_119"]
            )
        ),

        "m1_single_mae": np.mean(
            np.abs(
                sub["target"]
                - sub["prediction_single"]
            )
        ),

        "m1_tta_mae": np.mean(
            np.abs(
                sub["target"]
                - sub["prediction_tta8"]
            )
        ),
    })


culture_errors = pd.DataFrame(
    culture_errors
)

culture = culture.merge(
    culture_errors,
    on=["culture", "n"],
    validate="one_to_one",
)


culture_results = {
    "M0_119": metrics(
        culture["target_mean"],
        culture["m0_mean"],
    ),

    "M1_single": metrics(
        culture["target_mean"],
        culture["m1_single_mean"],
    ),

    "M1_TTA8": metrics(
        culture["target_mean"],
        culture["m1_tta_mean"],
    ),
}


print("\n" + "=" * 88)
print("CULTURE-LEVEL RESULTS — 48 CULTURES")
print("=" * 88)

for name, r in culture_results.items():

    print(
        f"\n{name}"
    )

    for key, value in r.items():
        print(
            f"{key:10s}: {value:.6f}"
        )


# ================================================================
# 9. PAIRED CULTURE ANALYSIS: M1_TTA vs M0
# ================================================================

delta = (
    culture["m1_tta_mae"]
    - culture["m0_mae"]
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
    culture["m1_tta_mae"],
    culture["m0_mae"],
    alternative="two-sided",
)


print("\n" + "=" * 88)
print("M1_TTA8 vs M0 — MEAN ERROR BY CULTURE")
print("=" * 88)

print(
    "Cultures where M1 has lower MAE       :",
    f"{n_better}/48",
)

print(
    "Cultures where M1 has higher MAE      :",
    f"{n_worse}/48",
)

print(
    "Ties                                  :",
    n_equal,
)

print(
    "Mean culture-level MAE delta (M1-M0)  :",
    f"{delta.mean():+.6f}",
)

print(
    "Median culture-level MAE delta (M1-M0):",
    f"{delta.median():+.6f}",
)

print(
    "Wilcoxon paired two-sided statistic  :",
    f"{wilcox.statistic:.6f}",
)

print(
    "Wilcoxon paired two-sided p          :",
    f"{wilcox.pvalue:.8g}",
)


# ================================================================
# 10. CLUSTER BOOTSTRAP BY CULTURE
# ================================================================

print("\n" + "=" * 88)
print("CULTURE-CLUSTER BOOTSTRAP — 1,000 RESAMPLES")
print("=" * 88)

rng = np.random.default_rng(42)

culture_names = np.array(
    sorted(
        oof["culture"].unique()
    )
)

indices_by_culture = {
    c: np.flatnonzero(
        oof["culture"].to_numpy()
        == c
    )
    for c in culture_names
}

bootstrap_rows = []

for b in range(1000):

    draw = rng.choice(
        culture_names,
        size=len(culture_names),
        replace=True,
    )

    idx = np.concatenate(
        [
            indices_by_culture[c]
            for c in draw
        ]
    )

    yy = y[idx]

    m0_pred = (
        oof["pred_M0_119"]
        .to_numpy()[idx]
    )

    m1_pred = (
        oof["prediction_tta8"]
        .to_numpy()[idx]
    )

    m0_r2 = r2_score(
        yy,
        m0_pred,
    )

    m1_r2 = r2_score(
        yy,
        m1_pred,
    )

    m0_mae = mean_absolute_error(
        yy,
        m0_pred,
    )

    m1_mae = mean_absolute_error(
        yy,
        m1_pred,
    )

    bootstrap_rows.append({
        "bootstrap": b,
        "m0_r2": m0_r2,
        "m1_r2": m1_r2,
        "delta_r2": m1_r2 - m0_r2,
        "m0_mae": m0_mae,
        "m1_mae": m1_mae,
        "delta_mae": m1_mae - m0_mae,
    })


boot = pd.DataFrame(
    bootstrap_rows
)


def ci95(series):
    """Return the percentile-based 95% interval for bootstrap values."""

    return np.percentile(
        series,
        [2.5, 97.5],
    )


r2_ci = ci95(
    boot["delta_r2"]
)

mae_ci = ci95(
    boot["delta_mae"]
)

print(
    "Delta R2 M1-M0 :",
    f"{boot['delta_r2'].mean():+.6f}",
    f"95% CI [{r2_ci[0]:+.6f}, {r2_ci[1]:+.6f}]",
)

print(
    "Delta MAE M1-M0:",
    f"{boot['delta_mae'].mean():+.6f}",
    f"95% CI [{mae_ci[0]:+.6f}, {mae_ci[1]:+.6f}]",
)


# ================================================================
# 11. TTA EFFECT
# ================================================================

tta_delta_r2 = []

for fold in range(5):

    sub = fold_df[
        fold_df["fold"] == fold
    ]

    single_r2 = float(
        sub.loc[
            sub["model"] == "M1_single",
            "r2",
        ].iloc[0]
    )

    tta_r2 = float(
        sub.loc[
            sub["model"] == "M1_TTA8",
            "r2",
        ].iloc[0]
    )

    tta_delta_r2.append(
        tta_r2 - single_r2
    )


print("\n" + "=" * 88)
print("TTA EFFECT")
print("=" * 88)

print(
    "R2 gain by fold:",
    ", ".join(
        f"{x:+.4f}"
        for x in tta_delta_r2
    ),
)

print(
    "Mean R2 gain     :",
    f"{np.mean(tta_delta_r2):+.6f}",
)


# ================================================================
# 12. SAVE
# ================================================================

global_df = pd.DataFrame(
    [
        {
            "model": name,
            **r,
        }
        for name, r in results.items()
    ]
)

culture_global_df = pd.DataFrame(
    [
        {
            "model": name,
            **r,
        }
        for name, r in culture_results.items()
    ]
)

oof.to_csv(
    OUT / "m1_oof_predictions.csv",
    index=False,
)

global_df.to_csv(
    OUT / "m1_vs_m0_global_metrics.csv",
    index=False,
)

fold_df.to_csv(
    OUT / "m1_vs_m0_per_fold.csv",
    index=False,
)

culture.to_csv(
    OUT / "m1_vs_m0_per_culture.csv",
    index=False,
)

culture_global_df.to_csv(
    OUT / "m1_vs_m0_culture_metrics.csv",
    index=False,
)

boot.to_csv(
    OUT / "m1_vs_m0_cluster_bootstrap.csv",
    index=False,
)


print("\n" + "=" * 88)
print("OUTPUT FILES")
print("=" * 88)

for path in [
    OUT / "m1_oof_predictions.csv",
    OUT / "m1_vs_m0_global_metrics.csv",
    OUT / "m1_vs_m0_per_fold.csv",
    OUT / "m1_vs_m0_per_culture.csv",
    OUT / "m1_vs_m0_culture_metrics.csv",
    OUT / "m1_vs_m0_cluster_bootstrap.csv",
]:
    print(path)

print("\nDONE")
