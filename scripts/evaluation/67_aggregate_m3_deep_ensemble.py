"""
Aggregate the five-seed M3 deep ensemble and quantify predictive uncertainty.

Purpose
-------
Align M3 predictions from five training seeds (42-46) within each frozen
culture-wise fold, compute the per-nucleus ensemble mean and sample standard
deviation, and evaluate the ensemble on the full OOF set.

The script also computes:
- nucleus-level metrics for each seed and the ensemble mean;
- culture-level ensemble metrics;
- uncertainty/error association;
- uncertainty deciles;
- selective-prediction (abstention) curves.

Inputs
------
- seed 42: ``outputs/m3/fold{0..4}/test_predictions.csv``
- seeds 43-46:
  ``outputs/m3_ensemble/fold{0..4}/seed{seed}/test_predictions.csv``

Outputs
-------
Written under ``outputs/m3_ensemble/analysis/``:
- ``m3_ensemble_oof_predictions.csv``
- ``m3_ensemble_metrics.csv``
- ``m3_ensemble_culture_predictions.csv``
- ``m3_uncertainty_deciles.csv``
- ``m3_abstention_curve.csv``
- ``m3_ensemble_summary.txt``

Notes
-----
Ensemble uncertainty is the sample standard deviation across five predictions
(``ddof=1``). Selective-prediction metrics are computed only on retained nuclei
after removing the most uncertain predictions; rejected cases are not solved
or re-predicted.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "outputs/m3_ensemble/analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [42, 43, 44, 45, 46]
FOLDS = [0, 1, 2, 3, 4]


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
        "r2":
            r2_score(
                y,
                pred,
            ),

        "mae":
            mean_absolute_error(
                y,
                pred,
            ),

        "rmse":
            np.sqrt(
                mean_squared_error(
                    y,
                    pred,
                )
            ),

        "pearson":
            pearsonr(
                y,
                pred,
            ).statistic,

        "spearman":
            spearmanr(
                y,
                pred,
            ).statistic,
    }


def format_pvalue(value):
    """Format very small p-values without reporting numerical underflow as p=0."""
    value = float(value)
    if value == 0.0:
        return "<1e-300"
    return f"{value:.3e}"


# ================================================================
# LOAD ALL SEEDS
# ================================================================

all_folds = []


for fold in FOLDS:

    print()
    print("=" * 100)
    print(f"FOLD {fold}")
    print("=" * 100)

    seed_tables = []


    for seed in SEEDS:

        if seed == 42:

            path = (
                ROOT
                / f"outputs/m3/fold{fold}"
                / "test_predictions.csv"
            )

        else:

            path = (
                ROOT
                / "outputs/m3_ensemble"
                / f"fold{fold}"
                / f"seed{seed}"
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
            "prediction_tta8",
        }

        missing = (
            required
            - set(x.columns)
        )

        if missing:
            raise RuntimeError(
                f"Fold {fold} seed {seed}: "
                f"missing columns {missing}"
            )


        if not (
            x["fold"] == fold
        ).all():

            raise RuntimeError(
                f"Inconsistent fold: "
                f"fold {fold}, seed {seed}"
            )


        if x["filename"].duplicated().any():
            raise RuntimeError(
                f"Duplicate filename: "
                f"fold {fold}, seed {seed}"
            )


        x = (
            x[
                [
                    "filename",
                    "culture",
                    "fold",
                    "target_DL",
                    "prediction_tta8",
                ]
            ]
            .rename(
                columns={
                    "prediction_tta8":
                        f"pred_seed{seed}",

                    "culture":
                        f"culture_seed{seed}",

                    "fold":
                        f"fold_seed{seed}",

                    "target_DL":
                        f"target_seed{seed}",
                }
            )
        )


        seed_tables.append(
            x
        )

        print(
            f"seed {seed}: "
            f"{len(x):5d} nuclei"
        )


    # ------------------------------------------------------------
    # ALIGN SEEDS
    # ------------------------------------------------------------

    merged = seed_tables[0]

    for x in seed_tables[1:]:

        merged = merged.merge(
            x,
            on="filename",
            how="inner",
            validate="one_to_one",
        )


    expected_n = len(
        seed_tables[0]
    )

    if len(merged) != expected_n:
        raise RuntimeError(
            f"Fold {fold}: incomplete merge "
            f"{len(merged)} vs {expected_n}"
        )


    # ------------------------------------------------------------
    # VERIFY SAME CULTURE / FOLD / TARGET
    # ------------------------------------------------------------

    ref_culture = (
        merged["culture_seed42"]
    )

    ref_fold = (
        merged["fold_seed42"]
    )

    ref_target = (
        merged["target_seed42"]
        .to_numpy(
            dtype=np.float64
        )
    )


    for seed in SEEDS[1:]:

        if not (
            ref_culture
            ==
            merged[
                f"culture_seed{seed}"
            ]
        ).all():

            raise RuntimeError(
                f"Culture mismatch "
                f"fold {fold}, seed {seed}"
            )


        if not (
            ref_fold
            ==
            merged[
                f"fold_seed{seed}"
            ]
        ).all():

            raise RuntimeError(
                f"Fold mismatch "
                f"fold {fold}, seed {seed}"
            )


        target = (
            merged[
                f"target_seed{seed}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        max_diff = float(
            np.max(
                np.abs(
                    ref_target
                    - target
                )
            )
        )

        if max_diff > 1e-6:
            raise RuntimeError(
                f"Target mismatch "
                f"fold {fold}, seed {seed}: "
                f"{max_diff}"
            )


    # ------------------------------------------------------------
    # CLEAN TABLE
    # ------------------------------------------------------------

    keep = pd.DataFrame({
        "filename":
            merged["filename"],

        "culture":
            merged["culture_seed42"],

        "fold":
            merged["fold_seed42"],

        "DL":
            merged["target_seed42"],
    })


    for seed in SEEDS:

        keep[
            f"pred_seed{seed}"
        ] = (
            merged[
                f"pred_seed{seed}"
            ]
        )


    pred_cols = [
        f"pred_seed{s}"
        for s in SEEDS
    ]


    pred_matrix = (
        keep[pred_cols]
        .to_numpy(
            dtype=np.float64
        )
    )


    keep["ensemble_mean"] = (
        pred_matrix.mean(
            axis=1
        )
    )


    keep["ensemble_std"] = (
        pred_matrix.std(
            axis=1,
            ddof=1,
        )
    )


    keep["ensemble_min"] = (
        pred_matrix.min(
            axis=1
        )
    )


    keep["ensemble_max"] = (
        pred_matrix.max(
            axis=1
        )
    )


    keep["ensemble_range"] = (
        keep["ensemble_max"]
        -
        keep["ensemble_min"]
    )


    keep["abs_error"] = (
        np.abs(
            keep["DL"]
            -
            keep["ensemble_mean"]
        )
    )


    all_folds.append(
        keep
    )


# ================================================================
# GLOBAL OOF
# ================================================================

oof = pd.concat(
    all_folds,
    ignore_index=True,
)


if len(oof) != 30877:
    raise RuntimeError(
        f"Expected 30877 nuclei, "
        f"got {len(oof)}"
    )


if oof["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filename in ensemble OOF."
    )


if oof["culture"].nunique() != 48:
    raise RuntimeError(
        f"Expected 48 cultures, "
        f"got {oof['culture'].nunique()}"
    )


# ================================================================
# METRICS — EACH SEED + ENSEMBLE
# ================================================================

rows = []


for seed in SEEDS:

    r = metrics(
        oof["DL"],
        oof[f"pred_seed{seed}"],
    )

    rows.append({
        "model":
            f"seed{seed}",

        **r,
    })


ensemble_metrics = metrics(
    oof["DL"],
    oof["ensemble_mean"],
)


rows.append({
    "model":
        "ensemble_mean",

    **ensemble_metrics,
})


metrics_df = pd.DataFrame(
    rows
)


# ================================================================
# CULTURE LEVEL
# ================================================================

culture = (
    oof
    .groupby(
        "culture",
        sort=True,
    )
    .agg(
        n=(
            "filename",
            "size",
        ),

        DL=(
            "DL",
            "mean",
        ),

        ensemble_mean=(
            "ensemble_mean",
            "mean",
        ),

        mean_uncertainty=(
            "ensemble_std",
            "mean",
        ),
    )
    .reset_index()
)


culture_metrics = metrics(
    culture["DL"],
    culture["ensemble_mean"],
)


# ================================================================
# UNCERTAINTY VS ERROR
# ================================================================

rho = spearmanr(
    oof["ensemble_std"],
    oof["abs_error"],
)

pear = pearsonr(
    oof["ensemble_std"],
    oof["abs_error"],
)


# ================================================================
# UNCERTAINTY DECILES
# ================================================================

oof["uncertainty_decile"] = pd.qcut(
    oof["ensemble_std"],
    q=10,
    labels=False,
    duplicates="drop",
) + 1


deciles = (
    oof
    .groupby(
        "uncertainty_decile"
    )
    .agg(
        n=(
            "filename",
            "size",
        ),

        uncertainty_mean=(
            "ensemble_std",
            "mean",
        ),

        uncertainty_median=(
            "ensemble_std",
            "median",
        ),

        mae=(
            "abs_error",
            "mean",
        ),

        median_abs_error=(
            "abs_error",
            "median",
        ),
    )
    .reset_index()
)


# ================================================================
# ABSTENTION CURVE
# ================================================================

abstention_rows = []


for reject_pct in [
    0,
    5,
    10,
    20,
    30,
    40,
    50,
]:

    if reject_pct == 0:

        kept = oof.copy()

        threshold = float(
            oof["ensemble_std"].max()
        )

    else:

        threshold = float(
            np.percentile(
                oof["ensemble_std"],
                100 - reject_pct,
            )
        )

        kept = oof[
            oof["ensemble_std"]
            <= threshold
        ]


    r = metrics(
        kept["DL"],
        kept["ensemble_mean"],
    )


    abstention_rows.append({
        "reject_pct":
            reject_pct,

        "kept_n":
            len(kept),

        "kept_pct":
            100.0
            * len(kept)
            / len(oof),

        "uncertainty_threshold":
            threshold,

        **r,
    })


abstention = pd.DataFrame(
    abstention_rows
)


# ================================================================
# SAVE
# ================================================================

OOF_PATH = (
    OUT
    / "m3_ensemble_oof_predictions.csv"
)

METRICS_PATH = (
    OUT
    / "m3_ensemble_metrics.csv"
)

CULTURE_PATH = (
    OUT
    / "m3_ensemble_culture_predictions.csv"
)

DECILES_PATH = (
    OUT
    / "m3_uncertainty_deciles.csv"
)

ABSTENTION_PATH = (
    OUT
    / "m3_abstention_curve.csv"
)

SUMMARY_PATH = (
    OUT
    / "m3_ensemble_summary.txt"
)


oof.to_csv(
    OOF_PATH,
    index=False,
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)

culture.to_csv(
    CULTURE_PATH,
    index=False,
)

deciles.to_csv(
    DECILES_PATH,
    index=False,
)

abstention.to_csv(
    ABSTENTION_PATH,
    index=False,
)


# ================================================================
# TERMINAL
# ================================================================

print()
print("=" * 100)
print("M3 DEEP ENSEMBLE — GLOBAL OOF")
print("=" * 100)

print(
    metrics_df
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("ENSEMBLE — CULTURE LEVEL")
print("=" * 100)

for k, v in culture_metrics.items():
    print(
        f"{k:10s}: {v:.6f}"
    )


print()
print("=" * 100)
print("UNCERTAINTY ↔ ABSOLUTE ERROR")
print("=" * 100)

print(
    f"Spearman rho : "
    f"{rho.statistic:+.6f}"
)

print(
    "p-value      :",
    format_pvalue(rho.pvalue),
)

print(
    f"Pearson r    : "
    f"{pear.statistic:+.6f}"
)

print(
    "p-value      :",
    format_pvalue(pear.pvalue),
)


print()
print("=" * 100)
print("UNCERTAINTY DECILES")
print("=" * 100)

print(
    deciles
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("ABSTENTION")
print("=" * 100)

print(
    abstention[
        [
            "reject_pct",
            "kept_n",
            "mae",
            "r2",
            "pearson",
            "spearman",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ================================================================
# SUMMARY TXT
# ================================================================

seed42 = metrics_df[
    metrics_df["model"] == "seed42"
].iloc[0]


summary = [
    "=" * 100,
    "M3 DEEP ENSEMBLE — FINAL SUMMARY",
    "=" * 100,
    "",
    "30877 nuclei / 48 cultures / 5 folds / 5 seeds",
    "",
    "SEED42 ORIGINAL",
    f"R2  : {seed42['r2']:.6f}",
    f"MAE : {seed42['mae']:.6f}",
    "",
    "ENSEMBLE",
    f"R2        : {ensemble_metrics['r2']:.6f}",
    f"MAE       : {ensemble_metrics['mae']:.6f}",
    f"RMSE      : {ensemble_metrics['rmse']:.6f}",
    f"Pearson   : {ensemble_metrics['pearson']:.6f}",
    f"Spearman  : {ensemble_metrics['spearman']:.6f}",
    "",
    "DELTA ENSEMBLE - SEED42",
    (
        f"Delta R2  : "
        f"{ensemble_metrics['r2'] - seed42['r2']:+.6f}"
    ),
    (
        f"Delta MAE : "
        f"{ensemble_metrics['mae'] - seed42['mae']:+.6f}"
    ),
    "",
    "CULTURE LEVEL ENSEMBLE",
    f"R2        : {culture_metrics['r2']:.6f}",
    f"MAE       : {culture_metrics['mae']:.6f}",
    f"Pearson   : {culture_metrics['pearson']:.6f}",
    f"Spearman  : {culture_metrics['spearman']:.6f}",
    "",
    "UNCERTAINTY VS ABSOLUTE ERROR",
    (
        f"Spearman rho : "
        f"{rho.statistic:+.6f}"
    ),
    f"p-value      : {format_pvalue(rho.pvalue)}",
    "",
    "ABSTENTION",
    abstention[
        [
            "reject_pct",
            "kept_n",
            "mae",
            "r2",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    ),
]


SUMMARY_PATH.write_text(
    "\n".join(summary),
    encoding="utf-8",
)


print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(OOF_PATH)
print(METRICS_PATH)
print(CULTURE_PATH)
print(DECILES_PATH)
print(ABSTENTION_PATH)
print(SUMMARY_PATH)

print()
print("DONE")
