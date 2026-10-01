"""
Build the frozen M0-M4 static comparison table.

Purpose
-------
Merge the M0-M3 OOF table with M4 DINOv2+Ridge predictions, verify exact
culture/fold alignment and target consistency, and compute comparable
nucleus-, culture-, and fold-level metrics for M0-M4.

Inputs
------
- ``outputs/m3/m3_oof_predictions.csv``
- ``outputs/m4/m4_oof_predictions.csv``

Outputs
-------
Written under ``outputs/model_comparison/``:
- ``m0_m4_final_comparison.csv``
- ``m0_m4_per_fold.csv``
- ``m0_m4_culture_predictions.csv``
- ``m0_m4_final_comparison.md``
- ``m0_m4_final_summary.txt``

Methodological notes
--------------------
M2 and M3 use OCT4/FOXA2/SOX17 only as auxiliary training supervision.
M4 is a frozen DINOv2 representation followed by a Ridge regression head and
serves as an independent representation comparison.

The metadata table records ImageNet pretraining for M1, M2, and M3, matching
the actual training configurations.
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

M3_OOF = (
    ROOT
    / "outputs/m3/m3_oof_predictions.csv"
)

M4_OOF = (
    ROOT
    / "outputs/m4/m4_oof_predictions.csv"
)

OUT = (
    ROOT
    / "outputs/model_comparison"
)

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


# ================================================================
# LOAD M0-M3
# ================================================================

if not M3_OOF.exists():
    raise FileNotFoundError(
        M3_OOF
    )

base = pd.read_csv(
    M3_OOF
)

required_base = {
    "filename",
    "culture",
    "fold",
    "DL",
    "pred_M0_119",
    "m1_tta8",
    "m2_tta8",
    "m3_tta8",
}

missing = (
    required_base
    - set(base.columns)
)

if missing:
    raise RuntimeError(
        f"Missing M0-M3 columns: {missing}"
    )

if len(base) != 30877:
    raise RuntimeError(
        f"M0-M3: expected 30877 rows, got {len(base)}"
    )

if base["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filename in M0-M3."
    )


# ================================================================
# LOAD M4
# ================================================================

if not M4_OOF.exists():
    raise FileNotFoundError(
        M4_OOF
    )

m4 = pd.read_csv(
    M4_OOF
)

required_m4 = {
    "filename",
    "culture",
    "fold",
    "DL",
    "prediction_m4",
}

missing = (
    required_m4
    - set(m4.columns)
)

if missing:
    raise RuntimeError(
        f"Missing M4 columns: {missing}"
    )

if len(m4) != 30877:
    raise RuntimeError(
        f"M4: expected 30877 rows, got {len(m4)}"
    )

if m4["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filename in M4."
    )


# ================================================================
# MERGE
# ================================================================

m4small = (
    m4[
        [
            "filename",
            "culture",
            "fold",
            "DL",
            "prediction_m4",
        ]
    ]
    .rename(
        columns={
            "culture":
                "culture_m4",

            "fold":
                "fold_m4",

            "DL":
                "DL_m4",
        }
    )
)

df = base.merge(
    m4small,
    on="filename",
    how="inner",
    validate="one_to_one",
)


if len(df) != 30877:
    raise RuntimeError(
        f"Incomplete merge: {len(df)} rows"
    )


if not (
    df["culture"]
    == df["culture_m4"]
).all():

    raise RuntimeError(
        "Culture mismatch M0-M3 vs M4."
    )


if not (
    df["fold"]
    == df["fold_m4"]
).all():

    raise RuntimeError(
        "Fold mismatch M0-M3 vs M4."
    )


target_diff = float(
    np.max(
        np.abs(
            df["DL"].to_numpy(
                dtype=np.float64
            )
            -
            df["DL_m4"].to_numpy(
                dtype=np.float64
            )
        )
    )
)

print(
    "Maximum M0-M3 / M4 target difference:",
    f"{target_diff:.12g}",
)

if target_diff > 1e-6:
    raise RuntimeError(
        "Target DL mismatch M4."
    )


df = df.drop(
    columns=[
        "culture_m4",
        "fold_m4",
        "DL_m4",
    ]
)


# ================================================================
# MODELS
# ================================================================

models = {
    "M0_RF":
        "pred_M0_119",

    "M1_ResNet18_DL":
        "m1_tta8",

    "M2_ResNet18_multitask":
        "m2_tta8",

    "M3_ConvNeXt_multitask":
        "m3_tta8",

    "M4_DINOv2_Ridge":
        "prediction_m4",
}


metadata = {
    "M0_RF": {
        "architecture":
            "Random Forest",
        "mode":
            "Handcrafted features",
        "auxiliary":
            "No",
        "pretraining":
            "No",
        "tta":
            "No",
    },

    "M1_ResNet18_DL": {
        "architecture":
            "ResNet18 2.5D",
        "mode":
            "End-to-end DL regression",
        "auxiliary":
            "No",
        "pretraining":
            "ImageNet",
        "tta":
            "x8",
    },

    "M2_ResNet18_multitask": {
        "architecture":
            "ResNet18 2.5D",
        "mode":
            "Multitask",
        "auxiliary":
            "OCT4/FOXA2/SOX17",
        "pretraining":
            "ImageNet",
        "tta":
            "x8",
    },

    "M3_ConvNeXt_multitask": {
        "architecture":
            "ConvNeXt-Tiny 2.5D",
        "mode":
            "Multitask",
        "auxiliary":
            "OCT4/FOXA2/SOX17",
        "pretraining":
            "ImageNet",
        "tta":
            "x8",
    },

    "M4_DINOv2_Ridge": {
        "architecture":
            "DINOv2 ViT-S + Ridge",
        "mode":
            "Frozen representation",
        "auxiliary":
            "No",
        "pretraining":
            "DINOv2",
        "tta":
            "No",
    },
}


# ================================================================
# GLOBAL NUCLEUS METRICS
# ================================================================

global_rows = []

for model, col in models.items():

    r = metrics(
        df["DL"],
        df[col],
    )

    global_rows.append({
        "model":
            model,

        **metadata[model],

        **{
            f"nucleus_{k}": v
            for k, v in r.items()
        },
    })


global_df = pd.DataFrame(
    global_rows
)


# ================================================================
# CULTURE LEVEL
# ================================================================

agg_dict = {
    "DL":
        "mean",
}

for model, col in models.items():
    agg_dict[col] = "mean"


culture = (
    df
    .groupby(
        "culture",
        sort=True,
    )
    .agg(
        agg_dict
    )
    .reset_index()
)


if len(culture) != 48:
    raise RuntimeError(
        f"Expected 48 cultures, got {len(culture)}"
    )


culture_rows = []

for model, col in models.items():

    r = metrics(
        culture["DL"],
        culture[col],
    )

    culture_rows.append({
        "model":
            model,

        **{
            f"culture_{k}": v
            for k, v in r.items()
        },
    })


culture_df = pd.DataFrame(
    culture_rows
)


# ================================================================
# PER FOLD
# ================================================================

fold_rows = []

for fold in sorted(
    df["fold"].unique()
):

    sub = df[
        df["fold"] == fold
    ]

    for model, col in models.items():

        r = metrics(
            sub["DL"],
            sub[col],
        )

        fold_rows.append({
            "fold":
                int(fold),

            "model":
                model,

            "n":
                len(sub),

            **r,
        })


fold_df = pd.DataFrame(
    fold_rows
)


fold_summary = (
    fold_df
    .groupby(
        "model"
    )
    .agg(
        fold_r2_mean=(
            "r2",
            "mean",
        ),

        fold_r2_std=(
            "r2",
            "std",
        ),

        fold_mae_mean=(
            "mae",
            "mean",
        ),

        fold_mae_std=(
            "mae",
            "std",
        ),

        fold_pearson_mean=(
            "pearson",
            "mean",
        ),

        fold_spearman_mean=(
            "spearman",
            "mean",
        ),
    )
    .reset_index()
)


# ================================================================
# FINAL MERGE
# ================================================================

final = (
    global_df
    .merge(
        culture_df,
        on="model",
        validate="one_to_one",
    )
    .merge(
        fold_summary,
        on="model",
        validate="one_to_one",
    )
)


# Keep model order
order = {
    model: i
    for i, model in enumerate(
        models.keys()
    )
}

final["_order"] = (
    final["model"]
    .map(order)
)

final = (
    final
    .sort_values("_order")
    .drop(
        columns="_order"
    )
    .reset_index(
        drop=True
    )
)


# ================================================================
# DELTAS VS M0
# ================================================================

m0 = (
    final[
        final["model"]
        == "M0_RF"
    ]
    .iloc[0]
)

final["delta_r2_vs_M0"] = (
    final["nucleus_r2"]
    - m0["nucleus_r2"]
)

final["delta_mae_vs_M0"] = (
    final["nucleus_mae"]
    - m0["nucleus_mae"]
)


# ================================================================
# SAVE
# ================================================================

FINAL_PATH = (
    OUT
    / "m0_m4_final_comparison.csv"
)

FOLD_PATH = (
    OUT
    / "m0_m4_per_fold.csv"
)

CULTURE_PATH = (
    OUT
    / "m0_m4_culture_predictions.csv"
)

final.to_csv(
    FINAL_PATH,
    index=False,
)

fold_df.to_csv(
    FOLD_PATH,
    index=False,
)

culture.to_csv(
    CULTURE_PATH,
    index=False,
)


# ================================================================
# TERMINAL TABLE
# ================================================================

compact = final[
    [
        "model",
        "nucleus_r2",
        "nucleus_mae",
        "nucleus_pearson",
        "nucleus_spearman",
        "culture_r2",
        "culture_mae",
        "culture_pearson",
        "culture_spearman",
        "fold_r2_mean",
        "fold_r2_std",
        "fold_mae_mean",
        "fold_mae_std",
    ]
]


print()
print("=" * 145)
print(
    "M0-M4 FINAL COMPARISON — "
    "30877 NUCLEI / 48 CULTURES / "
    "5-FOLD CULTURE-WISE"
)
print("=" * 145)

print(
    compact
    .round(6)
    .to_string(
        index=False
    )
)


# ================================================================
# PERFORMANCE ORDER
# ================================================================

print()
print("=" * 100)
print("NUCLEUS-LEVEL STATIC PERFORMANCE")
print("=" * 100)

print(
    final[
        [
            "model",
            "nucleus_r2",
            "nucleus_mae",
        ]
    ]
    .sort_values(
        "nucleus_r2",
        ascending=False,
    )
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("CULTURE-LEVEL STATIC PERFORMANCE")
print("=" * 100)

print(
    final[
        [
            "model",
            "culture_r2",
            "culture_mae",
        ]
    ]
    .sort_values(
        "culture_r2",
        ascending=False,
    )
    .round(6)
    .to_string(
        index=False
    )
)


# ================================================================
# MARKDOWN
# ================================================================

MD_PATH = (
    OUT
    / "m0_m4_final_comparison.md"
)

lines = []

lines.append(
    "# M0-M4 static culture-wise comparison"
)

lines.append("")

lines.append(
    "| Model | Architecture | "
    "Nucleus R2 | Nucleus MAE | "
    "Culture R2 | Culture MAE | "
    "Fold R2 mean +/- SD |"
)

lines.append(
    "|---|---|---:|---:|---:|---:|---:|"
)


for _, r in final.iterrows():

    lines.append(
        f"| {r['model']} "
        f"| {r['architecture']} "
        f"| {r['nucleus_r2']:.4f} "
        f"| {r['nucleus_mae']:.4f} "
        f"| {r['culture_r2']:.4f} "
        f"| {r['culture_mae']:.4f} "
        f"| {r['fold_r2_mean']:.4f} +/- "
        f"{r['fold_r2_std']:.4f} |"
    )


lines.append("")
lines.append(
    "Evaluation: 5-fold culture-wise OOF, "
    "30877 nuclei from 48 cultures."
)

lines.append("")
lines.append(
    "M2 and M3 use OCT4/FOXA2/SOX17 only as "
    "auxiliary training targets. Fluorescence is "
    "never an inference input."
)

lines.append("")
lines.append(
    "M4 is a frozen DINOv2 representation with "
    "a Ridge regression head and is used primarily "
    "as an independent representation control."
)


MD_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ================================================================
# SHORT SUMMARY
# ================================================================

SUMMARY_PATH = (
    OUT
    / "m0_m4_final_summary.txt"
)

summary = []

summary.append(
    "M0-M4 FINAL COMPARISON"
)

summary.append(
    "30877 nuclei / 48 cultures / "
    "5-fold culture-wise"
)

summary.append("")

summary.append(
    compact
    .round(6)
    .to_string(
        index=False
    )
)

summary.append("")
summary.append(
    "Static supervised performance only."
)

summary.append(
    "Longitudinal robustness must be interpreted "
    "separately."
)


SUMMARY_PATH.write_text(
    "\n".join(summary),
    encoding="utf-8",
)


print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(FINAL_PATH)
print(FOLD_PATH)
print(CULTURE_PATH)
print(MD_PATH)
print(SUMMARY_PATH)

print()
print("DONE")
