"""
Build a compact frozen snapshot of the final Bright2Nuc evaluation results.

Purpose
-------
Combine the validated M0-M4 static comparison with M3 deep-ensemble metrics and
the selective-prediction table, then write the compact final tables and text
summary used by the release.

Inputs
------
- ``outputs/model_comparison/m0_m4_final_comparison.csv``
- ``outputs/m3_ensemble/analysis/m3_ensemble_metrics.csv``
- ``outputs/m3_ensemble/analysis/m3_abstention_curve.csv``

Outputs
-------
Written under ``outputs/final_results/``:
- ``final_model_comparison.csv``
- ``final_model_comparison.md``
- ``final_results_summary.txt``

Notes
-----
The compact text snapshot also includes validated culture-level ensemble,
uncertainty, longitudinal, and D01-D00 confounder summary values from the
frozen final analysis. Those values are intentionally kept as release snapshot
constants here rather than recomputing the longitudinal pipeline.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

STATIC = (
    ROOT
    / "outputs/model_comparison/m0_m4_final_comparison.csv"
)

ENSEMBLE = (
    ROOT
    / "outputs/m3_ensemble/analysis/m3_ensemble_metrics.csv"
)

ABSTENTION = (
    ROOT
    / "outputs/m3_ensemble/analysis/m3_abstention_curve.csv"
)

OUT = (
    ROOT
    / "outputs/final_results"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# LOAD
# ================================================================

static = pd.read_csv(STATIC)
ensemble = pd.read_csv(ENSEMBLE)
abstention = pd.read_csv(ABSTENTION)


print("=" * 100)
print("FINAL RESULTS SNAPSHOT")
print("=" * 100)

print()
print("Static columns:")
print(list(static.columns))

print()
print("Ensemble columns:")
print(list(ensemble.columns))


# ================================================================
# STATIC M0-M4
# ================================================================

static_keep = static[
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
    ]
].copy()


# ================================================================
# ENSEMBLE
# ================================================================

ens = (
    ensemble[
        ensemble["model"]
        == "ensemble_mean"
    ]
    .iloc[0]
)


ensemble_row = pd.DataFrame(
    [
        {
            "model":
                "M3_DeepEnsemble",

            "nucleus_r2":
                ens["r2"],

            "nucleus_mae":
                ens["mae"],

            "nucleus_pearson":
                ens["pearson"],

            "nucleus_spearman":
                ens["spearman"],

            "culture_r2":
                0.891968,

            "culture_mae":
                0.089277,

            "culture_pearson":
                0.955536,

            "culture_spearman":
                0.855406,
        }
    ]
)


final = pd.concat(
    [
        static_keep,
        ensemble_row,
    ],
    ignore_index=True,
)


# ================================================================
# SAVE MAIN TABLE
# ================================================================

CSV_PATH = (
    OUT
    / "final_model_comparison.csv"
)

MD_PATH = (
    OUT
    / "final_model_comparison.md"
)

TXT_PATH = (
    OUT
    / "final_results_summary.txt"
)


final.to_csv(
    CSV_PATH,
    index=False,
)


MD_PATH.write_text(
    final
    .round(6)
    .to_markdown(
        index=False
    ),
    encoding="utf-8",
)


# ================================================================
# ENSEMBLE IMPROVEMENT
# ================================================================

m3 = final[
    final["model"]
    == "M3_ConvNeXt_multitask"
].iloc[0]


delta_r2 = (
    ensemble_row.iloc[0]["nucleus_r2"]
    - m3["nucleus_r2"]
)

delta_mae = (
    ensemble_row.iloc[0]["nucleus_mae"]
    - m3["nucleus_mae"]
)


# ================================================================
# ABSTENTION
# ================================================================

abstention_main = (
    abstention[
        abstention["reject_pct"]
        .isin(
            [
                0,
                5,
                10,
                20,
            ]
        )
    ]
    [
        [
            "reject_pct",
            "kept_n",
            "mae",
            "r2",
        ]
    ]
    .copy()
)


# ================================================================
# SUMMARY
# ================================================================

lines = []

lines.append(
    "=" * 100
)

lines.append(
    "BRIGHT2NUC — FINAL RESULTS SNAPSHOT"
)

lines.append(
    "=" * 100
)

lines.append("")

lines.append(
    "STATIC / OOF MODEL COMPARISON"
)

lines.append(
    final
    .round(6)
    .to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "=" * 100
)

lines.append(
    "M3 -> DEEP ENSEMBLE"
)

lines.append(
    "=" * 100
)

lines.append(
    f"M3 nucleus R2       : {m3['nucleus_r2']:.6f}"
)

lines.append(
    f"Ensemble nucleus R2 : {ensemble_row.iloc[0]['nucleus_r2']:.6f}"
)

lines.append(
    f"Delta R2            : {delta_r2:+.6f}"
)

lines.append("")

lines.append(
    f"M3 nucleus MAE       : {m3['nucleus_mae']:.6f}"
)

lines.append(
    f"Ensemble nucleus MAE : {ensemble_row.iloc[0]['nucleus_mae']:.6f}"
)

lines.append(
    f"Delta MAE            : {delta_mae:+.6f}"
)

lines.append("")

lines.append(
    "UNCERTAINTY"
)

lines.append(
    "Spearman uncertainty vs absolute error: +0.518948"
)

lines.append("")

lines.append(
    "ABSTENTION"
)

lines.append(
    abstention_main
    .round(6)
    .to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "LONGITUDINAL — CHANGE RELATIVE TO DAY00"
)

lines.append(
    "M1 Day01=-0.049212  Day02=-0.042802"
)

lines.append(
    "M2 Day01=-0.065931  Day02=-0.061759"
)

lines.append(
    "M3 Day01=-0.102497  Day02=-0.049414"
)

lines.append(
    "M4 Day01=-0.094914  Day02=-0.024706"
)

lines.append("")

lines.append(
    "PRIMARY D01-D00 CONFOUNDER SIGNAL"
)

lines.append(
    "Contrast rho: M1=+0.0857 M2=+0.0286 M3=-0.7714 M4=-0.3714"
)

lines.append(
    "Tenengrad rho: M1=+0.0857 M2=+0.0286 M3=+0.0286 M4=+0.3714"
)

lines.append(
    "Spots/frame rho: M1=-0.5429 M2=-0.3714 M3=+0.7143 M4=+0.6000"
)


TXT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ================================================================
# DISPLAY
# ================================================================

print()
print("=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

print(
    final
    .round(6)
    .to_string(
        index=False
    )
)

print()
print("=" * 100)
print("M3 -> DEEP ENSEMBLE")
print("=" * 100)

print(
    f"Delta R2  : {delta_r2:+.6f}"
)

print(
    f"Delta MAE : {delta_mae:+.6f}"
)

print()
print("=" * 100)
print("ABSTENTION")
print("=" * 100)

print(
    abstention_main
    .round(6)
    .to_string(
        index=False
    )
)

print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(CSV_PATH)
print(MD_PATH)
print(TXT_PATH)

print()
print("DONE")
