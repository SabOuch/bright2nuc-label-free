"""
Generate the base uncertainty and selective-prediction figures for M3 ensemble.

Purpose
-------
Create the three source panels used by the publication uncertainty figure:
1. mean absolute error across uncertainty deciles;
2. selective-prediction / abstention performance;
3. per-nucleus ensemble uncertainty versus absolute prediction error.

Inputs
------
Read from ``outputs/m3_ensemble/analysis/``:
- ``m3_ensemble_oof_predictions.csv``
- ``m3_uncertainty_deciles.csv``
- ``m3_abstention_curve.csv``

Outputs
-------
Written under ``outputs/figures/`` as both PNG and PDF:
- ``m3_uncertainty_deciles_mae``
- ``m3_abstention_curve``
- ``m3_uncertainty_vs_error``

Reproducibility
---------------
The scatter panel uses a deterministic 12,000-nucleus visual subsample with
NumPy seed 42. Correlation is computed on the complete OOF set, not on the
visual subsample.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS = (
    ROOT
    / "outputs/m3_ensemble/analysis"
)

FIGDIR = (
    ROOT
    / "outputs/figures"
)

FIGDIR.mkdir(
    parents=True,
    exist_ok=True,
)


OOF_PATH = (
    ANALYSIS
    / "m3_ensemble_oof_predictions.csv"
)

DECILES_PATH = (
    ANALYSIS
    / "m3_uncertainty_deciles.csv"
)

ABSTENTION_PATH = (
    ANALYSIS
    / "m3_abstention_curve.csv"
)


for p in [
    OOF_PATH,
    DECILES_PATH,
    ABSTENTION_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


oof = pd.read_csv(
    OOF_PATH
)

deciles = pd.read_csv(
    DECILES_PATH
)

abstention = pd.read_csv(
    ABSTENTION_PATH
)


# ================================================================
# STYLE
# ================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})


def save_figure(fig, stem):
    """Save one Matplotlib figure as publication PNG and PDF."""

    png = FIGDIR / f"{stem}.png"
    pdf = FIGDIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    print("Saved:", png)
    print("Saved:", pdf)


# ================================================================
# FIGURE 1 — ERROR BY UNCERTAINTY DECILE
# ================================================================

fig, ax = plt.subplots(
    figsize=(7.4, 4.8)
)

x = (
    deciles["uncertainty_decile"]
    .to_numpy()
)

mae = (
    deciles["mae"]
    .to_numpy()
)

ax.plot(
    x,
    mae,
    marker="o",
    linewidth=2,
)

ax.set_xlabel(
    "Uncertainty decile"
)

ax.set_ylabel(
    "Mean absolute error"
)

ax.set_title(
    "Prediction error increases with model uncertainty"
)

ax.set_xticks(
    x
)

ax.grid(
    alpha=0.25
)


for xi, yi in zip(
    x,
    mae,
):

    ax.annotate(
        f"{yi:.3f}",
        (xi, yi),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )


save_figure(
    fig,
    "m3_uncertainty_deciles_mae",
)

plt.close(fig)


# ================================================================
# FIGURE 2 — ABSTENTION CURVE
# ================================================================

fig, ax1 = plt.subplots(
    figsize=(7.4, 4.8)
)


reject = (
    abstention["reject_pct"]
    .to_numpy()
)

mae = (
    abstention["mae"]
    .to_numpy()
)

r2 = (
    abstention["r2"]
    .to_numpy()
)


line1 = ax1.plot(
    reject,
    mae,
    marker="o",
    linewidth=2,
    label="MAE",
)

ax1.set_xlabel(
    "Most uncertain nuclei rejected (%)"
)

ax1.set_ylabel(
    "MAE"
)

ax1.grid(
    alpha=0.25
)


ax2 = ax1.twinx()

line2 = ax2.plot(
    reject,
    r2,
    marker="s",
    linestyle="--",
    linewidth=2,
    label="R²",
)

ax2.set_ylabel(
    "R²"
)


lines = (
    line1
    + line2
)

labels = [
    line.get_label()
    for line in lines
]

ax1.legend(
    lines,
    labels,
    loc="center right",
)


ax1.set_title(
    "Selective prediction improves performance"
)


# Highlight 20% point
idx20 = np.flatnonzero(
    reject == 20
)

if len(idx20) == 1:

    i = idx20[0]

    ax1.annotate(
        (
            f"20% rejected\n"
            f"MAE={mae[i]:.3f}\n"
            f"R²={r2[i]:.3f}"
        ),
        xy=(
            reject[i],
            mae[i],
        ),
        xytext=(
            reject[i] + 7,
            mae[i] + 0.012,
        ),
        arrowprops={
            "arrowstyle": "->",
        },
        fontsize=9,
    )


save_figure(
    fig,
    "m3_abstention_curve",
)

plt.close(fig)


# ================================================================
# FIGURE 3 — UNCERTAINTY VS ERROR
# ================================================================

unc = (
    oof["ensemble_std"]
    .to_numpy(
        dtype=np.float64
    )
)

err = (
    oof["abs_error"]
    .to_numpy(
        dtype=np.float64
    )
)


rho = spearmanr(
    unc,
    err,
).statistic


# Bin by uncertainty quantiles for readable trend
n_bins = 20

bin_id = pd.qcut(
    unc,
    q=n_bins,
    labels=False,
    duplicates="drop",
)

tmp = pd.DataFrame({
    "uncertainty":
        unc,

    "error":
        err,

    "bin":
        bin_id,
})


trend = (
    tmp
    .groupby(
        "bin",
        observed=True,
    )
    .agg(
        uncertainty_median=(
            "uncertainty",
            "median",
        ),

        error_median=(
            "error",
            "median",
        ),

        error_mean=(
            "error",
            "mean",
        ),
    )
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(7.4, 5.2)
)


# Background point cloud
rng = np.random.default_rng(
    42
)

sample_n = min(
    12000,
    len(oof),
)

sample_idx = rng.choice(
    len(oof),
    size=sample_n,
    replace=False,
)


ax.scatter(
    unc[sample_idx],
    err[sample_idx],
    s=6,
    alpha=0.12,
    rasterized=True,
    label="Nuclei",
)


ax.plot(
    trend["uncertainty_median"],
    trend["error_mean"],
    marker="o",
    linewidth=2.2,
    label="Mean error by uncertainty bin",
)


ax.set_xlabel(
    "Deep ensemble uncertainty (SD)"
)

ax.set_ylabel(
    "Absolute prediction error"
)

ax.set_title(
    (
        "Uncertainty is associated with prediction error\n"
        f"Spearman ρ = {rho:.3f}"
    )
)

ax.grid(
    alpha=0.25
)

ax.legend()


save_figure(
    fig,
    "m3_uncertainty_vs_error",
)

plt.close(fig)


# ================================================================
# CHECKS / SUMMARY
# ================================================================

print()
print("=" * 88)
print("FIGURE SUMMARY")
print("=" * 88)

print(
    f"N nuclei                  : {len(oof)}"
)

print(
    f"Spearman uncertainty/error: {rho:+.6f}"
)

print()

for reject_pct in [
    0,
    5,
    10,
    20,
]:

    row = abstention[
        abstention["reject_pct"]
        == reject_pct
    ]

    if len(row) == 1:

        r = row.iloc[0]

        print(
            f"Reject {reject_pct:2d}% | "
            f"MAE={r['mae']:.6f} | "
            f"R2={r['r2']:.6f}"
        )


print()
print("=" * 88)
print("FILES CREATED")
print("=" * 88)

for p in sorted(
    FIGDIR.glob(
        "m3_*.png"
    )
):
    print(p)

for p in sorted(
    FIGDIR.glob(
        "m3_*.pdf"
    )
):
    print(p)

print()
print("DONE")
