"""
Create the publication-style final static model-comparison figure.

Purpose
-------
Plot nucleus-level R² and MAE for the frozen M0-M4 comparison and the M3 deep
ensemble, while distinguishing the best single-model family from the final
ensemble system.

Input
-----
- ``outputs/final_results/final_model_comparison.csv``

Outputs
-------
Written under ``outputs/figures/final/``:
- ``final_model_comparison_publication.png``
- ``final_model_comparison_publication.pdf``

Notes
-----
The ensemble gain annotations are computed from the input table at runtime;
performance values are not hard-coded in this figure script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

SRC = (
    ROOT
    / "outputs/final_results"
    / "final_model_comparison.csv"
)

OUT = (
    ROOT
    / "outputs/figures/final"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# LOAD
# ================================================================

df = pd.read_csv(SRC)

order = [
    "M0_RF",
    "M1_ResNet18_DL",
    "M2_ResNet18_multitask",
    "M3_ConvNeXt_multitask",
    "M4_DINOv2_Ridge",
    "M3_DeepEnsemble",
]

labels = {
    "M0_RF":
        "M0\nRF",

    "M1_ResNet18_DL":
        "M1\nResNet18",

    "M2_ResNet18_multitask":
        "M2\nResNet18\nmultitask",

    "M3_ConvNeXt_multitask":
        "M3\nConvNeXt\nmultitask",

    "M4_DINOv2_Ridge":
        "M4\nDINOv2\n+ Ridge",

    "M3_DeepEnsemble":
        "M3\nDeep\nEnsemble",
}

df = (
    df
    .set_index("model")
    .loc[order]
    .reset_index()
)

x = np.arange(len(df))

r2 = df["nucleus_r2"].to_numpy(float)
mae = df["nucleus_mae"].to_numpy(float)

xlabels = [
    labels[m]
    for m in df["model"]
]


# ================================================================
# DELTAS M3 -> ENSEMBLE
# ================================================================

m3_idx = order.index(
    "M3_ConvNeXt_multitask"
)

ens_idx = order.index(
    "M3_DeepEnsemble"
)

delta_r2 = (
    r2[ens_idx]
    - r2[m3_idx]
)

delta_mae = (
    mae[ens_idx]
    - mae[m3_idx]
)


# ================================================================
# STYLE
# ================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
})


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.8, 5.8),
)


def style_bars(bars):
    """Apply consistent emphasis to M3 and the final deep ensemble."""

    for i, bar in enumerate(bars):

        if i == m3_idx:
            bar.set_alpha(0.85)
            bar.set_hatch("//")
            bar.set_linewidth(1.5)

        elif i == ens_idx:
            bar.set_alpha(1.0)
            bar.set_hatch("xx")
            bar.set_linewidth(2.2)

        else:
            bar.set_alpha(0.60)
            bar.set_linewidth(0.8)


# ================================================================
# PANEL A — R2
# ================================================================

ax = axes[0]

bars = ax.bar(
    x,
    r2,
    edgecolor="black",
)

style_bars(bars)

ax.set_xticks(
    x,
    xlabels,
)

ax.set_ylabel(
    "Nucleus-level R²"
)

ax.set_title(
    "A. Nucleus-level predictive performance"
)

ax.set_ylim(
    0,
    0.84,
)

ax.grid(
    axis="y",
    alpha=0.20,
)

ax.set_axisbelow(True)


for i, (bar, value) in enumerate(
    zip(bars, r2)
):

    weight = (
        "bold"
        if i in [m3_idx, ens_idx]
        else "normal"
    )

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.014,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight=weight,
    )


ax.annotate(
    "Best single model",
    xy=(
        m3_idx,
        r2[m3_idx],
    ),
    xytext=(
        m3_idx - 0.55,
        0.785,
    ),
    ha="center",
    fontsize=9,
    arrowprops={
        "arrowstyle": "->",
        "lw": 1,
    },
)


ax.annotate(
    "Final system",
    xy=(
        ens_idx,
        r2[ens_idx],
    ),
    xytext=(
        ens_idx - 0.15,
        0.815,
    ),
    ha="center",
    fontsize=9,
    fontweight="bold",
    arrowprops={
        "arrowstyle": "->",
        "lw": 1.2,
    },
)


ax.text(
    0.03,
    0.94,
    f"Ensemble gain: ΔR² = {delta_r2:+.3f}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox={
        "boxstyle": "round,pad=0.3",
        "facecolor": "white",
        "alpha": 0.85,
        "edgecolor": "0.7",
    },
)


# ================================================================
# PANEL B — MAE
# ================================================================

ax = axes[1]

bars = ax.bar(
    x,
    mae,
    edgecolor="black",
)

style_bars(bars)

ax.set_xticks(
    x,
    xlabels,
)

ax.set_ylabel(
    "Nucleus-level MAE"
)

ax.set_title(
    "B. Nucleus-level prediction error"
)

ax.set_ylim(
    0,
    0.275,
)

ax.grid(
    axis="y",
    alpha=0.20,
)

ax.set_axisbelow(True)


for i, (bar, value) in enumerate(
    zip(bars, mae)
):

    weight = (
        "bold"
        if i in [m3_idx, ens_idx]
        else "normal"
    )

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.006,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight=weight,
    )


ax.annotate(
    "Best single model",
    xy=(
        m3_idx,
        mae[m3_idx],
    ),
    xytext=(
        m3_idx - 0.55,
        0.158,
    ),
    ha="center",
    fontsize=9,
    arrowprops={
        "arrowstyle": "->",
        "lw": 1,
    },
)


ax.annotate(
    "Final system",
    xy=(
        ens_idx,
        mae[ens_idx],
    ),
    xytext=(
        ens_idx - 0.10,
        0.155,
    ),
    ha="center",
    fontsize=9,
    fontweight="bold",
    arrowprops={
        "arrowstyle": "->",
        "lw": 1.2,
    },
)


ax.text(
    0.03,
    0.94,
    f"Ensemble gain: ΔMAE = {delta_mae:+.3f}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox={
        "boxstyle": "round,pad=0.3",
        "facecolor": "white",
        "alpha": 0.85,
        "edgecolor": "0.7",
    },
)


# ================================================================
# EMPHASIZE X LABELS
# ================================================================

for ax in axes:

    ticks = ax.get_xticklabels()

    ticks[m3_idx].set_fontweight(
        "bold"
    )

    ticks[ens_idx].set_fontweight(
        "bold"
    )


# ================================================================
# TITLE
# ================================================================

fig.suptitle(
    "Brightfield-only differentiation prediction",
    fontsize=15,
    fontweight="bold",
)

fig.text(
    0.5,
    0.925,
    "Culture-wise out-of-fold evaluation",
    ha="center",
    fontsize=10,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.90,
    ]
)


# ================================================================
# SAVE
# ================================================================

PNG = (
    OUT
    / "final_model_comparison_publication.png"
)

PDF = (
    OUT
    / "final_model_comparison_publication.pdf"
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ================================================================
# CHECK
# ================================================================

print("=" * 90)
print("FINAL PUBLICATION FIGURE")
print("=" * 90)

print()

print(
    df[
        [
            "model",
            "nucleus_r2",
            "nucleus_mae",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)

print()

print(
    f"M3 -> Ensemble ΔR²  = {delta_r2:+.6f}"
)

print(
    f"M3 -> Ensemble ΔMAE = {delta_mae:+.6f}"
)

print()

print("Saved:", PNG)
print("Saved:", PDF)

print()
print("DONE")
