"""
Create the publication-style longitudinal change-from-Day00 figure.

Purpose
-------
Plot the frozen mean prediction changes for M1-M4 relative to each model's own
Day00 baseline. The plotted values are loaded from the committed release result
table rather than duplicated as constants in the figure code.

Input
-----
- ``results/longitudinal_model_delta_from_day00.csv``

Outputs
-------
Written under ``outputs/figures/longitudinal/``:
- ``longitudinal_delta_from_day00_pretty.png``
- ``longitudinal_delta_from_day00_pretty.pdf``

Interpretation
--------------
This is a descriptive sensitivity analysis over six cultures with no matched
longitudinal fluorescence ground truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# ============================================================
# OUTPUTS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/figures/longitudinal"
OUT.mkdir(parents=True, exist_ok=True)

PNG = OUT / "longitudinal_delta_from_day00_pretty.png"
PDF = OUT / "longitudinal_delta_from_day00_pretty.pdf"


# ============================================================
# FROZEN LONGITUDINAL RESULTS
# Mean predicted-score change relative to each model's Day00.
# Values are read from the committed release table so that the
# figure and the published numeric result share one source of truth.
# ============================================================

RESULTS = ROOT / "results/longitudinal_model_delta_from_day00.csv"

if not RESULTS.exists():
    raise FileNotFoundError(RESULTS)

delta_df = pd.read_csv(RESULTS)

expected_columns = {"day", "M1", "M2", "M3", "M4"}
missing = expected_columns - set(delta_df.columns)

if missing:
    raise RuntimeError(
        f"Missing longitudinal result columns: {sorted(missing)}"
    )

expected_days = ["Day00", "Day01", "Day02"]

if delta_df["day"].tolist() != expected_days:
    raise RuntimeError(
        "Expected longitudinal rows in order Day00, Day01, Day02."
    )

days = np.arange(len(expected_days))

models = {
    "M1 · ResNet18": {
        "values": delta_df["M1"].to_numpy(dtype=float),
        "color": "#6599C2",
        "width": 2.6,
    },
    "M2 · ResNet18 multitask": {
        "values": delta_df["M2"].to_numpy(dtype=float),
        "color": "#2799AA",
        "width": 2.6,
    },
    "M3 · ConvNeXt multitask": {
        "values": delta_df["M3"].to_numpy(dtype=float),
        "color": "#30458D",
        "width": 3.3,
    },
    "M4 · DINOv2 + Ridge": {
        "values": delta_df["M4"].to_numpy(dtype=float),
        "color": "#B64F87",
        "width": 2.6,
    },
}


# ============================================================
# STYLE
# ============================================================

NAVY = "#152544"
TEXT = "#35445E"
GRID = "#E6EBF3"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(11.5, 6.3), dpi=180)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

fig.suptitle(
    "Longitudinal variation across model families",
    fontsize=19,
    fontweight="bold",
    color=NAVY,
    y=0.97,
)

ax.set_title(
    "Mean predicted-score change relative to each model's Day00 baseline",
    fontsize=11.5,
    color=TEXT,
    pad=54,
)


# ============================================================
# REFERENCE LINE
# ============================================================

ax.axhline(
    y=0,
    color="#8998B0",
    linewidth=1.5,
    linestyle=(0, (5, 4)),
    zorder=1,
)


# ============================================================
# MODEL CURVES
# ============================================================

for name, info in models.items():

    values = np.array(info["values"])

    ax.plot(
        days,
        values,
        label=name,
        color=info["color"],
        linewidth=info["width"],
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
        solid_capstyle="round",
        zorder=3,
    )


# ============================================================
# ENDPOINT LABELS
# ============================================================

# Vertical adjustments to keep close labels readable.
offsets = {
    "M1 · ResNet18": 9,
    "M2 · ResNet18 multitask": -3,
    "M3 · ConvNeXt multitask": -12,
    "M4 · DINOv2 + Ridge": 0,
}

short_names = {
    "M1 · ResNet18": "M1",
    "M2 · ResNet18 multitask": "M2",
    "M3 · ConvNeXt multitask": "M3",
    "M4 · DINOv2 + Ridge": "M4",
}

for name, info in models.items():

    final_value = info["values"][-1]

    ax.annotate(
        f"{short_names[name]}  {final_value:+.3f}",
        xy=(2, final_value),
        xytext=(13, offsets[name]),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=info["color"],
    )


# ============================================================
# AXES
# ============================================================

ax.set_xlim(-0.10, 2.43)
ax.set_ylim(-0.115, 0.015)

ax.set_xticks(days)
ax.set_xticklabels(["Day00", "Day01", "Day02"])

ax.set_xlabel(
    "Acquisition day",
    color=NAVY,
    labelpad=12,
)

ax.set_ylabel(
    "Mean predicted-score change vs Day00",
    color=NAVY,
    labelpad=12,
)

ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

ax.grid(
    axis="y",
    color=GRID,
    linewidth=1,
    zorder=0,
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for side in ["left", "bottom"]:
    ax.spines[side].set_color("#AAB5C7")
    ax.spines[side].set_linewidth(1)

ax.tick_params(
    axis="both",
    colors=TEXT,
    length=0,
    pad=9,
)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.13),
    ncol=4,
    frameon=False,
    fontsize=10,
    handlelength=2.5,
    columnspacing=1.4,
)


# ============================================================
# FOOTNOTE
# ============================================================

fig.text(
    0.12,
    0.025,
    "Descriptive sensitivity analysis · Six cultures · "
    "No matched longitudinal fluorescence ground truth",
    fontsize=9,
    color="#718096",
)


# ============================================================
# SAVE
# ============================================================

fig.subplots_adjust(
    left=0.12,
    right=0.88,
    bottom=0.16,
    top=0.75,
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

print(f"PNG : {PNG}")
print(f"PDF : {PDF}")
print("DONE")
