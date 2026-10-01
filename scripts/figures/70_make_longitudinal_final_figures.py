"""
Generate the longitudinal sensitivity and acquisition-confounder figures.

Purpose
-------
Visualize model predictions over Day00-Day02 for M1-M4, show individual
trajectories for the six longitudinal cultures, and summarize descriptive
Spearman associations between acquisition changes and prediction changes.

Inputs
------
Read from ``outputs/longitudinal/``:
- ``m4_longitudinal_culture_day.csv``
- ``longitudinal_confounder_correlations.csv``

Outputs
-------
Written under ``outputs/figures/longitudinal/``:
- mean M1-M4 trajectories;
- one six-culture trajectory figure per model;
- D01-D00, D02-D01, and D02-D00 confounder heatmaps.

The script also writes:
- ``outputs/longitudinal/longitudinal_model_mean_trajectories.csv``

Interpretation
--------------
This is a descriptive sensitivity analysis over six cultures. The longitudinal
data do not provide matched fluorescence ground truth, so prediction changes
must not be interpreted as validated biological trajectories.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

LONG = ROOT / "outputs/longitudinal"

FIGDIR = (
    ROOT
    / "outputs/figures/longitudinal"
)

FIGDIR.mkdir(
    parents=True,
    exist_ok=True,
)


CULTURE_DAY_PATH = (
    LONG
    / "m4_longitudinal_culture_day.csv"
)

CORR_PATH = (
    LONG
    / "longitudinal_confounder_correlations.csv"
)


if not CULTURE_DAY_PATH.exists():
    raise FileNotFoundError(
        CULTURE_DAY_PATH
    )

if not CORR_PATH.exists():
    raise FileNotFoundError(
        CORR_PATH
    )


df = pd.read_csv(
    CULTURE_DAY_PATH
)

corr = pd.read_csv(
    CORR_PATH
)


# ================================================================
# CHECKS
# ================================================================

required_df = {
    "culture",
    "day",
    "m1",
    "m2",
    "m3",
    "m4",
}

missing = (
    required_df
    - set(df.columns)
)

if missing:
    raise RuntimeError(
        f"Missing longitudinal columns: {missing}"
    )


required_corr = {
    "delta",
    "confounder",
    "model",
    "rho",
}

missing = (
    required_corr
    - set(corr.columns)
)

if missing:
    raise RuntimeError(
        f"Missing correlation columns: {missing}"
    )


if len(df) != 18:
    raise RuntimeError(
        f"Expected 18 culture-day pairs, got {len(df)}"
    )


if df["culture"].nunique() != 6:
    raise RuntimeError(
        f"Expected 6 cultures, got {df['culture'].nunique()}"
    )


DAYS = [
    "Day00",
    "Day01",
    "Day02",
]

MODELS = {
    "M1": "m1",
    "M2": "m2",
    "M3": "m3",
    "M4": "m4",
}


# ================================================================
# STYLE
# ================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})


def save_figure(fig, stem):
    """Save one longitudinal figure as PNG and PDF."""

    png = (
        FIGDIR
        / f"{stem}.png"
    )

    pdf = (
        FIGDIR
        / f"{stem}.pdf"
    )

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
# ORDER DATA
# ================================================================

day_map = {
    "Day00": 0,
    "Day01": 1,
    "Day02": 2,
}

df["day_number"] = (
    df["day"]
    .map(day_map)
)

df = (
    df
    .sort_values(
        [
            "culture",
            "day_number",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ================================================================
# FIGURE 1 — MEAN TRAJECTORIES M1-M4
# ================================================================

means = (
    df
    .groupby(
        "day",
        sort=False,
    )[
        list(
            MODELS.values()
        )
    ]
    .mean()
    .reindex(
        DAYS
    )
)


fig, ax = plt.subplots(
    figsize=(7.6, 5.2)
)


for label, col in MODELS.items():

    ax.plot(
        np.arange(3),
        means[col].to_numpy(),
        marker="o",
        linewidth=2.2,
        label=label,
    )


ax.set_xticks(
    np.arange(3),
    DAYS,
)

ax.set_xlabel(
    "Acquisition day"
)

ax.set_ylabel(
    "Mean predicted differentiation level"
)

ax.set_title(
    "Longitudinal predictions depend on model representation"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    title="Model",
)

save_figure(
    fig,
    "longitudinal_mean_trajectories_m1_m4",
)

plt.close(fig)


# ================================================================
# FIGURES 2-5 — EACH MODEL, 6 CULTURES + MEAN
# ================================================================

for label, col in MODELS.items():

    fig, ax = plt.subplots(
        figsize=(7.6, 5.2)
    )

    for culture in sorted(
        df["culture"].unique()
    ):

        sub = (
            df[
                df["culture"]
                == culture
            ]
            .set_index(
                "day"
            )
            .reindex(
                DAYS
            )
        )

        ax.plot(
            np.arange(3),
            sub[col].to_numpy(),
            marker="o",
            linewidth=1.2,
            alpha=0.45,
            label=culture,
        )


    mean_values = (
        means[col]
        .to_numpy()
    )

    ax.plot(
        np.arange(3),
        mean_values,
        marker="o",
        linewidth=3.2,
        label="Mean",
    )


    ax.set_xticks(
        np.arange(3),
        DAYS,
    )

    ax.set_xlabel(
        "Acquisition day"
    )

    ax.set_ylabel(
        "Predicted differentiation level"
    )

    ax.set_title(
        f"{label} — longitudinal trajectories across 6 cultures"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        ncol=2,
        loc="best",
    )


    save_figure(
        fig,
        f"longitudinal_{label.lower()}_six_cultures",
    )

    plt.close(fig)


# ================================================================
# HEATMAP FUNCTION
# ================================================================

PRIMARY_CONFOUNDERS = [
    "contrast_std_median",
    "tenengrad_norm_median",
    "spots_per_frame_median",
]

DISPLAY_NAMES = {
    "contrast_std_median":
        "Contrast",

    "tenengrad_norm_median":
        "Tenengrad / focus",

    "spots_per_frame_median":
        "Spots per frame",
}


def make_heatmap(
    delta_name,
    stem,
):
    """Render one model-by-confounder Spearman heatmap for a day contrast."""

    sub = (
        corr[
            (
                corr["delta"]
                == delta_name
            )
            &
            (
                corr["confounder"]
                .isin(
                    PRIMARY_CONFOUNDERS
                )
            )
        ]
        .copy()
    )


    matrix = (
        sub
        .pivot(
            index="confounder",
            columns="model",
            values="rho",
        )
        .reindex(
            PRIMARY_CONFOUNDERS
        )
        [
            [
                "M1",
                "M2",
                "M3",
                "M4",
            ]
        ]
    )


    values = (
        matrix
        .to_numpy(
            dtype=float
        )
    )


    fig, ax = plt.subplots(
        figsize=(7.0, 4.4)
    )


    im = ax.imshow(
        values,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )


    ax.set_xticks(
        np.arange(
            len(matrix.columns)
        ),
        matrix.columns,
    )

    ax.set_yticks(
        np.arange(
            len(matrix.index)
        ),
        [
            DISPLAY_NAMES[x]
            for x in matrix.index
        ],
    )


    for i in range(
        values.shape[0]
    ):

        for j in range(
            values.shape[1]
        ):

            ax.text(
                j,
                i,
                f"{values[i, j]:+.2f}",
                ha="center",
                va="center",
            )


    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Acquisition characteristic"
    )

    ax.set_title(
        (
            "Association between acquisition changes "
            "and prediction changes\n"
            f"{delta_name} — Spearman ρ, n=6 cultures"
        )
    )


    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Spearman ρ"
    )


    save_figure(
        fig,
        stem,
    )

    plt.close(fig)


# ================================================================
# FIGURES 6-8 — CONFOUNDER HEATMAPS
# ================================================================

make_heatmap(
    "D01-D00",
    "longitudinal_confounders_D01_D00",
)

make_heatmap(
    "D02-D01",
    "longitudinal_confounders_D02_D01",
)

make_heatmap(
    "D02-D00",
    "longitudinal_confounders_D02_D00",
)


# ================================================================
# REPORT TABLE
# ================================================================

mean_table = pd.DataFrame({
    "day":
        DAYS,
})


for label, col in MODELS.items():

    mean_table[label] = (
        means[col]
        .to_numpy()
    )


MEAN_PATH = (
    LONG
    / "longitudinal_model_mean_trajectories.csv"
)

mean_table.to_csv(
    MEAN_PATH,
    index=False,
)


# ================================================================
# SUMMARY
# ================================================================

print()
print("=" * 100)
print("MEAN LONGITUDINAL TRAJECTORIES")
print("=" * 100)

print(
    mean_table
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("PRIMARY CONFOUNDER CONTROL — D01-D00")
print("=" * 100)

primary = (
    corr[
        (
            corr["delta"]
            == "D01-D00"
        )
        &
        (
            corr["confounder"]
            .isin(
                PRIMARY_CONFOUNDERS
            )
        )
    ]
    .pivot(
        index="confounder",
        columns="model",
        values="rho",
    )
    .reindex(
        PRIMARY_CONFOUNDERS
    )
    [
        [
            "M1",
            "M2",
            "M3",
            "M4",
        ]
    ]
)

print(
    primary
    .round(4)
    .to_string()
)


print()
print("=" * 100)
print("FILES CREATED")
print("=" * 100)

for p in sorted(
    FIGDIR.glob("*.png")
):
    print(p)

for p in sorted(
    FIGDIR.glob("*.pdf")
):
    print(p)

print()
print(MEAN_PATH)

print()
print("DONE")
