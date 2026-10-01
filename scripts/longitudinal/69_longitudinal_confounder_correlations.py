"""
Relate longitudinal prediction changes to acquisition confounder changes.

Purpose
-------
Merge acquisition-quality/density summaries with M1-M4 culture × day
predictions, compute Day01-Day00, Day02-Day01, and Day02-Day00 changes, and
estimate Spearman associations between each acquisition change and each model's
prediction change.

Inputs
------
Read from ``outputs/longitudinal/``:
- ``longitudinal_image_quality_density.csv``
- ``m4_longitudinal_culture_day.csv``

Outputs
-------
Written under ``outputs/longitudinal/``:
- ``longitudinal_confounder_deltas.csv``
- ``longitudinal_confounder_correlations.csv``
- ``longitudinal_confounder_summary.txt``

Interpretation
--------------
Culture is the analysis unit. Each correlation is based on six cultures and is
descriptive only; the reported p-values are diagnostic rather than
confirmatory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

LONG = ROOT / "outputs/longitudinal"

LONG.mkdir(parents=True, exist_ok=True)

QUALITY_PATH = (
    LONG
    / "longitudinal_image_quality_density.csv"
)

MODEL_PATH = (
    LONG
    / "m4_longitudinal_culture_day.csv"
)

OUT_DELTA = (
    LONG
    / "longitudinal_confounder_deltas.csv"
)

OUT_CORR = (
    LONG
    / "longitudinal_confounder_correlations.csv"
)

OUT_SUMMARY = (
    LONG
    / "longitudinal_confounder_summary.txt"
)


# ================================================================
# LOAD
# ================================================================

quality = pd.read_csv(
    QUALITY_PATH
)

models = pd.read_csv(
    MODEL_PATH
)


print("=" * 110)
print("LONGITUDINAL CONFOUNDER CONTROL — M1/M2/M3/M4")
print("=" * 110)

print()
print("Quality shape :", quality.shape)
print("Model shape   :", models.shape)

print()
print("Quality columns:")
print(list(quality.columns))

print()
print("Model columns:")
print(list(models.columns))


# ================================================================
# REQUIRED COLUMNS
# ================================================================

confounders = [
    "contrast_std_median",
    "tenengrad_norm_median",
    "spots_per_frame_median",
    "n_spots",
]

model_cols = [
    "m1",
    "m2",
    "m3",
    "m4",
]


required_quality = {
    "culture",
    "day",
    *confounders,
}

required_models = {
    "culture",
    "day",
    *model_cols,
}


missing_q = (
    required_quality
    - set(quality.columns)
)

missing_m = (
    required_models
    - set(models.columns)
)


if missing_q:
    raise RuntimeError(
        f"Missing quality columns: {missing_q}"
    )

if missing_m:
    raise RuntimeError(
        f"Missing model columns: {missing_m}"
    )


# ================================================================
# CLEAN KEYS
# ================================================================

for df in [quality, models]:

    df["culture"] = (
        df["culture"]
        .astype(str)
        .str.strip()
    )

    df["day"] = (
        df["day"]
        .astype(str)
        .str.strip()
    )


if quality[
    ["culture", "day"]
].duplicated().any():

    raise RuntimeError(
        "Duplicate culture/day pairs in quality."
    )


if models[
    ["culture", "day"]
].duplicated().any():

    raise RuntimeError(
        "Duplicate culture/day pairs in models."
    )


# ================================================================
# MERGE
# ================================================================

qsmall = quality[
    [
        "culture",
        "day",
        *confounders,
    ]
].copy()


msmall = models[
    [
        "culture",
        "day",
        *model_cols,
    ]
].copy()


df = qsmall.merge(
    msmall,
    on=[
        "culture",
        "day",
    ],
    how="inner",
    validate="one_to_one",
)


print()
print("Merged shape :", df.shape)
print(
    "Cultures     :",
    df["culture"].nunique(),
)

print(
    "Days         :",
    sorted(df["day"].unique()),
)


if len(df) != 18:
    raise RuntimeError(
        f"Expected 18 rows, got {len(df)}"
    )


if df["culture"].nunique() != 6:
    raise RuntimeError(
        "Expected 6 cultures."
    )


expected_days = {
    "Day00",
    "Day01",
    "Day02",
}

if set(df["day"]) != expected_days:
    raise RuntimeError(
        f"Unexpected days: {sorted(df['day'].unique())}"
    )


# ================================================================
# PIVOT
# ================================================================

variables = (
    confounders
    + model_cols
)


wide = {}


for var in variables:

    p = df.pivot(
        index="culture",
        columns="day",
        values=var,
    )

    p = p[
        [
            "Day00",
            "Day01",
            "Day02",
        ]
    ]

    wide[var] = p


cultures = list(
    wide["m1"].index
)


print()
print("Cultures :", cultures)


# ================================================================
# DELTAS
# ================================================================

delta_defs = {
    "D01-D00":
        ("Day01", "Day00"),

    "D02-D01":
        ("Day02", "Day01"),

    "D02-D00":
        ("Day02", "Day00"),
}


delta_rows = []


for culture in cultures:

    row = {
        "culture":
            culture,
    }

    for delta_name, (
        day_b,
        day_a,
    ) in delta_defs.items():

        for var in variables:

            value = (
                wide[var]
                .loc[culture, day_b]
                -
                wide[var]
                .loc[culture, day_a]
            )

            row[
                f"{var}__{delta_name}"
            ] = float(value)

    delta_rows.append(
        row
    )


delta_df = pd.DataFrame(
    delta_rows
)


delta_df.to_csv(
    OUT_DELTA,
    index=False,
)


# ================================================================
# SPEARMAN CONFOUNDER ↔ MODEL DELTA
# ================================================================

corr_rows = []


for delta_name in delta_defs:

    for conf in confounders:

        x = (
            delta_df[
                f"{conf}__{delta_name}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        for model in model_cols:

            y = (
                delta_df[
                    f"{model}__{delta_name}"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            result = spearmanr(
                x,
                y,
            )

            corr_rows.append({
                "delta":
                    delta_name,

                "confounder":
                    conf,

                "model":
                    model.upper(),

                "n":
                    len(x),

                "rho":
                    float(
                        result.statistic
                    ),

                "p_descriptive":
                    float(
                        result.pvalue
                    ),
            })


corr_df = pd.DataFrame(
    corr_rows
)


corr_df["abs_rho"] = (
    corr_df["rho"]
    .abs()
)


corr_df.to_csv(
    OUT_CORR,
    index=False,
)


# ================================================================
# DISPLAY MATRICES
# ================================================================

for delta_name in delta_defs:

    print()
    print("=" * 110)
    print(delta_name)
    print("=" * 110)

    sub = corr_df[
        corr_df["delta"]
        == delta_name
    ]

    matrix = (
        sub
        .pivot(
            index="confounder",
            columns="model",
            values="rho",
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

    print()
    print("SPEARMAN RHO")
    print(
        matrix
        .round(4)
        .to_string()
    )


# ================================================================
# FULL TABLE
# ================================================================

print()
print("=" * 110)
print("TABLE COMPLETE")
print("=" * 110)

print(
    corr_df[
        [
            "delta",
            "confounder",
            "model",
            "n",
            "rho",
            "p_descriptive",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ================================================================
# LARGEST ASSOCIATIONS — DESCRIPTIVE ONLY
# ================================================================

print()
print("=" * 110)
print("LARGEST |RHO| VALUES — DESCRIPTIVE ONLY")
print("=" * 110)

top = (
    corr_df
    .sort_values(
        "abs_rho",
        ascending=False,
    )
    .head(16)
)


print(
    top[
        [
            "delta",
            "confounder",
            "model",
            "rho",
            "p_descriptive",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ================================================================
# MODEL-SPECIFIC COMPARISON
# ================================================================

print()
print("=" * 110)
print("D01-D00 — DIRECT MODEL COMPARISON")
print("=" * 110)

d01 = corr_df[
    corr_df["delta"]
    == "D01-D00"
]


for conf in confounders:

    sub = (
        d01[
            d01["confounder"]
            == conf
        ]
        .set_index(
            "model"
        )
    )

    print()
    print(conf)

    for model in [
        "M1",
        "M2",
        "M3",
        "M4",
    ]:

        print(
            f"  {model}: "
            f"rho={sub.loc[model, 'rho']:+.4f} "
            f"p_desc={sub.loc[model, 'p_descriptive']:.4f}"
        )


# ================================================================
# SUMMARY TXT
# ================================================================

lines = []

lines.append(
    "LONGITUDINAL CONFOUNDER CONTROL"
)

lines.append(
    "6 cultures x 3 days"
)

lines.append(
    "Spearman correlations are descriptive "
    "because n=6 cultures per delta."
)

lines.append("")

lines.append(
    "Primary acquisition-quality variables:"
)

lines.append(
    "- contrast_std_median"
)

lines.append(
    "- tenengrad_norm_median"
)

lines.append(
    "- spots_per_frame_median"
)

lines.append(
    "- n_spots (secondary; acquisition-length dependent)"
)

lines.append("")


for delta_name in delta_defs:

    lines.append(
        "=" * 90
    )

    lines.append(
        delta_name
    )

    lines.append(
        "=" * 90
    )

    sub = corr_df[
        corr_df["delta"]
        == delta_name
    ]

    matrix = (
        sub
        .pivot(
            index="confounder",
            columns="model",
            values="rho",
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

    lines.append(
        matrix
        .round(4)
        .to_string()
    )

    lines.append("")


OUT_SUMMARY.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


print()
print("=" * 110)
print("FILES")
print("=" * 110)

print(OUT_DELTA)
print(OUT_CORR)
print(OUT_SUMMARY)

print()
print("DONE")
