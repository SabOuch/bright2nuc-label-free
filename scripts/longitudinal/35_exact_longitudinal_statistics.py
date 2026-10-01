"""
Compute exact small-sample longitudinal statistics for M3 and latent PC1.

Purpose
-------
Use the six-culture common-support table to perform an exact Friedman-style
permutation test over Day00-Day02 and exact pairwise analyses for M3 and PC1.

Input
-----
- ``outputs/longitudinal/common_support_culture_day.csv``

Output
------
- ``outputs/longitudinal/final_longitudinal_statistics.csv``

Statistical notes
-----------------
With six cultures and three days, the global permutation space contains
``(3!)^6 = 46,656`` configurations. Pairwise comparisons report exact Wilcoxon
and exhaustive sign-flip p-values, with Holm adjustment across the three day
contrasts within each endpoint.

Interpretation
--------------
These analyses are descriptive sensitivity tests because no matched
longitudinal fluorescence ground truth is available.
"""

from pathlib import Path
from itertools import product, permutations

import numpy as np
import pandas as pd

from scipy.stats import (
    rankdata,
    wilcoxon,
)


ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "outputs"
    / "longitudinal"
)

OUT.mkdir(parents=True, exist_ok=True)

INPUT = (
    OUT
    / "common_support_culture_day.csv"
)

DAYS = [
    "Day00",
    "Day01",
    "Day02",
]


df = pd.read_csv(
    INPUT
)

cultures = sorted(
    df["culture"].unique()
)


# ============================================================
# TABLE OF n CULTURES × 3 DAYS
# ============================================================

def make_matrix(column):
    """Return the culture × day matrix for one longitudinal endpoint."""

    pivot = (
        df.pivot(
            index="culture",
            columns="day",
            values=column,
        )
        .loc[
            cultures,
            DAYS,
        ]
    )

    return pivot.to_numpy(
        dtype=np.float64
    )


# ============================================================
# FRIEDMAN STATISTIC
# ============================================================

def friedman_q(x):
    """Compute the Friedman rank statistic for a culture × day matrix."""

    n, k = x.shape

    ranks = np.stack(
        [
            rankdata(
                row,
                method="average",
            )
            for row in x
        ],
        axis=0,
    )

    rank_sums = ranks.sum(
        axis=0
    )

    q = (
        12.0
        / (
            n
            * k
            * (k + 1)
        )
        * np.sum(
            rank_sums ** 2
        )
        -
        3.0
        * n
        * (k + 1)
    )

    return float(q)


# ============================================================
# EXACT PERMUTATION TEST
#
# 3! permutations per culture.
# With 6 cultures:
#
# Only 6^6 = 46,656 configurations.
# ============================================================

def exact_friedman_p(x):
    """Enumerate all within-culture day permutations for an exact p-value."""

    n, k = x.shape

    observed = friedman_q(
        x
    )

    ranks = np.stack(
        [
            rankdata(
                row,
                method="average",
            )
            for row in x
        ],
        axis=0,
    )

    perms = list(
        permutations(
            range(k)
        )
    )

    n_extreme = 0
    n_total = 0


    for choices in product(
        perms,
        repeat=n,
    ):

        rank_sums = np.zeros(
            k,
            dtype=np.float64,
        )


        for i, perm in enumerate(
            choices
        ):

            rank_sums += ranks[
                i,
                list(perm),
            ]


        q = (
            12.0
            / (
                n
                * k
                * (k + 1)
            )
            * np.sum(
                rank_sums ** 2
            )
            -
            3.0
            * n
            * (k + 1)
        )


        if q >= observed - 1e-12:

            n_extreme += 1


        n_total += 1


    return (
        observed,
        n_extreme / n_total,
        n_total,
    )


# ============================================================
# EXACT SIGN-FLIP
# ============================================================

def signflip_p(delta):
    """Compute the exact two-sided sign-flip p-value for the mean delta."""

    delta = np.asarray(
        delta,
        dtype=np.float64,
    )

    observed = abs(
        np.mean(
            delta
        )
    )

    values = []


    for signs in product(
        [-1.0, 1.0],
        repeat=len(delta),
    ):

        signs = np.asarray(
            signs
        )

        values.append(
            abs(
                np.mean(
                    delta
                    * signs
                )
            )
        )


    values = np.asarray(
        values
    )


    return float(
        np.mean(
            values
            >= observed - 1e-12
        )
    )


# ============================================================
# HOLM
# ============================================================

def holm_adjust(pvalues):
    """Apply Holm step-down multiplicity correction to a p-value vector."""

    pvalues = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    m = len(
        pvalues
    )

    order = np.argsort(
        pvalues
    )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    running = 0.0


    for rank, idx in enumerate(
        order
    ):

        value = (
            (m - rank)
            * pvalues[idx]
        )

        running = max(
            running,
            value,
        )

        adjusted[idx] = min(
            running,
            1.0,
        )


    return adjusted


# ============================================================
# ANALYSIS
# ============================================================

all_rows = []


for label, column in [
    (
        "M3",
        "m3",
    ),
    (
        "PC1",
        "pc1",
    ),
]:

    x = make_matrix(
        column
    )


    print()
    print("=" * 100)
    print(label)
    print("=" * 100)


    print()
    print(
        "Culture × day matrix:"
    )

    matrix_df = pd.DataFrame(
        x,
        index=cultures,
        columns=DAYS,
    )

    print(
        matrix_df.to_string()
    )


    q, global_p, n_perm = (
        exact_friedman_p(
            x
        )
    )


    print()
    print(
        "EXACT GLOBAL TEST"
    )

    print(
        "Friedman Q :",
        f"{q:.6f}",
    )

    print(
        "Permutations :",
        n_perm,
    )

    print(
        "exact p:",
        f"{global_p:.8f}",
    )


    comparisons = [
        (
            "Day01-Day00",
            0,
            1,
        ),
        (
            "Day02-Day01",
            1,
            2,
        ),
        (
            "Day02-Day00",
            0,
            2,
        ),
    ]


    temporary = []


    for name, a, b in comparisons:

        delta = (
            x[:, b]
            - x[:, a]
        )


        w = wilcoxon(
            delta,
            alternative="two-sided",
            method="exact",
        )


        sf = signflip_p(
            delta
        )


        temporary.append({
            "endpoint":
                label,

            "comparison":
                name,

            "mean_delta":
                float(
                    delta.mean()
                ),

            "median_delta":
                float(
                    np.median(
                        delta
                    )
                ),

            "n_negative":
                int(
                    np.sum(
                        delta < 0
                    )
                ),

            "n_positive":
                int(
                    np.sum(
                        delta > 0
                    )
                ),

            "wilcoxon_p_raw":
                float(
                    w.pvalue
                ),

            "signflip_p_raw":
                float(
                    sf
                ),

            "friedman_Q":
                q,

            "friedman_exact_p":
                global_p,
        })


    wilcoxon_raw = [
        r[
            "wilcoxon_p_raw"
        ]
        for r in temporary
    ]

    signflip_raw = [
        r[
            "signflip_p_raw"
        ]
        for r in temporary
    ]


    wilcoxon_holm = holm_adjust(
        wilcoxon_raw
    )

    signflip_holm = holm_adjust(
        signflip_raw
    )


    print()
    print(
        "PAIRWISE COMPARISONS"
    )


    for i, row in enumerate(
        temporary
    ):

        row[
            "wilcoxon_p_holm"
        ] = float(
            wilcoxon_holm[i]
        )

        row[
            "signflip_p_holm"
        ] = float(
            signflip_holm[i]
        )


        print()
        print(
            row[
                "comparison"
            ]
        )

        print(
            "  Mean Δ:",
            f"{row['mean_delta']:+.6f}",
        )

        print(
            "  signs   :",
            f"{row['n_negative']} neg / "
            f"{row['n_positive']} pos",
        )

        print(
            "  Raw Wilcoxon  :",
            f"{row['wilcoxon_p_raw']:.6f}",
        )

        print(
            "  Wilcoxon Holm :",
            f"{row['wilcoxon_p_holm']:.6f}",
        )

        print(
            "  Raw sign-flip :",
            f"{row['signflip_p_raw']:.6f}",
        )

        print(
            "  Sign-flip Holm:",
            f"{row['signflip_p_holm']:.6f}",
        )


        all_rows.append(
            row
        )


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(
    all_rows
)

out_path = (
    OUT
    / "final_longitudinal_statistics.csv"
)

results.to_csv(
    out_path,
    index=False,
)


print()
print("=" * 100)
print("FINAL TABLE")
print("=" * 100)

print(
    results.to_string(
        index=False
    )
)


print()
print(
    "Saved:",
    out_path,
)

print()
print("DONE")
