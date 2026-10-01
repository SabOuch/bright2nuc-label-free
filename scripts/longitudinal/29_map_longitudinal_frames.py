"""
Map longitudinal tracking frames to filtered brightfield time points.

Purpose
-------
For each of the six longitudinal cultures and three acquisition days, align
tracking-stack frame indices with the corresponding filtered brightfield TIFF
time points. Matching is based on downsampled image correlation and solved
globally as a one-to-one assignment.

External data
-------------
The longitudinal dataset root is read from ``BRIGHT2NUC_DATA_ROOT``. If the
variable is not defined, ``data/raw/`` inside the repository is used.

Inputs
------
Expected below the longitudinal dataset root:
- ``Single_cell_velocities_In-silico_3D/In-silico_3D/``
- ``Single_cell_velocities_Brightfield_3D/Brightfield_3D/results_filtered/``

Outputs
-------
Written under ``outputs/longitudinal/``:
- ``frame_to_bf_mapping.csv``
- ``frame_mapping_summary.csv``

Notes
-----
The assignment is one-to-one so that the same brightfield time point cannot be
assigned to multiple tracking frames.
"""

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import tifffile

from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]

ZENODO = Path(
    os.environ.get(
        "BRIGHT2NUC_DATA_ROOT",
        ROOT / "data" / "raw",
    )
)

INS = (
    ZENODO
    / "Single_cell_velocities_In-silico_3D"
    / "In-silico_3D"
)

FILTERED = (
    ZENODO
    / "Single_cell_velocities_Brightfield_3D"
    / "Brightfield_3D"
    / "results_filtered"
)

OUT = (
    ROOT
    / "outputs"
    / "longitudinal"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


CULTURES = [
    "02A",
    "05D",
    "08A",
    "11D",
    "14A",
    "17D",
]

DAYS = [
    "Day00",
    "Day01",
    "Day02",
]


def get_t(path):
    """Extract the integer time index from a filtered brightfield filename."""

    m = re.search(
        r"_t(\d+)_filtered\.tif$",
        path.name,
    )

    if m is None:
        raise RuntimeError(
            f"Unable to extract t: {path.name}"
        )

    return int(
        m.group(1)
    )


rows = []
summary = []


print("=" * 88)
print("MAPPING FRAME TRACKING -> BF tX")
print("=" * 88)


for culture in CULTURES:

    for day in DAYS:

        stem = (
            f"{culture}_Live5min_{day}"
        )

        stack_path = (
            INS
            / f"{stem}.tif"
        )

        candidates = sorted(
            FILTERED.glob(
                f"{stem}_t*_filtered.tif"
            ),
            key=get_t,
        )


        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            culture,
            day,
        )
        print(
            "------------------------------------------------------------"
        )


        if not stack_path.exists():

            print("❌ Missing stack")

            continue


        stack = tifffile.imread(
            stack_path
        )


        print(
            "Stack      :",
            stack.shape,
        )

        print(
            "Candidates:",
            len(candidates),
        )


        # --------------------------------------------------------
        # Spatial reduction used only to compute the
        # correlations quickly.
        # --------------------------------------------------------

        A = (
            stack[
                :,
                ::2,
                ::8,
                ::8,
            ]
            .astype(
                np.float32
            )
        )


        candidate_arrays = []

        candidate_ts = []


        for path in candidates:

            arr = tifffile.imread(
                path
            )


            if arr.shape != stack.shape[1:]:

                print(
                    "⚠️ Ignored shape:",
                    path.name,
                    arr.shape,
                )

                continue


            candidate_arrays.append(
                arr[
                    ::2,
                    ::8,
                    ::8,
                ].astype(
                    np.float32
                )
            )

            candidate_ts.append(
                get_t(path)
            )


        if len(candidate_arrays) < stack.shape[0]:

            raise RuntimeError(
                f"{culture} {day}: "
                "not enough compatible candidates."
            )


        B = np.stack(
            candidate_arrays
        )


        # --------------------------------------------------------
        # Vectorized Pearson correlation
        # --------------------------------------------------------

        A = A.reshape(
            A.shape[0],
            -1,
        )

        B = B.reshape(
            B.shape[0],
            -1,
        )


        A -= A.mean(
            axis=1,
            keepdims=True,
        )

        B -= B.mean(
            axis=1,
            keepdims=True,
        )


        A_std = A.std(
            axis=1,
            keepdims=True,
        )

        B_std = B.std(
            axis=1,
            keepdims=True,
        )


        A_std[
            A_std == 0
        ] = 1

        B_std[
            B_std == 0
        ] = 1


        A /= A_std
        B /= B_std


        corr = (
            A @ B.T
        ) / A.shape[1]


        # --------------------------------------------------------
        # Global one-to-one matching.
        #
        # This prevents the same tX from being assigned to two FRAME values.
        # --------------------------------------------------------

        frame_idx, candidate_idx = (
            linear_sum_assignment(
                -corr
            )
        )


        assignment = {
            int(frame):
            int(candidate)
            for frame, candidate
            in zip(
                frame_idx,
                candidate_idx,
            )
        }


        mapping = []

        correlations = []


        for frame in range(
            stack.shape[0]
        ):

            j = assignment[
                frame
            ]

            t = candidate_ts[
                j
            ]

            score = float(
                corr[
                    frame,
                    j,
                ]
            )


            order = np.argsort(
                corr[frame]
            )[::-1]


            second = next(
                int(x)
                for x in order
                if int(x) != j
            )


            second_t = (
                candidate_ts[
                    second
                ]
            )

            second_corr = float(
                corr[
                    frame,
                    second,
                ]
            )


            mapping.append(
                t
            )

            correlations.append(
                score
            )


            rows.append({
                "culture":
                    culture,

                "day":
                    day,

                "frame":
                    frame,

                "bf_t":
                    t,

                "correlation":
                    score,

                "second_t":
                    second_t,

                "second_correlation":
                    second_corr,

                "margin":
                    score
                    - second_corr,
            })


        used = set(
            mapping
        )

        unused = sorted(
            set(candidate_ts)
            - used
        )


        natural = (
            mapping
            == list(
                range(
                    len(mapping)
                )
            )
        )


        print(
            "FRAME -> t :",
            ", ".join(
                f"{i}->{t}"
                for i, t
                in enumerate(mapping)
            )
        )


        print(
            "Unused t values:",
            unused,
        )


        print(
            "Natural mapping:",
            "YES"
            if natural
            else "NO",
        )


        print(
            "Mean correlation:",
            f"{np.mean(correlations):.6f}",
        )

        print(
            "Minimum correlation:",
            f"{np.min(correlations):.6f}",
        )


        summary.append({
            "culture":
                culture,

            "day":
                day,

            "n_frames_stack":
                stack.shape[0],

            "n_candidates":
                len(candidate_ts),

            "natural_mapping":
                natural,

            "unused_t":
                ",".join(
                    map(
                        str,
                        unused,
                    )
                ),

            "mean_correlation":
                float(
                    np.mean(
                        correlations
                    )
                ),

            "min_correlation":
                float(
                    np.min(
                        correlations
                    )
                ),

            "min_margin":
                float(
                    pd.DataFrame(
                        rows
                    )
                    .query(
                        "culture == @culture "
                        "and day == @day"
                    )[
                        "margin"
                    ]
                    .min()
                ),
        })


mapping_df = pd.DataFrame(
    rows
)

summary_df = pd.DataFrame(
    summary
)


mapping_path = (
    OUT
    / "frame_to_bf_mapping.csv"
)

summary_path = (
    OUT
    / "frame_mapping_summary.csv"
)


mapping_df.to_csv(
    mapping_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


print()
print("=" * 88)
print("GLOBAL SUMMARY")
print("=" * 88)

print(
    summary_df.to_string(
        index=False
    )
)


print()
print(
    "Natural mappings:",
    int(
        summary_df[
            "natural_mapping"
        ].sum()
    ),
    "/",
    len(summary_df),
)


print()
print(
    "Minimum global correlation:",
    f"{summary_df['min_correlation'].min():.6f}",
)


print()
print("Saved files:")
print(mapping_path)
print(summary_path)

print()
print("DONE")
