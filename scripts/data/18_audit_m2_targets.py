"""
Audit the provenance and metadata of the fluorescence-derived M2 targets.

Purpose
-------
Inspect the available transcription-factor resources before freezing the M2
auxiliary supervision. The script:
- inspects the reconstructed per-nucleus TF table when available;
- summarizes dataset tags and ``TF_class``;
- inventories the original TF TIFF files;
- searches TIFF/OME/ImageJ metadata for channel and time-related information;
- searches accompanying text-like metadata files for relevant marker names.

Inputs
------
- ``data/raw/TF_prediction/Positions_Classes.csv``
- ``data/raw/TF_prediction/Rawdata/TF/``
- optionally ``outputs/data_audit/tf_expression_reconstruction.csv``

Outputs
-------
No dataset or target file is created. Results are printed to the console.

Notes
-----
This script is deliberately an audit only. It does not infer missing biological
time information and does not create or modify M2 targets.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/TF_prediction"
TF_RAW = RAW / "Rawdata/TF"
POS_PATH = RAW / "Positions_Classes.csv"

EXPR_PATH = (
    ROOT
    / "outputs/data_audit/tf_expression_reconstruction.csv"
)


print("=" * 88)
print("AUDIT OF THE M2 AUXILIARY TARGETS")
print("=" * 88)


# =====================================================================
# 1. TF EXPRESSION TABLE ALREADY PRODUCED
# =====================================================================

print("\n" + "=" * 88)
print("1. EXISTING TF EXPRESSION TABLE")
print("=" * 88)

if EXPR_PATH.exists():

    expr = pd.read_csv(EXPR_PATH)

    print("File :", EXPR_PATH)
    print("Rows :", len(expr))

    print("\nColumns:")

    for c in expr.columns:
        print(" -", c)

    print("\nFirst rows:")
    print(
        expr.head(3).to_string(index=False)
    )

    numeric = expr.select_dtypes(
        include=[np.number]
    )

    print("\nNumeric columns:")
    print(
        list(numeric.columns)
    )

else:

    print(
        "WARNING: "
        "tf_expression_reconstruction.csv is missing."
    )


# =====================================================================
# 2. DATASET TAGS + TF_CLASS
# =====================================================================

print("\n" + "=" * 88)
print("2. CURRENT TAGS AND TF_CLASS")
print("=" * 88)

pos = pd.read_csv(POS_PATH)

pos["culture"] = pos["filename"].str.replace(
    r"\.tif\d+\.tif$",
    ".tif",
    regex=True,
)

pos["dataset_tag"] = pos["culture"].str.extract(
    r"(Day\d+|EndPoint)",
    expand=False,
)

print(
    pos.groupby("dataset_tag")
    .agg(
        n_nuclei=("filename", "size"),
        n_cultures=("culture", "nunique"),
        dl_mean=("TF_class", "mean"),
        dl_std=("TF_class", "std"),
        dl_median=("TF_class", "median"),
    )
    .to_string()
)


# =====================================================================
# 3. RAW TF TIFF INVENTORY
# =====================================================================

print("\n" + "=" * 88)
print("3. RAW TF TIFF FILES")
print("=" * 88)

tf_files = sorted(
    TF_RAW.glob("*.tif")
)

print("Number of files:", len(tf_files))

signatures = {}

for path in tf_files:

    with tifffile.TiffFile(path) as tif:

        series = tif.series[0]

        key = (
            tuple(series.shape),
            str(series.dtype),
            str(series.axes),
        )

        signatures.setdefault(
            key,
            [],
        ).append(path.name)


print("\nSignatures shape / dtype / axes :")

for key, names in signatures.items():

    print(
        f"\nshape={key[0]} "
        f"dtype={key[1]} "
        f"axes={key[2]} "
        f"n={len(names)}"
    )

    print(
        " examples:",
        ", ".join(names[:5])
    )


# =====================================================================
# 4. TIFF METADATA SEARCH
# =====================================================================

print("\n" + "=" * 88)
print("4. SEARCH TIFF METADATA")
print("=" * 88)

patterns = [
    r"OCT4",
    r"FOXA2",
    r"SOX17",
    r"DAPI",
    r"SOX2",
    r"\b0\s*h\b",
    r"\b24\s*h\b",
    r"\b48\s*h\b",
    r"\b72\s*h\b",
    r"time",
    r"hour",
    r"channel",
]

compiled = [
    re.compile(
        p,
        flags=re.IGNORECASE,
    )
    for p in patterns
]

n_files_with_hits = 0

for path in tf_files:

    texts = []

    with tifffile.TiffFile(path) as tif:

        if tif.ome_metadata:
            texts.append(
                (
                    "OME",
                    str(tif.ome_metadata),
                )
            )

        if tif.imagej_metadata:
            texts.append(
                (
                    "ImageJ",
                    str(tif.imagej_metadata),
                )
            )

        if len(tif.pages):

            description = (
                tif.pages[0].description
            )

            if description:
                texts.append(
                    (
                        "Description",
                        str(description),
                    )
                )

            # All first-page TIFF tags as text
            tags_text = "\n".join(
                f"{tag.name}={tag.value}"
                for tag in tif.pages[0].tags.values()
            )

            texts.append(
                (
                    "TIFF tags",
                    tags_text,
                )
            )

    file_hits = []

    for source_name, text in texts:

        for regex in compiled:

            m = regex.search(text)

            if m:

                start = max(
                    0,
                    m.start() - 100,
                )

                end = min(
                    len(text),
                    m.end() + 200,
                )

                snippet = (
                    text[start:end]
                    .replace("\n", " ")
                    .replace("\r", " ")
                )

                file_hits.append(
                    (
                        source_name,
                        regex.pattern,
                        snippet,
                    )
                )

    if file_hits:

        n_files_with_hits += 1

        print(
            f"\n--- {path.name} ---"
        )

        # Avoid flooding terminal with duplicates.
        seen = set()

        for source_name, pattern, snippet in file_hits:

            key = (
                source_name,
                pattern,
                snippet,
            )

            if key in seen:
                continue

            seen.add(key)

            print(
                f"[{source_name}] "
                f"match={pattern}"
            )

            print(
                " ",
                snippet[:500],
            )


print(
    "\nFiles containing at least one keyword:",
    f"{n_files_with_hits}/{len(tf_files)}",
)


# =====================================================================
# 5. EXPLICIT OME CHANNEL NAMES
# =====================================================================

print("\n" + "=" * 88)
print("5. OME CHANNEL NAMES")
print("=" * 88)

channel_name_pattern = re.compile(
    r'<Channel[^>]*?(?:Name="([^"]*)")?[^>]*>',
    flags=re.IGNORECASE,
)

ome_found = False

for path in tf_files:

    with tifffile.TiffFile(path) as tif:

        ome = tif.ome_metadata

    if not ome:
        continue

    matches = (
        channel_name_pattern
        .findall(ome)
    )

    matches = [
        x
        for x in matches
        if x
    ]

    if matches:

        ome_found = True

        print(
            path.name,
            "->",
            matches,
        )


if not ome_found:
    print(
        "No explicit channel name was "
        "found in the OME metadata."
    )


# =====================================================================
# 6. SEARCH TEXT-LIKE FILES IN ARCHIVE
# =====================================================================

print("\n" + "=" * 88)
print("6. OTHER METADATA FILES")
print("=" * 88)

allowed_suffixes = {
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".md",
}

text_files = [
    p
    for p in RAW.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower()
        in allowed_suffixes
    )
]

print(
    "Text files found:",
    len(text_files),
)

keywords = re.compile(
    r"OCT4|FOXA2|SOX17|"
    r"\b24\s*h\b|"
    r"\b48\s*h\b|"
    r"\b72\s*h\b",
    flags=re.IGNORECASE,
)

for path in text_files:

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:
        continue

    match = keywords.search(text)

    if match:

        start = max(
            0,
            match.start() - 150,
        )

        end = min(
            len(text),
            match.end() + 300,
        )

        print(
            f"\n--- {path.relative_to(ROOT)} ---"
        )

        print(
            text[start:end]
            .replace("\n", " ")[:600]
        )


print("\n" + "=" * 88)
print("CONCLUSION")
print("=" * 88)

print(
    "This script performs an audit only."
)
print(
    "No M2 target was created."
)
print(
    "No biological mapping was assumed."
)

print("\nDONE")
