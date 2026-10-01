"""
Create the publication-style diagram of the final Bright2Nuc prediction system.

Purpose
-------
Summarize the label-free brightfield input, per-nucleus crop representation,
M3 ConvNeXt-Tiny multitask model, five-seed deep ensemble, auxiliary training
targets, final differentiation prediction, and uncertainty-aware selective
prediction.

Output
------
Written under ``outputs/figures/final/``:
- ``final_pipeline_publication.png``
- ``final_pipeline_publication.pdf``

Notes
-----
The auxiliary OCT4/FOXA2/SOX17 targets are training supervision only and are
never used as inference inputs. The diagram describes the evaluated final
system; it does not imply external deployment.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/figures/final"
OUT.mkdir(parents=True, exist_ok=True)

png_path = OUT / "final_pipeline_publication.png"
pdf_path = OUT / "final_pipeline_publication.pdf"

fig, ax = plt.subplots(figsize=(16, 8), dpi=180)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# ------------------------------------------------------------------
# COLORS
# ------------------------------------------------------------------
bg = "#fcfcfd"
input_c = "#dbeafe"       # light blue
crop_c = "#dcfce7"        # light green
m3_c = "#fde68a"          # warm yellow
ens_c = "#ddd6fe"         # light purple
final_c = "#bfdbfe"       # blue
uncert_c = "#fecdd3"      # pink
aux_c = "#f3f4f6"         # light gray
edge = "#111827"
arrow_c = "#1f2937"
subtitle_c = "#374151"

fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def add_box(x, y, w, h, title, body, fc, ec=edge, lw=2.0,
            dashed=False, title_size=15, body_size=12):
    """Draw one rounded pipeline box with a title and multiline body."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2, y + h * 0.72, title,
        ha="center", va="center",
        fontsize=title_size, fontweight="bold", color="#111111"
    )

    ax.text(
        x + w / 2, y + h * 0.34, body,
        ha="center", va="center",
        fontsize=body_size, color="#222222",
        linespacing=1.35
    )

def add_arrow(x1, y1, x2, y2, style="-|>", lw=1.8):
    """Draw a directional connection between two pipeline elements."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=20,
        linewidth=lw,
        color=arrow_c,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)

# ------------------------------------------------------------------
# TITLES
# ------------------------------------------------------------------
ax.text(
    8, 8.55,
    "Bright2Nuc — final brightfield-only prediction pipeline",
    ha="center", va="center",
    fontsize=22, fontweight="bold", color="#111111"
)

ax.text(
    8, 7.72,
    "Culture-wise 5-fold out-of-fold evaluation",
    ha="center", va="center",
    fontsize=18, fontweight="bold", color="#111111"
)

ax.text(
    8, 7.20,
    "No nuclei from held-out cultures are used for training",
    ha="center", va="center",
    fontsize=12.5, color=subtitle_c
)

# ------------------------------------------------------------------
# MAIN PIPELINE BOXES
# ------------------------------------------------------------------
add_box(
    0.4, 4.8, 2.2, 1.9,
    "Brightfield input",
    "Label-free microscopy\nNo fluorescence input",
    input_c
)

add_box(
    3.0, 4.8, 2.4, 1.9,
    "Per-nucleus crops",
    "3D brightfield stacks\n16 × 64 × 64",
    crop_c
)

add_box(
    5.8, 4.55, 2.7, 2.25,
    "M3",
    "ConvNeXt-Tiny\nmultitask learning\n\nPrimary output:\ndifferentiation score",
    m3_c
)

add_box(
    9.0, 4.55, 2.7, 2.25,
    "Deep ensemble",
    "5 seeds per fold\n\nMean prediction\n+ disagreement (SD)",
    ens_c
)

add_box(
    12.2, 5.05, 2.9, 1.85,
    "Final prediction",
    "Differentiation score\nR² = 0.718\nMAE = 0.117",
    final_c
)

add_box(
    12.0, 1.45, 3.5, 2.05,
    "Uncertainty-aware output",
    "Ensemble SD estimates uncertainty\nSupports selective prediction\n\nReject top 20% uncertain:\nR² = 0.827, MAE = 0.084",
    uncert_c,
    title_size=13.5
)

# Auxiliary supervision box
add_box(
    5.45, 1.0, 3.5, 1.65,
    "Auxiliary training targets",
    "OCT4\nFOXA2\nSOX17",
    aux_c,
    dashed=True,
    title_size=13.5
)

ax.text(
    7.2, 0.55,
    "Training supervision only — never used as model input",
    ha="center", va="center",
    fontsize=11.5, color="#444444", style="italic"
)

# ------------------------------------------------------------------
# ARROWS
# ------------------------------------------------------------------
add_arrow(2.6, 5.75, 3.0, 5.75)
add_arrow(5.4, 5.75, 5.8, 5.75)
add_arrow(8.5, 5.75, 9.0, 5.75)
add_arrow(11.7, 5.75, 12.2, 5.95)

# Auxiliary dashed arrow toward M3
aux_arrow = FancyArrowPatch(
    (7.2, 2.65), (7.2, 4.55),
    arrowstyle="-|>",
    mutation_scale=18,
    linewidth=1.8,
    linestyle="--",
    color=arrow_c,
    shrinkA=2,
    shrinkB=2,
)
ax.add_patch(aux_arrow)

# Arrow from ensemble to uncertainty box
add_arrow(10.35, 4.55, 13.6, 3.5)

# ------------------------------------------------------------------
# SMALL ACCENT LABELS
# ------------------------------------------------------------------
ax.text(
    7.15, 6.95, "Best single-model family",
    ha="center", va="center",
    fontsize=11.5, color="#92400e",
    bbox=dict(boxstyle="round,pad=0.25", fc="#fffbeb", ec="#f59e0b", alpha=0.95)
)

ax.text(
    13.65, 7.25, "Final system",
    ha="center", va="center",
    fontsize=11.5, color="#1d4ed8",
    bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#60a5fa", alpha=0.95)
)

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
plt.tight_layout()
fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())

print("=" * 90)
print("FINAL PIPELINE FIGURE")
print("=" * 90)
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("DONE")
