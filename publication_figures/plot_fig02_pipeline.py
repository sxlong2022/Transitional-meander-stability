"""Publication figure: integrated diagnostic framework pipeline (Fig 02).

Generates a flowchart showing data sources → processing → analysis → diagnosis.
Uses matplotlib patches and FancyArrowPatch for a clean schematic.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

import sys
_pkg_dir = Path(__file__).resolve().parent
_root_dir = _pkg_dir.parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from publication_figures.figure_utils import setup_style, COLORS, save_fig, DOUBLE_COL_WIDTH
except ImportError:
    from figure_utils import setup_style, COLORS, save_fig, DOUBLE_COL_WIDTH

# ── colour palette for box categories ──────────────────────────────
C_DATA = "#D6EAF8"       # light blue
C_DATA_EDGE = COLORS["blue"]
C_PROC = "#FEF3E2"       # light orange
C_PROC_EDGE = COLORS["orange"]
C_ANLY = "#D5F5E3"       # light green
C_ANLY_EDGE = COLORS["green"]
C_DIAG = "#FADBD8"       # light red
C_DIAG_EDGE = COLORS["vermilion"]

# ── Layout constants ────────────────────────────────────────────
COL_CX = [0.135, 0.378, 0.622, 0.865]   # 4 column x-centres
BW = 0.16    # box width (narrower to leave room for bypass arrow)
BH = 0.11    # standard box height (reduced for uniform spacing)
YMAX = 0.84  # axes upper limit

# ── Row positions (bottom of box) — uniform grid, gap=0.05 ──
ROW = [0.62, 0.46, 0.30, 0.14]  # Row 0 (top) to Row 3 (bottom)


def _box(ax, cx, y, h, fc, ec, title, subtitle="", section=""):
    """Draw a rounded box centred at cx, bottom at y, height h."""
    x = cx - BW / 2
    box = FancyBboxPatch(
        (x, y), BW, h,
        boxstyle="round,pad=0.018",
        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2,
    )
    ax.add_patch(box)
    cy = y + h / 2
    if subtitle:
        ax.text(cx, cy + 0.023, title,
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", zorder=3)
        ax.text(cx, cy - 0.003, subtitle,
                ha="center", va="top", fontsize=6.0, color="#444", zorder=3)
    else:
        ax.text(cx, cy, title,
                ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=3)
    if section:
        ax.text(cx, y + 0.002, section,
                ha="center", va="bottom", fontsize=5.0, color="#888", zorder=3)
    return (x, y, BW, h)


def _arrow(ax, start_box, end_box, start_side="right", end_side="left",
           rad=0.05, **kw):
    """Draw a curved arrow between two boxes."""
    sx, sy, sw, sh = start_box
    ex, ey, ew, eh = end_box
    if start_side == "right":
        xs, ys = sx + sw, sy + sh / 2
    elif start_side == "bottom":
        xs, ys = sx + sw / 2, sy
    elif start_side == "left":
        xs, ys = sx, sy + sh / 2
    else:
        xs, ys = sx + sw, sy + sh / 2

    if end_side == "left":
        xe, ye = ex, ey + eh / 2
    elif end_side == "top":
        xe, ye = ex + ew / 2, ey + eh
    elif end_side == "right":
        xe, ye = ex + ew, ey + eh / 2
    else:
        xe, ye = ex, ey + eh / 2

    arrow = FancyArrowPatch(
        (xs, ys), (xe, ye),
        arrowstyle="->,head_width=3,head_length=3",
        connectionstyle=f"arc3,rad={rad}",
        color=kw.get("color", "#555"),
        linewidth=kw.get("linewidth", 1.2),
        linestyle=kw.get("linestyle", "-"),
        zorder=5,
    )
    ax.add_patch(arrow)


def plot_fig02_pipeline() -> None:
    """Create the framework pipeline schematic."""
    setup_style()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, YMAX)
    ax.axis("off")

    # ── Column header labels (exactly at column centres) ────────────
    col_labels = ["Data Sources", "Processing", "Analysis", "Diagnosis"]
    for cx, label in zip(COL_CX, col_labels):
        ax.text(cx, YMAX - 0.01, label,
                ha="center", va="top",
                fontsize=9, fontweight="bold", color="#222")
        # thin rule under header
        ax.plot([cx - BW / 2, cx + BW / 2], [YMAX - 0.07, YMAX - 0.07],
                color="#aaa", linewidth=0.8, zorder=0)

    # ── Column 1: Data Sources ────────────────────────────────────
    d1 = _box(ax, COL_CX[0], ROW[0], BH, C_DATA, C_DATA_EDGE,
              "Hydrometric Data",
              r"$B$, $H$, $Q$, $n$, $S$, $D_{50}$", "§2.2")
    d2 = _box(ax, COL_CX[0], ROW[1], BH, C_DATA, C_DATA_EDGE,
              "Landsat Imagery",
              "L5/7/8/9, 2000–2025 (GEE)", "§3.2")

    # ── Column 2: Processing ─────────────────────────────────────
    p1 = _box(ax, COL_CX[1], ROW[0], BH, C_PROC, C_PROC_EDGE,
              "Hydraulic Compilation",
              r"$\beta$, $\mathrm{Fr}$, $C_f$, $\theta$ per year", "§2.2")
    p2 = _box(ax, COL_CX[1], ROW[1], BH, C_PROC, C_PROC_EDGE,
              "DSWE Water Masks",
              "Annual median composites", "§3.2")
    p3 = _box(ax, COL_CX[1], ROW[2], BH, C_PROC, C_PROC_EDGE,
              "RivGraph Extraction",
              r"$B(s)$, $C(s)$ + trunk", "§3.2")

    # ── Column 3: Analysis ──────────────────────────────────────
    a1 = _box(ax, COL_CX[2], ROW[0], BH, C_ANLY, C_ANLY_EDGE,
              "OS Stability Solver",
              r"$c_i(\alpha)$, $\alpha_\mathrm{crit}$", "§3.1")
    a2 = _box(ax, COL_CX[2], ROW[1], BH, C_ANLY, C_ANLY_EDGE,
              "Curvature Correction",
              r"$\nu\beta$ product, $E$", "§3.1")
    a3 = _box(ax, COL_CX[2], ROW[2], BH, C_ANLY, C_ANLY_EDGE,
              "Spectral Analysis",
              r"FFT $\to$ $\lambda_B$, $\lambda_C$, ACF", "§3.3")
    a4 = _box(ax, COL_CX[2], ROW[3], BH, C_ANLY, C_ANLY_EDGE,
              "Bar Mode Diagnostics",
              r"$\sigma_\mathrm{width}$, $c_{i,\max}$", "§3.4")

    # ── Column 4: Diagnosis ─────────────────────────────────────
    o1 = _box(ax, COL_CX[3], ROW[0], BH, C_DIAG, C_DIAG_EDGE,
              "Temporal & Spatial",
              "Stability Diagnosis", "§4.1–4.3")
    o2 = _box(ax, COL_CX[3], ROW[1], BH, C_DIAG, C_DIAG_EDGE,
              "Scale Separation",
              r"$\lambda_\mathrm{OS}$ / $\lambda_B$", "§5.1")
    o3 = _box(ax, COL_CX[3], ROW[2], BH, C_DIAG, C_DIAG_EDGE,
              "Cross-Enhancement",
              r"$\sigma_\mathrm{width}$ regime", "§5.3")

    # ── Arrows ───────────────────────────────────────────────────
    # Data → Processing
    _arrow(ax, d1, p1)
    _arrow(ax, d2, p2)

    # Processing → Analysis
    _arrow(ax, p1, a1)
    _arrow(ax, p1, a2)
    _arrow(ax, p2, p3, start_side="bottom", end_side="top")  # masks → rivgraph (vertical, same column)
    _arrow(ax, p3, a3)
    _arrow(ax, p3, a4)
    # OS solver feeds into bar mode diagnostics (bypass left of Analysis col)
    sx, sy, sw, sh = a1
    ex, ey, ew, eh = a4
    xs, ys = sx, sy + sh / 2          # a1 left edge, mid-height
    xe, ye = ex, ey + eh / 2          # a4 left edge, mid-height
    xw = sx - 0.035                   # waypoint x: left of column
    # Three-segment dashed path: left → down → right-to-target
    ax.plot([xs, xw, xw, xe], [ys, ys, ye, ye],
            color="#888", linewidth=1.0, linestyle="--", zorder=4)
    # Arrowhead at the end
    ax.annotate("",
               xy=(xe, ye), xytext=(xw, ye),
               arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.08",
                              color="#888", linewidth=1.0, linestyle="-",
                              shrinkA=0, shrinkB=0))

    # Analysis → Diagnosis
    _arrow(ax, a1, o1)
    _arrow(ax, a2, o1)
    _arrow(ax, a1, o2)   # OS → scale separation
    _arrow(ax, a3, o2)
    _arrow(ax, a4, o3)


    # ── Save ──────────────────────────────────────────────────────
    OUTPUT_DIR = Path(__file__).resolve().parent / "output"
    fig.savefig(str(OUTPUT_DIR / "fig02_pipeline.pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(str(OUTPUT_DIR / "fig02_pipeline.png"), dpi=150, bbox_inches="tight")
    print("Figure 2 (pipeline schematic) saved.")


if __name__ == "__main__":
    plot_fig02_pipeline()
