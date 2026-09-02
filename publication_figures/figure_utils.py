"""Publication figure utilities — AWR style (Okabe-Ito palette, Times New Roman).

Usage
-----
>>> from publication_figures.figure_utils import setup_style, COLORS, save_fig
>>> setup_style()
>>> fig, ax = plt.subplots()
>>> save_fig(fig, "fig01_study_area")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────
FIG_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR.mkdir(exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REVISION_FIG_DIR = PROJECT_ROOT / "manuscript" / "ESPL_revision"
# ── Okabe-Ito colorblind-safe palette ─────────────────────
COLORS = {
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "black":     "#000000",
}
COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["vermilion"],
    COLORS["green"],
    COLORS["orange"],
    COLORS["skyblue"],
    COLORS["purple"],
    COLORS["yellow"],
    COLORS["black"],
]

# ── Journal style setup ───────────────────────────────────
def setup_style() -> None:
    """Set matplotlib rcParams for AWR / JFM publication figures."""
    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        # Axes
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
        # Ticks
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.top": True,
        "ytick.right": True,
        # Legend
        "legend.fontsize": 8,
        "legend.frameon": False,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        # Savefig
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Figure
        "figure.dpi": 150,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF (vector) and PNG (preview)."""
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    print(f"Saved: {FIG_DIR / name}.pdf / .png")
    if REVISION_FIG_DIR.exists():
        fig.savefig(REVISION_FIG_DIR / f"{name}.pdf")
        print(f"Copied to revision workspace: {REVISION_FIG_DIR / name}.pdf")

# ── Column widths (Elsevier single/double) ────────────────
SINGLE_COL_WIDTH = 3.5   # inches (single column, ~89 mm)
DOUBLE_COL_WIDTH = 7.0   # inches (double column, ~178 mm)
