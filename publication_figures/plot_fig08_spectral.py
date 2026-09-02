"""Plot Figure 8: Spectral analysis of B(s) and C(s) trunk profiles.

Panels:
  (a) Dominant wavelengths temporal trend (\u03bb_B and \u03bb_C vs year)
  (b) Autocorrelation e-folding lengths for B(s) and C(s) vs year
  (c) B_mean temporal trend with error bars (B_mean \u00b1 B_std)
Requires spectral_summary.csv in RESULTS_DIR.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

import sys
_pkg_dir = Path(__file__).resolve().parent
_root_dir = _pkg_dir.parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from publication_figures.figure_utils import (
        setup_style,
        COLORS,
        save_fig,
        RESULTS_DIR,
        DOUBLE_COL_WIDTH,
    )
except ImportError:
    from figure_utils import (
        setup_style,
        COLORS,
        save_fig,
        RESULTS_DIR,
        DOUBLE_COL_WIDTH,
    )


def main() -> None:
    """Create spectral analysis figure."""
    setup_style()

    # ── Load data ─────────────────────────────────────
    spectral_csv = RESULTS_DIR / "spectral" / "spectral_summary.csv"
    df = pd.read_csv(spectral_csv)

    # Convert wavelength to km
    df["B_lambda_km"] = df["B_lambda_m"] / 1000.0
    df["C_lambda_km"] = df["C_lambda_m"] / 1000.0

    # ── Setup figure ──────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COL_WIDTH, 9.0), sharex=True)

    # Panel (a): Dominant wavelengths vs year
    ax_a = axes[0]
    
    # Plot B(s) wavelength
    ax_a.scatter(
        df["year"],
        df["B_lambda_km"],
        s=40,
        color=COLORS["blue"],
        alpha=0.6,
        zorder=3,
        label="Width $\\lambda_B$"
    )
    ax_a.plot(
        df["year"],
        df["B_lambda_km"],
        color=COLORS["blue"],
        linewidth=1.2,
        alpha=0.5,
        zorder=2,
    )
    
    # Plot C(s) wavelength
    ax_a.scatter(
        df["year"],
        df["C_lambda_km"],
        s=40,
        color=COLORS["vermilion"],
        marker='^',
        alpha=0.6,
        zorder=3,
        label="Curvature $\\lambda_C$"
    )
    ax_a.plot(
        df["year"],
        df["C_lambda_km"],
        color=COLORS["vermilion"],
        linewidth=1.2,
        alpha=0.5,
        linestyle="--",
        zorder=2,
    )
    
    ax_a.set_ylabel("Dominant wavelength (km)", fontsize=10)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="upper right", fontsize=9)
    ax_a.text(0.02, 0.95, "(a)", transform=ax_a.transAxes, fontsize=11, fontweight="bold",
              verticalalignment="top")

    # Panel (b): Autocorrelation e-folding lengths vs year
    ax_b = axes[1]
    ax_b.scatter(
        df["year"],
        df["B_efold_m"],
        s=40,
        color=COLORS["blue"],
        alpha=0.6,
        zorder=3,
        label="Width $l_B^{e}$"
    )
    ax_b.plot(
        df["year"],
        df["B_efold_m"],
        color=COLORS["blue"],
        linewidth=1.2,
        alpha=0.5,
        zorder=2,
    )
    ax_b.scatter(
        df["year"],
        df["C_efold_m"],
        s=40,
        color=COLORS["vermilion"],
        marker='^',
        alpha=0.6,
        zorder=3,
        label="Curvature $l_C^{e}$"
    )
    ax_b.plot(
        df["year"],
        df["C_efold_m"],
        color=COLORS["vermilion"],
        linewidth=1.2,
        alpha=0.5,
        linestyle="--",
        zorder=2,
    )
    ax_b.set_ylabel("ACF $e$-folding length (m)", fontsize=10)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=9)
    ax_b.text(0.02, 0.95, "(b)", transform=ax_b.transAxes, fontsize=11, fontweight="bold",
              verticalalignment="top")

    # Panel (c): B_mean \u00b1 B_std vs year
    ax_c = axes[2]
    ax_c.errorbar(
        df["year"],
        df["B_mean"],
        yerr=df["B_std"],
        fmt="o",
        color=COLORS["vermilion"],
        ecolor=COLORS["vermilion"],
        elinewidth=0.8,
        capsize=3,
        markersize=4,
        alpha=0.6,
        zorder=3,
    )
    ax_c.plot(
        df["year"],
        df["B_mean"],
        color=COLORS["vermilion"],
        linewidth=1.2,
        alpha=0.5,
        zorder=2,
    )
    ax_c.set_xlabel("Year", fontsize=10)
    ax_c.set_ylabel("Wet width $\\overline{B}_\\mathrm{wet}$ (m)", fontsize=10)
    ax_c.grid(True, alpha=0.3)
    ax_c.text(0.02, 0.95, "(c)", transform=ax_c.transAxes, fontsize=11, fontweight="bold",
              verticalalignment="top")

    # Year ticks
    for ax in axes:
        ax.set_xlim(1999, 2026)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    # ── Finalize ──────────────────────────────────────
    fig.tight_layout()
    save_fig(fig, "fig08_spectral")
    plt.close(fig)


if __name__ == "__main__":
    main()
