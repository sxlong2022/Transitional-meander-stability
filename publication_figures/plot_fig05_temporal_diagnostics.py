"""Publication figure: Temporal diagnostics of planar 2D SWE-Exner stability (σ_r,max, λ_max, curvature).

Script generates Figure 05 showing three-panel temporal evolution:
  (a) Peak exponential growth rate σ_r,max vs year
  (b) Preferred bar wavelength λ_max vs year (with bankfull depth H on secondary axis)
  (c) Curvature modulation E (%) vs year

Run via:
    python publication_figures/plot_fig05_temporal_diagnostics.py
"""
from __future__ import annotations

# stdlib
import sys
from pathlib import Path

# third-party
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd

# local
_pkg_dir = Path(__file__).resolve().parent
_root_dir = _pkg_dir.parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from publication_figures.figure_utils import (
        COLORS,
        DOUBLE_COL_WIDTH,
        RESULTS_DIR,
        save_fig,
        setup_style,
    )
except ImportError:
    from figure_utils import (
        COLORS,
        DOUBLE_COL_WIDTH,
        RESULTS_DIR,
        save_fig,
        setup_style,
    )


def plot_temporal_diagnostics() -> None:
    """Generate 3-panel temporal diagnostics figure."""
    setup_style()

    # Load temporal 2D stability CSV
    csv_path = RESULTS_DIR / "temporal_stability_2d.csv"
    df = pd.read_csv(csv_path)

    # Extract data
    years = df["year"].values
    sigma_r_max = df["sigma_r_max"].values
    lambda_max_m = df["lambda_max_m"].values
    H_m = df["H_m"].values
    curvature_enh_pct = df["E_pct"].values

    # Create 3-row figure
    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COL_WIDTH, 8.5))

    # ── Panel (a): σ_r,max vs year ─────────────────────────────────────────
    ax = axes[0]
    ax.scatter(years, sigma_r_max, color=COLORS["blue"], s=35, alpha=0.8, zorder=3, edgecolors="k", linewidth=0.5)
    ax.plot(years, sigma_r_max, color=COLORS["blue"], linewidth=1.2, alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.6, label=r"Neutral stability ($\sigma_r=0$)")
    ax.set_ylabel(r"Peak growth rate $\sigma_{r,\max}$")
    ax.set_xlim(years.min() - 1, years.max() + 1)
    ax.set_ylim(0.0, 0.35)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    
    ax.text(0.02, 0.92, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold",
            verticalalignment="top")

    # ── Panel (b): λ_max vs year (dual axis) ────────────────────────────
    ax = axes[1]
    ax.scatter(years, lambda_max_m, color=COLORS["green"], s=35, alpha=0.8, zorder=3, edgecolors="k", linewidth=0.5)
    ax.plot(years, lambda_max_m, color=COLORS["green"], linewidth=1.2, alpha=0.7)
    ax.set_ylabel(r"Preferred wavelength $\lambda_{\max}$ (m)", color=COLORS["green"])
    ax.tick_params(axis="y", labelcolor=COLORS["green"])
    ax.set_xlim(years.min() - 1, years.max() + 1)
    ax.set_ylim(800, 1500)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Secondary y-axis for H_m (bankfull depth)
    ax_sec = ax.twinx()
    ax_sec.scatter(years, H_m, color=COLORS["vermilion"], s=25, alpha=0.7, marker="^", zorder=2, edgecolors="k", linewidth=0.5)
    ax_sec.plot(years, H_m, color=COLORS["vermilion"], linewidth=1.0, alpha=0.5, linestyle="--")
    ax_sec.set_ylabel(r"Bankfull depth $H$ (m)", color=COLORS["vermilion"])
    ax_sec.tick_params(axis="y", labelcolor=COLORS["vermilion"])
    ax_sec.set_ylim(1.0, 6.0)

    ax.text(0.02, 0.92, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold",
            verticalalignment="top")

    # ── Panel (c): Curvature modulation (bar chart) ────────────────────────
    ax = axes[2]
    colors_bar = [COLORS["vermilion"] if x < 0 else COLORS["blue"] for x in curvature_enh_pct]
    ax.bar(years, curvature_enh_pct, color=colors_bar, width=0.8, alpha=0.75, edgecolor="k", linewidth=0.4)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Curvature modulation $E$ (%)")
    ax.set_xlim(years.min() - 1, years.max() + 1)
    ax.set_ylim(-3.5, 0.5)
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontsize=11, fontweight="bold",
            verticalalignment="top")

    for ax_i in axes:
        ax_i.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_i.xaxis.set_minor_locator(MultipleLocator(1))

    fig.tight_layout()
    save_fig(fig, "fig05_temporal_diagnostics")
    plt.close(fig)

if __name__ == "__main__":
    plot_temporal_diagnostics()
