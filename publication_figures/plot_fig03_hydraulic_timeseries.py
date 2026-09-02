"""Publication figure: Hydraulic parameters time series (β, Fr, C_f).

Generates 3-panel figure showing temporal evolution of dimensionless hydraulic
parameters over the Gaocun–Sunkou reach (2000–2021).

Usage
-----
python plot_fig03_hydraulic_timeseries.py
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
        COLOR_CYCLE,
    )
except ImportError:
    from figure_utils import (
        setup_style,
        COLORS,
        save_fig,
        RESULTS_DIR,
        DOUBLE_COL_WIDTH,
        COLOR_CYCLE,
    )


def plot_hydraulic_timeseries() -> None:
    """Create 3-panel hydraulic time series figure."""
    # ── Load data ────────────────────────────────────────
    csv_path = RESULTS_DIR / "hydraulic_params_timeseries.csv"
    df = pd.read_csv(csv_path)
    
    # ── Setup style ──────────────────────────────────────
    setup_style()
    
    # ── Create figure ────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(DOUBLE_COL_WIDTH, 8.0),
        sharex=True,
    )
    
    # ── Panel (a): β = B/H ─────────────────────────────
    ax_a = axes[0]
    mask_beta = df["beta"].notna()
    years_beta = df.loc[mask_beta, "year"].values
    beta_vals = df.loc[mask_beta, "beta"].values
    
    ax_a.scatter(years_beta, beta_vals, color=COLOR_CYCLE[0], s=30, zorder=3, label="β")
    ax_a.plot(years_beta, beta_vals, color=COLOR_CYCLE[0], linewidth=1.0, alpha=0.7)
    
    ax_a.set_ylabel(r"Aspect ratio $\beta$", fontsize=10)
    ax_a.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)
    ax_a.grid(True, which="minor", alpha=0.15, linestyle="--", linewidth=0.3)
    ax_a.minorticks_on()
    ax_a.text(0.02, 0.95, "(a)", transform=ax_a.transAxes, 
              fontsize=11, fontweight="bold", verticalalignment="top")
    
    # ── Panel (b): Fr ───────────────────────────────────
    ax_b = axes[1]
    mask_fr = df["Fr"].notna()
    years_fr = df.loc[mask_fr, "year"].values
    fr_vals = df.loc[mask_fr, "Fr"].values
    
    ax_b.scatter(years_fr, fr_vals, color=COLOR_CYCLE[1], s=30, zorder=3, label="Fr")
    ax_b.plot(years_fr, fr_vals, color=COLOR_CYCLE[1], linewidth=1.0, alpha=0.7)
    
    ax_b.set_ylabel(r"Froude number $\mathrm{Fr}$", fontsize=10)
    ax_b.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)
    ax_b.grid(True, which="minor", alpha=0.15, linestyle="--", linewidth=0.3)
    ax_b.minorticks_on()
    ax_b.text(0.02, 0.95, "(b)", transform=ax_b.transAxes, 
              fontsize=11, fontweight="bold", verticalalignment="top")
    
    # ── Panel (c): C_f ───────────────────────────────────
    ax_c = axes[2]
    # Dual estimation: energy slope (primary) vs Manning conversion (secondary)
    mask_cf = df["Cf_energy"].notna()
    years_cf = df.loc[mask_cf, "year"].values
    cf_energy = df.loc[mask_cf, "Cf_energy"].values
    cf_manning = df.loc[mask_cf, "Cf_manning"].values  # may contain NaN

    # Error bars: demonstrate deviation between energy slope and Manning conversion
    yerr_lo = np.zeros_like(cf_energy)
    yerr_hi = np.zeros_like(cf_energy)
    for i in range(len(cf_energy)):
        if not np.isnan(cf_manning[i]):
            lo = min(cf_energy[i], cf_manning[i])
            hi = max(cf_energy[i], cf_manning[i])
            yerr_lo[i] = cf_energy[i] - lo
            yerr_hi[i] = hi - cf_energy[i]

    ax_c.errorbar(
        years_cf, cf_energy,
        yerr=[yerr_lo, yerr_hi],
        fmt='o', markersize=4, color=COLOR_CYCLE[2],
        ecolor=COLOR_CYCLE[2], elinewidth=0.8, capsize=2.5, capthick=0.8,
        zorder=3, label=r"$C_f$ (energy)",
    )
    ax_c.plot(years_cf, cf_energy, color=COLOR_CYCLE[2], linewidth=1.0, alpha=0.7)

    ax_c.set_ylabel(r"$C_f$", fontsize=10)
    ax_c.set_xlabel("Year", fontsize=10)
    ax_c.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)
    ax_c.grid(True, which="minor", alpha=0.15, linestyle="--", linewidth=0.3)
    ax_c.minorticks_on()
    ax_c.text(0.02, 0.95, "(c)", transform=ax_c.transAxes, 
              fontsize=11, fontweight="bold", verticalalignment="top")
    
    # ── Format axes ──────────────────────────────────────
    for ax in axes:
        ax.tick_params(which="major", length=4, width=0.6)
        ax.tick_params(which="minor", length=2, width=0.4)
    
    # Set x-axis limits to show full range with padding
    all_years = df["year"].values
    year_min, year_max = all_years.min(), all_years.max()
    year_pad = (year_max - year_min) * 0.02
    axes[2].set_xlim(year_min - year_pad, year_max + year_pad)
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].xaxis.set_minor_locator(MultipleLocator(1))
    
    # ── Tight layout ─────────────────────────────────────
    fig.tight_layout()
    
    # ── Save ─────────────────────────────────────────────
    save_fig(fig, "fig03_hydraulic_timeseries")


if __name__ == "__main__":
    plot_hydraulic_timeseries()
