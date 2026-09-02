"""Publication figure: spatial hydraulic parameters along 2016 reach (26 cross-sections).

Panels:
  (a) Aspect ratio β vs reach position (dam_km)
  (b) Froude number Fr vs reach position
  (c) Friction coefficient Cf vs reach position
  (d) Shields number θ vs reach position
"""
from __future__ import annotations

# stdlib
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local
import sys
_pkg_dir = Path(__file__).resolve().parent
_root_dir = _pkg_dir.parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from publication_figures.figure_utils import setup_style, COLORS, save_fig, RESULTS_DIR, DOUBLE_COL_WIDTH
except ImportError:
    from figure_utils import setup_style, COLORS, save_fig, RESULTS_DIR, DOUBLE_COL_WIDTH

def main():
    """Create 4-panel spatial hydraulic parameter figure for 2016 reach."""
    setup_style()
    
    # Load data
    data_file = RESULTS_DIR / "hydraulic_params_spatial_2016.csv"
    df = pd.read_csv(data_file)
    
    # Sort by dam_km
    df = df.sort_values("dam_km").reset_index(drop=True)
    
    # Create figure with 4 vertically stacked panels
    fig, axes = plt.subplots(4, 1, figsize=(DOUBLE_COL_WIDTH, 9), sharex=True)
    
    # Panel (a): Aspect ratio β
    ax = axes[0]
    ax.scatter(df["dam_km"], df["beta"], s=30, alpha=0.7, color=COLORS["blue"], zorder=3)
    ax.plot(df["dam_km"], df["beta"], linewidth=0.8, alpha=0.4, color=COLORS["blue"], zorder=2)
    ax.set_ylabel(r"Aspect ratio $\beta$", fontsize=10)
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=10, 
            verticalalignment="top", horizontalalignment="left", fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    
    # Panel (b): Froude number
    ax = axes[1]
    ax.scatter(df["dam_km"], df["Fr"], s=30, alpha=0.7, color=COLORS["vermilion"], zorder=3)
    ax.plot(df["dam_km"], df["Fr"], linewidth=0.8, alpha=0.4, color=COLORS["vermilion"], zorder=2)
    ax.set_ylabel(r"Froude number $\mathrm{Fr}$", fontsize=10)
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=10, 
            verticalalignment="top", horizontalalignment="left", fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    
    # Panel (c): Friction coefficient
    ax = axes[2]
    ax.scatter(df["dam_km"], df["Cf_energy"], s=30, alpha=0.7, color=COLORS["green"], zorder=3)
    ax.plot(df["dam_km"], df["Cf_energy"], linewidth=0.8, alpha=0.4, color=COLORS["green"], zorder=2)
    ax.set_ylabel(r"Friction $C_f$", fontsize=10)
    ax.text(0.02, 0.95, "(c)", transform=ax.transAxes, fontsize=10, 
            verticalalignment="top", horizontalalignment="left", fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    
    # Panel (d): Shields number
    ax = axes[3]
    ax.scatter(df["dam_km"], df["Shields"], s=30, alpha=0.7, color=COLORS["orange"], zorder=3)
    ax.plot(df["dam_km"], df["Shields"], linewidth=0.8, alpha=0.4, color=COLORS["orange"], zorder=2)
    ax.set_ylabel(r"Shields $\theta$", fontsize=10)
    ax.set_xlabel("Distance from dam (km)", fontsize=10)
    ax.text(0.02, 0.95, "(d)", transform=ax.transAxes, fontsize=10, 
            verticalalignment="top", horizontalalignment="left", fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    
    fig.tight_layout()
    save_fig(fig, "fig04_spatial_hydraulic")


if __name__ == "__main__":
    main()
