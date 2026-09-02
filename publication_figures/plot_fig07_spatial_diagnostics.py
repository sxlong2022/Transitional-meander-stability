"""Publication figure: spatial 2D SWE-Exner diagnostics along 2016 reach (26 cross-sections).

Panel:
  (a) Peak growth rate σ_r,max and preferred wavenumber k_max vs reach position
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

def plot_fig07_spatial_diagnostics() -> None:
    """Create single-panel spatial diagnostics figure for 2016 Gaocun-Sunkou reach."""
    setup_style()
    
    # Read spatial 2D diagnostics CSV
    csv_path = RESULTS_DIR / "spatial_2016_stability_2d.csv"
    df = pd.read_csv(csv_path)
    
    # Sort by dam_km for plotting
    df = df.sort_values("dam_km").reset_index(drop=True)
    
    # Extract data
    dam_km = df["dam_km"].values
    sigma_r_max = df["sigma_r_max"].values
    k_max = df["k_max"].values
    
    # Create figure: single panel
    fig, ax_a = plt.subplots(
        1, 1,
        figsize=(DOUBLE_COL_WIDTH, 3.4),
    )

    # ──────────────────────────────────────────────────────────────────
    # σ_r,max vs dam_km (primary) + k_max (secondary)
    # ──────────────────────────────────────────────────────────────────
    # Unstable vs damped mask
    unstable_mask = sigma_r_max > 0
    damped_mask = ~unstable_mask

    # Plot unstable points
    ax_a.scatter(dam_km[unstable_mask], sigma_r_max[unstable_mask], c=COLORS["blue"], s=35, alpha=0.85, zorder=4, edgecolors="k", linewidth=0.5, label=r"Unstable ($\sigma_{r,\max} > 0$)")
    # Plot damped points
    ax_a.scatter(dam_km[damped_mask], sigma_r_max[damped_mask], c="grey", s=45, marker="X", alpha=0.9, zorder=5, edgecolors="k", linewidth=0.6, label=r"Linearly stable ($\sigma_{r,\max} < 0$)")
    ax_a.plot(dam_km, sigma_r_max, color=COLORS["blue"], linewidth=0.9, alpha=0.4, zorder=2)
    
    # Horizontal line at σ_r = 0 (neutral stability)
    ax_a.axhline(y=0, color="k", linestyle="--", linewidth=0.9, alpha=0.6, label=r"Neutral threshold ($\sigma_r = 0$)")
    
    ax_a.set_ylabel(r"Peak growth rate $\sigma_{r,\max}$", fontsize=10, color=COLORS["blue"])
    ax_a.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax_a.set_ylim([-0.05, 0.85])
    ax_a.grid(True, alpha=0.25, linestyle=":")
    
    # Secondary y-axis for k_max
    ax_a2 = ax_a.twinx()
    ax_a2.scatter(dam_km[unstable_mask], k_max[unstable_mask], c=COLORS["vermilion"], s=30, alpha=0.8, marker="s", zorder=3, edgecolors="k", linewidth=0.5)
    ax_a2.plot(dam_km[unstable_mask], k_max[unstable_mask], color=COLORS["vermilion"], linewidth=0.8, alpha=0.4, zorder=2)
    ax_a2.set_ylabel(r"Preferred wavenumber $k_{\max} = 2\pi B / \lambda_{\max}$", fontsize=10, color=COLORS["vermilion"])
    ax_a2.tick_params(axis="y", labelcolor=COLORS["vermilion"])
    ax_a2.set_ylim(0.0, 12.0)
    
    ax_a.set_xlabel("Distance from Xiaolangdi Dam (km)", fontsize=10)
    ax_a.set_xlim(dam_km.min() - 5, dam_km.max() + 5)

    # Annotate damped sections
    for km, sig in zip(dam_km[damped_mask], sigma_r_max[damped_mask]):
        ax_a.annotate(
            rf"$\beta < \beta_c$" f"\n({km:.1f} km)",
            xy=(km, sig),
            xytext=(km - 8, sig + 0.15),
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
            fontsize=7.5,
            color="0.3",
            ha="center",
        )

    ax_a.legend(loc="upper left", frameon=False, fontsize=8)

    # ──────────────────────────────────────────────────────────────────
    # Layout and save
    # ──────────────────────────────────────────────────────────────────
    fig.tight_layout()
    save_fig(fig, "fig07_spatial_diagnostics")
    plt.close(fig)
    print("Figure 7 (spatial diagnostics) created successfully.")


if __name__ == "__main__":
    plot_fig07_spatial_diagnostics()
