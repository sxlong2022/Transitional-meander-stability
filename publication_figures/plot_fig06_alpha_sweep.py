"""Publication figure: σ_r(k) dispersion curves for early (2000) vs recent (2019) years.

Script generates Figure 06 showing planar 2D SWE-Exner wavenumber dispersion curves,
demonstrating smooth interior maxima, high-wavenumber cutoffs, and temporal shifts.

Run via:
    python publication_figures/plot_fig06_alpha_sweep.py
"""
from __future__ import annotations

# stdlib
import sys
from pathlib import Path

# third-party
import matplotlib.pyplot as plt
import numpy as np

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
from src.stability.solve_bar_stability import solve_bar_stability


def plot_alpha_sweep() -> None:
    """Generate k-sweep dispersion curves for early (2000) vs recent (2019)."""
    setup_style()

    # Representative years: early (2000) and recent (2019)
    # 2000: beta=252.6, Fr=0.480, Cf=0.00050, theta=1.4, H=2.11m, B=532.4m
    early_year = 2000
    early_params = {"beta": 252.6, "Fr": 0.480, "Cf": 0.00050, "theta": 1.4, "theta_c": 0.047, "H": 2.11, "B": 532.4}

    # 2019: beta=131.6, Fr=0.241, Cf=0.00200, theta=4.3, H=4.96m, B=652.4m
    recent_year = 2019
    recent_params = {"beta": 131.6, "Fr": 0.241, "Cf": 0.00200, "theta": 4.3, "theta_c": 0.047, "H": 4.96, "B": 652.4}

    k_arr = np.linspace(0.1, 15.0, 150)

    # Compute dispersion curves
    early_sigma = []
    for k in k_arr:
        eigvals, _ = solve_bar_stability(
            beta=early_params["beta"],
            Cf=early_params["Cf"],
            Fr=early_params["Fr"],
            theta=early_params["theta"],
            theta_c=early_params["theta_c"],
            k_wavenumber=k,
            N_cheb=36,
        )
        if len(eigvals) > 0:
            early_sigma.append(eigvals[0].real)
        else:
            early_sigma.append(np.nan)

    recent_sigma = []
    for k in k_arr:
        eigvals, _ = solve_bar_stability(
            beta=recent_params["beta"],
            Cf=recent_params["Cf"],
            Fr=recent_params["Fr"],
            theta=recent_params["theta"],
            theta_c=recent_params["theta_c"],
            k_wavenumber=k,
            N_cheb=36,
        )
        if len(eigvals) > 0:
            recent_sigma.append(eigvals[0].real)
        else:
            recent_sigma.append(np.nan)

    early_sigma = np.array(early_sigma)
    recent_sigma = np.array(recent_sigma)

    # Peak values
    idx_early_max = np.argmax(early_sigma)
    k_early_max = k_arr[idx_early_max]
    sig_early_max = early_sigma[idx_early_max]
    lambda_early = 2 * np.pi * early_params["B"] / k_early_max

    idx_recent_max = np.argmax(recent_sigma)
    k_recent_max = k_arr[idx_recent_max]
    sig_recent_max = recent_sigma[idx_recent_max]
    lambda_recent = 2 * np.pi * recent_params["B"] / k_recent_max

    # Create figure
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.8))

    # Plot dispersion curves
    ax.plot(
        k_arr,
        early_sigma,
        linewidth=1.6,
        color=COLORS["orange"],
        label=rf"Year {early_year} ($\beta={early_params['beta']:.0f}$, $k_{{\max}}={k_early_max:.2f}$, $\lambda_{{\max}} \approx {lambda_early:.0f}\,$m)",
        zorder=3,
    )
    ax.scatter([k_early_max], [sig_early_max], color=COLORS["orange"], edgecolors="k", s=50, zorder=5)
    ax.vlines(k_early_max, ymin=-0.06, ymax=sig_early_max, colors=COLORS["orange"], linestyles=":", linewidth=1.0, alpha=0.7)

    ax.plot(
        k_arr,
        recent_sigma,
        linewidth=1.6,
        color=COLORS["blue"],
        label=rf"Year {recent_year} ($\beta={recent_params['beta']:.0f}$, $k_{{\max}}={k_recent_max:.2f}$, $\lambda_{{\max}} \approx {lambda_recent:.0f}\,$m)",
        zorder=3,
    )
    ax.scatter([k_recent_max], [sig_recent_max], color=COLORS["blue"], edgecolors="k", s=50, zorder=5)
    ax.vlines(k_recent_max, ymin=-0.06, ymax=sig_recent_max, colors=COLORS["blue"], linestyles=":", linewidth=1.0, alpha=0.7)

    # Stability neutral line
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.9, alpha=0.6, label=r"Neutral threshold ($\sigma_r = 0$)")

    # Labels and formatting
    ax.set_xlabel(r"Width-scaled wavenumber $k = 2\pi B / \lambda$ (non-dimensional)")
    ax.set_ylabel(r"Exponential growth rate $\sigma_r$ (non-dimensional)")
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.06, 0.35)
    ax.grid(True, alpha=0.3, linestyle=":")

    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout()
    save_fig(fig, "fig06_alpha_sweep")
    plt.close(fig)


if __name__ == "__main__":
    plot_alpha_sweep()
