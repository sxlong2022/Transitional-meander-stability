"""Example 03: Compute and Plot Wavenumber Dispersion Curves sigma_r(k).

Demonstrates:
1. Solving the planar 2D SWE--Exner eigenvalue problem over a continuous wavenumber array k.
2. Comparing early wide transitional river conditions (beta=250, year 2000) with recent narrowed conditions (beta=130, year 2019).
3. Exporting a publication-ready vector figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.stability.solve_bar_stability import solve_bar_stability
from publication_figures.figure_utils import setup_style, COLORS, DOUBLE_COL_WIDTH, save_fig


def compute_dispersion_curve(
    beta: float,
    Cf: float,
    Fr: float,
    theta: float,
    k_array: np.ndarray,
    N_cheb: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute peak growth rate sigma_r and migration celerity c_migr across k_array."""
    sigma_r = np.zeros_like(k_array)
    c_migr = np.zeros_like(k_array)

    for i, k_val in enumerate(k_array):
        evs, _ = solve_bar_stability(
            beta=beta,
            Cf=Cf,
            Fr=Fr,
            theta=theta,
            k_wavenumber=k_val,
            nu=0.0,
            mode_m=1,
            N_cheb=N_cheb,
        )
        if len(evs) > 0:
            sigma_r[i] = np.real(evs[0])
            sigma_i = np.imag(evs[0])
            c_migr[i] = -sigma_i / k_val if k_val > 0 else 0.0
        else:
            sigma_r[i] = np.nan
            c_migr[i] = np.nan

    return sigma_r, c_migr


def main() -> None:
    print("=" * 72)
    print("EXAMPLE 03: COMPUTING AND PLOTTING WAVENUMBER DISPERSION CURVES")
    print("=" * 72)

    setup_style()

    k_arr = np.linspace(0.1, 10.0, 100)

    # 1. Early Condition (2000): beta=253.2, Cf=0.0007, Fr=0.45, theta=0.35
    print("Computing early condition dispersion curve (beta=253.2, Cf=0.0007)...")
    sigma_early, c_early = compute_dispersion_curve(
        beta=253.2, Cf=0.0007, Fr=0.45, theta=0.35, k_array=k_arr, N_cheb=36
    )

    # 2. Recent Condition (2019): beta=132.8, Cf=0.0020, Fr=0.25, theta=0.55
    print("Computing recent condition dispersion curve (beta=132.8, Cf=0.0020)...")
    sigma_recent, c_recent = compute_dispersion_curve(
        beta=132.8, Cf=0.0020, Fr=0.25, theta=0.55, k_array=k_arr, N_cheb=36
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.2))

    # Panel (a): Temporal growth rate sigma_r(k)
    ax1.plot(k_arr, sigma_early, "-", color=COLORS["vermilion"], lw=1.8, label="Early (2000, $\\beta=253.2$)")
    ax1.plot(k_arr, sigma_recent, "-", color=COLORS["blue"], lw=1.8, label="Recent (2019, $\\beta=132.8$)")

    # Mark peaks
    idx_e = np.nanargmax(sigma_early)
    idx_r = np.nanargmax(sigma_recent)
    ax1.plot(k_arr[idx_e], sigma_early[idx_e], "o", color=COLORS["vermilion"], ms=6)
    ax1.plot(k_arr[idx_r], sigma_recent[idx_r], "s", color=COLORS["blue"], ms=6)

    ax1.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax1.set_xlabel("Dimensionless Wavenumber $k = 2\\pi B / \\lambda$")
    ax1.set_ylabel("Exponential Growth Rate $\\sigma_r$")
    ax1.set_title("(a) Instability Growth Rate Spectrum")
    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(True, ls=":", alpha=0.4)

    # Panel (b): Downstream migration celerity c_migr(k)
    ax2.plot(k_arr, c_early, "--", color=COLORS["vermilion"], lw=1.8, label="Early (2000)")
    ax2.plot(k_arr, c_recent, "--", color=COLORS["blue"], lw=1.8, label="Recent (2019)")
    ax2.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax2.set_xlabel("Dimensionless Wavenumber $k = 2\\pi B / \\lambda$")
    ax2.set_ylabel("Migration Celerity $c_{\\mathrm{migr}} = -\\sigma_i / k$")
    ax2.set_title("(b) Downstream Bar Migration Celerity")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    save_fig(fig, "example_dispersion_comparison")
    print(f"\nFigure saved to {PROJECT_ROOT / 'publication_figures' / 'output' / 'example_dispersion_comparison.pdf'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
