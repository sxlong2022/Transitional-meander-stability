"""Publication figure: Model-derived scaling of curvature-enhanced growth rates (Fig 09).

Demonstrates that the peak growth rate sigma_r_max under curvature perturbation
approximately collapses with the bend parameter product nu*beta = B/R across varying aspect ratios.
Data from curved_beta_nu_sweep.csv (8 beta values x 5 nu values = 40 2D SWE--Exner solutions).

Run via:
    python publication_figures/plot_fig09_nubeta_collapse.py
"""
from __future__ import annotations

# stdlib
import sys
from pathlib import Path

# third-party
import matplotlib.pyplot as plt
import numpy as np
import csv

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
        COLOR_CYCLE,
        SINGLE_COL_WIDTH,
        RESULTS_DIR,
        save_fig,
        setup_style,
    )
except ImportError:
    from figure_utils import (
        COLORS,
        COLOR_CYCLE,
        SINGLE_COL_WIDTH,
        RESULTS_DIR,
        save_fig,
        setup_style,
    )


# ── Markers for each β series ────────────────────────────────────────
BETA_MARKERS = {
    5:   "o",
    10:  "s",
    20:  "^",
    50:  "D",
    80:  "v",
    130: "P",
    200: "X",
    300: "*",
}


def load_nubeta_sweep(csv_path: Path) -> dict:
    """Load curved_beta_nu_sweep.csv → dict keyed by beta.

    Returns
    -------
    data : {beta: {"nu": [...], "nubeta": [...], "omega_i_max": [...]}}
    """
    data: dict[int, dict[str, list[float]]] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            beta = int(float(row["beta"]))
            nu = float(row["nu_curvature"])
            omega = float(row["omega_i_max"])
            if beta not in data:
                data[beta] = {"nu": [], "nubeta": [], "omega_i_max": []}
            data[beta]["nu"].append(nu)
            data[beta]["nubeta"].append(nu * beta)
            data[beta]["omega_i_max"].append(omega)
    # Convert to arrays
    for beta in data:
        for key in data[beta]:
            data[beta][key] = np.array(data[beta][key])
    return data


def plot_nubeta_collapse() -> None:
    """Generate νβ product collapse figure."""
    setup_style()

    # ── Load data ──────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "beta_sweep" / "curved_beta_nu_sweep.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Curved beta-nu sweep CSV not found: {csv_path}\n"
            "Run: python -m src.stability.run_beta_sweep"
        )
    data = load_nubeta_sweep(csv_path)

    # ── Create figure ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 1.6, SINGLE_COL_WIDTH * 1.15))

    # Plot each β series
    # β < β_c (subcritical, ω_i,straight < 0): grey dashed lines
    # β ≥ β_c (bar-unstable, ω_i,straight > 0): colour solid lines
    SUBCRITICAL_BETAS = {5, 10, 20}  # omega_straight < 0 for these
    betas_sorted = sorted(data.keys())
    color_idx = 0  # separate counter so subcritical lines don't consume colour slots
    for i, beta in enumerate(betas_sorted):
        d = data[beta]
        marker = BETA_MARKERS.get(beta, "o")

        # Calculate relative growth rate enhancement percentage E (%)
        idx_straight = np.argmin(d["nu"])
        omega_straight = d["omega_i_max"][idx_straight]
        if np.isnan(omega_straight) or omega_straight == 0.0:
            E = np.zeros_like(d["omega_i_max"])
        else:
            E = (d["omega_i_max"] - omega_straight) / abs(omega_straight) * 100.0

        if beta in SUBCRITICAL_BETAS:
            # Subcritical: grey dashed, no marker fill, label with dagger note
            ax.plot(
                d["nubeta"], E,
                marker=marker,
                color="0.65",
                markeredgecolor="0.5",
                markeredgewidth=0.4,
                markerfacecolor="none",
                markersize=5.0,
                linestyle="--",
                linewidth=0.7,
                alpha=0.75,
                label=rf"$\beta = {beta}$" + r"$^{\dagger}$",
                zorder=3,
            )
        else:
            color = COLOR_CYCLE[color_idx % len(COLOR_CYCLE)]
            color_idx += 1
            ax.plot(
                d["nubeta"], E,
                marker=marker,
                color=color,
                markeredgecolor="k",
                markeredgewidth=0.4,
                markersize=5.5,
                linestyle="-",
                linewidth=0.8,
                alpha=0.85,
                label=rf"$\beta = {beta}$",
                zorder=5 + color_idx,
            )

    # ── Annotate field regime ──────────────────────────────────────
    # νβ ≈ 0.4 for Gaocun–Sunkou
    ax.axvline(
        0.4, color="0.4", linestyle=":", linewidth=0.8, zorder=2,
    )
    ax.text(
        0.38, 30.0, r"field regime" "\n" r"$\nu\beta \approx 0.4$",
        fontsize=7.5, color="0.3", ha="right", va="top",
    )

    # ── Formatting ─────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_ylim(-2, 50)
    ax.set_xlabel(r"Bend parameter $\nu\beta = B/R$")
    ax.set_ylabel(r"Relative change in growth rate $E$ (%)")

    ax.legend(
        loc="upper left",
        fontsize=6.5,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.7",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
        labelspacing=0.35,
    )

    # Add footnote explaining dagger symbol for subcritical series
    ax.annotate(
        r"$^{\dagger}$subcritical ($\sigma_{r,\max,\mathrm{straight}} < 0$; bar-stable)",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=6.0, color="0.45", va="bottom", ha="left",
    )

    fig.tight_layout()
    save_fig(fig, "fig09_nubeta_collapse")
    plt.close(fig)


if __name__ == "__main__":
    plot_nubeta_collapse()
