"""Publication figure: Cf–Fr stability phase diagram with field & lab data (Fig 10).

Contour map of planar 2D SWE--Exner growth rate sigma_r,max in (Cf, Fr) parameter space,
overlaid with:
  - Yellow River Gaocun–Sunkou temporal points (2000–2019)
  - Yellow River 2016 spatial points (26 cross-sections)
  - Laboratory comparison cases (Termini S1/S2, Van Dijk, Braudrick)

Run via:
    python publication_figures/plot_fig10_phase_diagram.py
"""
from __future__ import annotations

# stdlib
import sys
from pathlib import Path

# third-party
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


# ── Data loaders ──────────────────────────────────────────────────────


def load_phase_diagram_csv(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load sweep CSV → (cf_arr, fr_arr, omega_i_grid).

    CSV has columns: Cf, Fr, omega_i_max, alpha_crit
    Returns unique 1-D arrays and reshaped 2-D grid.
    """
    cf_list, fr_list, omega_list = [], [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cf_list.append(float(row["Cf"]))
            fr_list.append(float(row["Fr"]))
            omega_list.append(float(row["omega_i_max"]))

    cf_unique = np.array(sorted(set(cf_list)))
    fr_unique = np.array(sorted(set(fr_list)))
    n_cf, n_fr = len(cf_unique), len(fr_unique)

    omega_grid = np.full((n_cf, n_fr), np.nan)
    cf_idx = {v: i for i, v in enumerate(cf_unique)}
    fr_idx = {v: j for j, v in enumerate(fr_unique)}
    for c, f, o in zip(cf_list, fr_list, omega_list):
        omega_grid[cf_idx[c], fr_idx[f]] = o

    return cf_unique, fr_unique, omega_grid


def load_timeseries_csv(csv_path: Path) -> dict:
    """Load hydraulic_params_timeseries.csv → dict of arrays."""
    years, frs, cfs = [], [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr_val = row.get("Fr", "")
            cf_val = row.get("Cf_energy", "")
            if fr_val and cf_val:
                years.append(int(row["year"]))
                frs.append(float(fr_val))
                cfs.append(float(cf_val))
    return {
        "year": np.array(years),
        "Fr": np.array(frs),
        "Cf": np.array(cfs),
    }


def load_spatial_csv(csv_path: Path) -> dict:
    """Load hydraulic_params_spatial_2016.csv → dict of arrays."""
    frs, cfs = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frs.append(float(row["Fr"]))
            cfs.append(float(row["Cf_energy"]))
    return {"Fr": np.array(frs), "Cf": np.array(cfs)}


# ── Literature lab cases ──────────────────────────────────────────────

LAB_CASES = {
    "Termini S1": {"Fr": 0.73, "Cf": 0.0050, "beta": 16.67},
    "Termini S2": {"Fr": 0.90, "Cf": 0.0042, "beta": 9.09},
    "Van Dijk":   {"Fr": 0.58, "Cf": 0.017,  "beta": 20.0},
    "Braudrick":  {"Fr": 0.55, "Cf": 0.015,  "beta": 21.1},
}

# ── Literature field cases (bankfull estimates) ──────────────────────
# Cf = g H S / U²  (energy-gradient definition, consistent with this study)
# Fr = U / sqrt(g H)

FIELD_CASES = {
    "Mississippi (Vicksburg)": {
        "Fr": 0.14, "Cf": 0.0021, "beta": 29.4,
        "source": "Soar et al. (2005)",
    },
    "Brahmaputra (Bahadurabad)": {
        "Fr": 0.25, "Cf": 0.0011, "beta": 200,
        "source": "Coleman (1969)",
    },
    "Waal/Rhine (NL)": {
        "Fr": 0.14, "Cf": 0.0054, "beta": 19,
        "source": "Domhof et al. (2018)",
    },
}


# ── Main plotting function ───────────────────────────────────────────


def plot_phase_diagram() -> None:
    """Generate Cf–Fr phase diagram with c_i contours and data overlays."""
    setup_style()

    # ── Load sweep data ──────────────────────────────────────────
    sweep_csv = RESULTS_DIR / "phase_diagram" / "phase_diagram_omega_i.csv"
    if not sweep_csv.exists():
        raise FileNotFoundError(
            f"Phase diagram sweep CSV not found: {sweep_csv}\n"
            "Run: python -m src.stability.run_phase_diagram"
        )

    cf_arr, fr_arr, omega_grid = load_phase_diagram_csv(sweep_csv)

    # Pad one extra row at Cf=0.025 so contourf covers the full y-range
    # (avoids white gap above cf_arr.max when ylim extends to show lab points)
    cf_pad = 0.025
    cf_arr = np.append(cf_arr, cf_pad)
    omega_grid = np.vstack([omega_grid, omega_grid[-1, :]])  # extrapolate top row

    # Meshgrid for contour (Cf on y-axis, Fr on x-axis)
    FR, CF = np.meshgrid(fr_arr, cf_arr)

    # ── Load field data ──────────────────────────────────────────
    ts = load_timeseries_csv(RESULTS_DIR / "hydraulic_params_timeseries.csv")
    sp = load_spatial_csv(RESULTS_DIR / "hydraulic_params_spatial_2016.csv")

    # ── Create figure ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 0.8, 6.0))

    # ── Filled contours (σ_r,max) ───────────────────────────────
    # All σ_r > 0 in the sweep domain (universal field instability),
    # so use a sequential colormap for intensity gradient.
    levels = np.linspace(0.0, 1.4, 36)
    omega_display = np.clip(omega_grid, 0.0, 1.4)

    # Sequential colormap: light → dark = weak → strong instability
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0.0, vmax=1.4)

    cf_contour = ax.contourf(
        FR, CF, omega_display,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )

    # Contour lines for key growth-rate isolines
    cs_lines = ax.contour(
        FR, CF, omega_grid,
        levels=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        colors="0.35",
        linewidths=0.8,
        linestyles="--",
    )
    clabel_texts = ax.clabel(
        cs_lines, fmt=r"$\sigma_{r,\max}=%g$", fontsize=7, inline=True,
        inline_spacing=3,
    )
    bbox_props = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75,
                      edgecolor="none")
    for txt in clabel_texts:
        txt.set_bbox(bbox_props)

    # ── Colorbar ─────────────────────────────────────────────────
    cbar = fig.colorbar(cf_contour, ax=ax, location="top", shrink=0.75, pad=0.03)
    cbar.set_label(r"Peak growth rate $\sigma_{r,\max}$ ($\beta = 130$)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # ── Overlay: Yellow River temporal (2000–2019) ───────────────
    # Sort by year to draw the trajectory connecting line
    sort_idx = np.argsort(ts["year"])
    
    # Add arrow heads to the trajectory line to show direction
    for i in range(len(sort_idx) - 1):
        x1, y1 = ts["Fr"][sort_idx[i]], ts["Cf"][sort_idx[i]]
        x2, y2 = ts["Fr"][sort_idx[i+1]], ts["Cf"][sort_idx[i+1]]
        ax.annotate(
            "",
            xy=(x2, y2), xycoords='data',
            xytext=(x1, y1), textcoords='data',
            arrowprops=dict(
                arrowstyle="-|>", color="0.2", alpha=0.6,
                shrinkA=6, shrinkB=6, patchA=None, patchB=None,
                connectionstyle="arc3,rad=0.15"
            ),
            zorder=8,
        )

    sc_temp = ax.scatter(
        ts["Fr"], ts["Cf"],
        c=ts["year"],
        cmap="viridis",
        s=24,
        edgecolors="k",
        linewidths=1.0,
        zorder=10,
        marker="o",
        label="Gaocun–Sunkou (temporal)",
    )

    # Add a second colorbar for the years
    from matplotlib.ticker import MaxNLocator
    cbar_yr = fig.colorbar(sc_temp, ax=ax, location="bottom", shrink=0.5, pad=0.12)
    cbar_yr.set_label("Year", fontsize=10)
    cbar_yr.ax.tick_params(labelsize=8)
    cbar_yr.locator = MaxNLocator(integer=True, nbins=5)
    cbar_yr.update_ticks()

    # ── Overlay: Yellow River spatial (2016) ─────────────────────
    ax.scatter(
        sp["Fr"], sp["Cf"],
        color="white",
        s=24,
        edgecolors="k",
        linewidths=0.8,
        zorder=9,
        marker="s",
        alpha=0.9,
        label="Gaocun–Sunkou 2016 (spatial)",
    )

    # ── Overlay: Lab cases ───────────────────────────────────────
    lab_markers = ["D", "D", "^", "v"]
    lab_colors = [COLORS["blue"], COLORS["skyblue"],
                  COLORS["purple"], COLORS["green"]]
    for (name, params), mkr, clr in zip(LAB_CASES.items(), lab_markers, lab_colors):
        ax.scatter(
            params["Fr"], params["Cf"],
            color=clr,
            s=24,
            edgecolors="k",
            linewidths=1.0,
            zorder=11,
            marker=mkr,
            label=name,
        )

    # ── Overlay: Other natural rivers ────────────────────────
    field_markers = ["p", "h", "H"]  # pentagon, hexagon1, hexagon2
    field_colors = [COLORS["vermilion"], COLORS["orange"], COLORS["green"]]
    for (name, params), mkr, clr in zip(FIELD_CASES.items(), field_markers, field_colors):
        ax.scatter(
            params["Fr"], params["Cf"],
            color=clr,
            s=40,
            edgecolors="k",
            linewidths=1.0,
            zorder=11,
            marker=mkr,
            label=name,
        )
    # ── Formatting ───────────────────────────────────────────────
    ax.set_xlabel(r"Froude number $\mathrm{Fr}$")
    ax.set_ylabel(r"Friction coefficient $C_f$")
    ax.set_yscale("log")
    ax.set_xlim(fr_arr.min(), fr_arr.max())
    ax.set_ylim(cf_arr.min(), max(cf_arr.max(), 0.025))

    # ── Grouped legend (manual) ───────────────────────────────
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.legend as mlegend

    # Build legend with section titles
    handles, labels = ax.get_legend_handles_labels()

    # Reorder: YR temporal, YR spatial, then blank, field rivers, blank, lab cases
    # Current order: temporal(0), spatial(1), lab×4(2-5), field×3(6-8)
    from matplotlib.lines import Line2D
    blank = Line2D([], [], marker='None', linestyle='None', label='')
    title_yr = Line2D([], [], marker='None', linestyle='None', label='')
    title_field = Line2D([], [], marker='None', linestyle='None', label='')
    title_lab = Line2D([], [], marker='None', linestyle='None', label='')

    ordered_handles = [
        title_yr, handles[0], handles[1],
        title_field, handles[6], handles[7], handles[8],
        title_lab, handles[2], handles[3], handles[4], handles[5],
    ]
    ordered_labels = [
        r"$\bf{This\ study}$", labels[0], labels[1],
        r"$\bf{Natural\ rivers}$", labels[6], labels[7], labels[8],
        r"$\bf{Laboratory}$", labels[2], labels[3], labels[4], labels[5],
    ]

    leg = ax.legend(
        ordered_handles, ordered_labels,
        loc="lower right",
        fontsize=7,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.7",
        ncol=1,
        scatterpoints=1,
        markerscale=1.0,
        handletextpad=0.3,
        labelspacing=0.35,
    )

    fig.tight_layout()
    save_fig(fig, "fig10_phase_diagram")
    plt.close(fig)


if __name__ == "__main__":
    plot_phase_diagram()
