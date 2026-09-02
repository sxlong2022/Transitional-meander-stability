"""Figure 1 — Study area: (a) Google Earth overview, (b) JRC water occurrence.

Usage
-----
    python publication_figures/plot_fig01_study_area.py

Prerequisites
-------------
    publication_figures/output/fig01a_google_earth.jpg   (exported from Google Earth)
    data/GIS/Gaocun-Sunkou/GaocunSunkou_JRC_occurrence.tif
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
_pkg_dir = Path(__file__).resolve().parent
_root_dir = _pkg_dir.parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from publication_figures.figure_utils import setup_style, COLORS, save_fig, DOUBLE_COL_WIDTH
except ImportError:
    from figure_utils import setup_style, COLORS, save_fig, DOUBLE_COL_WIDTH

def plot_fig01(ge_path: Path, tif_path: Path, output_name: str) -> None:
    """Generate combined Figure 1 with two vertically stacked panels."""
    setup_style()

    # ── Load images ──────────────────────────────────────────
    ge_img = mpimg.imread(ge_path)  # Google Earth JPG → (H, W, 3)
    ge_aspect = ge_img.shape[1] / ge_img.shape[0]  # W / H

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        nodata = src.nodata
        mask = (data == 0) | (data == nodata) if nodata is not None else (data == 0)
        data_masked = np.ma.masked_array(data, mask=mask)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        jrc_aspect = src.width / src.height

    # ── Figure layout ────────────────────────────────────────
    fig_w = DOUBLE_COL_WIDTH  # 7.0 inches (Elsevier double-column)
    # Panel (a) height determined by Google Earth image aspect ratio
    ha = fig_w / ge_aspect
    # Panel (b) height determined by JRC TIF aspect ratio, add small extra for colorbar
    hb = fig_w / jrc_aspect
    vspace = 0.15  # inches between panels
    fig_h = ha + hb + vspace

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, 1, figure=fig,
                           height_ratios=[ha, hb],
                           hspace=vspace / fig_h)

    # ── Panel (a): Google Earth ──────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    ax_a.imshow(ge_img)
    ax_a.set_axis_off()
    ax_a.text(0.02, 0.95, '(a)', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top',
              color='white',
              bbox=dict(facecolor='black', alpha=0.5,
                        edgecolor='none', pad=3))

    # ── Panel (b): JRC water occurrence ──────────────────────
    ax_b = fig.add_subplot(gs[1])
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='white', alpha=0)
    im = ax_b.imshow(data_masked, cmap=cmap, extent=extent,
                     vmin=0, vmax=100, aspect='equal')
    ax_b.set_xlabel('Longitude ($^\\circ$E)')
    ax_b.set_ylabel('Latitude ($^\\circ$N)')
    ax_b.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax_b.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax_b.grid(True, linestyle='--', alpha=0.3, color='gray')

    # Colorbar
    divider = make_axes_locatable(ax_b)
    cax = divider.append_axes("right", size="2%", pad=0.08)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Water occurrence frequency (%)', rotation=90, labelpad=5)

    # Panel label
    ax_b.text(0.02, 0.95, '(b)', transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top',
              bbox=dict(facecolor='white', alpha=0.7,
                        edgecolor='none', pad=3))

    # ── Save ─────────────────────────────────────────────────
    save_fig(fig, output_name)
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(__file__).resolve().parent / "output"

    ge_path = output_dir / "fig01a_google_earth.jpg"
    tif_path = (project_root / "data" / "GIS" / "Gaocun-Sunkou"
                / "GaocunSunkou_JRC_occurrence.tif")

    if not ge_path.exists():
        print(f"ERROR: Google Earth image not found at {ge_path}")
        print("Please export from Google Earth and place in output/.")
        sys.exit(1)
    if not tif_path.exists():
        print(f"ERROR: JRC TIF not found at {tif_path}")
        sys.exit(1)

    print("Generating Figure 1 — Study area (combined)...")
    plot_fig01(ge_path, tif_path, "fig01_study_area")
    print("Done.")
