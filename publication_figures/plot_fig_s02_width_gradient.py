"""Plot Figure S2: Along-stream local width gradient |dB/ds| profiles and reach coverage CDF.

Panels:
  (a) Vertically stacked along-channel local width gradient profiles for representative
      morphological epochs (2001 unconfined wandering, 2016 transitional, 2021 regulated single-thread).
      Includes faint traces showing raw 100 m steps, bold curves showing 1.5 km macro-scale
      envelopes, and in-situ threshold labels for weak (|dB/ds| <= 0.10) and moderate (|dB/ds| <= 0.20)
      variations.
  (b) Cumulative reach coverage distribution functions (CDFs) demonstrating that 44%–54% of
      the reach satisfies |dB/ds| < 0.10 and 67%–78% satisfies |dB/ds| < 0.20 across all epochs.

Requires:
  Gaocun-Sunkou_{year}_trunk_0.csv in RESULTS_DIR / "trunks".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

setup_style()

trunk_dir = RESULTS_DIR / "trunks"
years = [2001, 2016, 2021]

all_data = {}
for yr in years:
    f = trunk_dir / f"Gaocun-Sunkou_{yr}_trunk_0.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    s = df['s_m'].values / 1000.0
    B = df['B_m'].values
    
    mask = np.isfinite(s) & np.isfinite(B)
    s, B = s[mask], B[mask]
    _, u_idx = np.unique(s, return_index=True)
    s, B = s[u_idx], B[u_idx]
    
    s_uni = np.arange(s[0], s[-1], 0.1) # 100m step
    B_uni = np.interp(s_uni, s, B)
    
    # Raw gradient
    dB_ds_raw = np.abs(np.gradient(B_uni, s_uni * 1000.0))
    # Savitzky-Golay smoothed width then gradient (window 1.5km = 15 points)
    B_sg = savgol_filter(B_uni, window_length=15, polyorder=2)
    dB_ds_sg = np.abs(np.gradient(B_sg, s_uni * 1000.0))
    
    all_data[yr] = {
        's': s_uni,
        'B': B_uni,
        'dB_raw': dB_ds_raw,
        'dB_sg': dB_ds_sg
    }

# Figure layout: 3-row stacked on left, CDF on right
fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, 4.2))
gs = fig.add_gridspec(3, 3, width_ratios=[1.4, 1.4, 1.1], hspace=0.18, wspace=0.32)

ax_2001 = fig.add_subplot(gs[0, 0:2])
ax_2016 = fig.add_subplot(gs[1, 0:2], sharex=ax_2001)
ax_2021 = fig.add_subplot(gs[2, 0:2], sharex=ax_2001)
ax_cdf  = fig.add_subplot(gs[:, 2])

axes_ts = [
    (ax_2001, 2001, r"2001 (Wandering, $\beta \approx 306$)", COLORS["vermilion"]),
    (ax_2016, 2016, r"2016 (Transitional, $\beta \approx 132$)", COLORS["blue"]),
    (ax_2021, 2021, r"2021 (Regulated, $\beta \approx 124$)", COLORS["green"])
]

# Panel (a): 3-row stacked profiles
for idx, (ax, yr, lbl, col) in enumerate(axes_ts):
    d = all_data[yr]
    
    # Threshold background and guideline
    ax.axhspan(0.0, 0.10, color='#EDF4FB', zorder=0)
    ax.axhline(0.10, color='#666666', ls='--', lw=0.75, zorder=2)
    ax.axhline(0.20, color='#999999', ls=':', lw=0.75, zorder=2)
    
    # Raw trace (faint) + Macro-envelope (bold)
    ax.plot(d['s'], d['dB_raw'], color=col, alpha=0.15, lw=0.5, zorder=1)
    ax.plot(d['s'], d['dB_sg'], color=col, alpha=0.95, lw=1.25, zorder=3)
    
    ax.set_xlim(0, 140)
    ax.set_ylim(-0.01, 0.52)
    ax.set_yticks([0.0, 0.10, 0.20, 0.40])
    ax.tick_params(labelsize=8.0)
    
    # Subplot year tag
    ax.text(0.015, 0.72, lbl, transform=ax.transAxes, fontsize=8.0, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.25', facecolor='white', alpha=0.90, edgecolor='#CCCCCC', lw=0.5), zorder=4)
    
    if idx < 2:
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel(r"Along-stream distance $s$ (km) [Gaocun $\rightarrow$ Sunkou]", fontsize=9.0)

# Panel (a) title and geographic orientation
ax_2001.set_title(r"(a) Along-stream local width gradient $|\mathrm{d}B/\mathrm{d}s|$ profiles", loc='left', fontsize=9.5, fontweight='bold', pad=8)

# Set shared y-axis label directly on middle subplot (ax_2016) with proper labelpad
ax_2016.set_ylabel(r"Local width gradient $|\mathrm{d}B/\mathrm{d}s|$", fontsize=9.0, labelpad=4)

# Plain text labels next to the dashed and dotted lines (without variable values)
ax_2001.text(116, 0.108, "Weak variation", fontsize=7.5, color='#333333', va='bottom', ha='left',
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.88, edgecolor='none'), zorder=5)
ax_2001.text(116, 0.208, "Moderate variation", fontsize=7.5, color='#555555', va='bottom', ha='left',
             bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.88, edgecolor='none'), zorder=5)

# Panel (b): Cumulative Distribution Functions (CDFs)
for ax, yr, lbl, col in axes_ts:
    d = all_data[yr]
    vals = np.sort(d['dB_sg'])
    cdf = np.arange(1, len(vals) + 1) / len(vals) * 100.0
    ax_cdf.plot(vals, cdf, label=f"{yr}", color=col, lw=1.4, zorder=3)

ax_cdf.axvspan(0.0, 0.10, color='#EDF4FB', zorder=0)
ax_cdf.axhline(50, color='#AAAAAA', ls=':', lw=0.6, zorder=1)
ax_cdf.axhline(75, color='#AAAAAA', ls=':', lw=0.6, zorder=1)

ax_cdf.axvline(0.10, color='#666666', ls='--', lw=0.75, zorder=2)
ax_cdf.axvline(0.20, color='#999999', ls=':', lw=0.75, zorder=2)

ax_cdf.text(0.43, 51.5, "50%", color='#555555', fontsize=7.0, ha='right')
ax_cdf.text(0.43, 76.5, "75%", color='#555555', fontsize=7.0, ha='right')

ax_cdf.text(0.05, 14, "Weak-variation\nzone ($< 0.10$)", color='#1A4E7A', fontsize=7.0, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#EDF4FB', alpha=0.9, edgecolor='none'))

ax_cdf.set_xlim(0, 0.45)
ax_cdf.set_ylim(0, 100)
ax_cdf.set_xlabel(r"Local gradient $|\mathrm{d}B/\mathrm{d}s|$", fontsize=9.0)
ax_cdf.set_ylabel(r"Cumulative reach fraction (%)", fontsize=9.0)
ax_cdf.set_title(r"(b) Reach coverage CDF", loc='left', fontsize=9.5, fontweight='bold', pad=8)
ax_cdf.legend(loc='lower right', fontsize=7.5, frameon=True, framealpha=0.92, edgecolor='#CCCCCC')
ax_cdf.tick_params(labelsize=8.0)

save_fig(fig, "fig_s02_width_gradient")
print("Figure S2 (Width Gradient) generated and saved successfully!")
