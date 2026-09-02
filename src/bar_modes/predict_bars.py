"""Bar mode prediction and diagnostics module.

Integrates the Crosato-Mosselman (2009) empirical formula, 2D SWE--Exner unstable wavenumber windows,
and reach-scale width gradient sigma_width diagnostics.

Typical usage::

    from src.bar_modes import predict_bar_mode_cm, compute_unstable_window
    m = predict_bar_mode_cm(beta=132.0, Cf=0.00176)
    window = compute_unstable_window(beta=132.0, Fr=0.257, Cf=0.00176)
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── sys.path hack for standalone execution ────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── UTF-8 stdout (Windows GBK workaround) ────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from src.diagnostics.diagnose_stability import (  # noqa: E402
    AlphaSweepResult,
    diagnose_stability,
)


# =====================================================================
# Data Classes
# =====================================================================

@dataclasses.dataclass
class BarDiagnostic:
    """Comprehensive bar mode diagnostic results.

Attributes
----------
    beta : float
        Half width-to-depth ratio B / (2H).
    Fr : float
        Froude number.
    Cf : float
        Friction coefficient.
    m_cm : int
        Crosato-Mosselman predicted bar mode number m.
    bar_regime : str
        Bar regime: single_row / multi_row / braided.
    alpha_crit : float
        OS 。
    omega_i_max : float
        OS 。
    lambda_crit_m : float
        OS （）； D_m。
    alpha_unstable_min : float
        。
    alpha_unstable_max : float
        。
    n_unstable_frac : float
        。
    lambda_unstable_min_m : float
        （）。
    lambda_unstable_max_m : float
        （）。
    """

    beta: float
    Fr: float
    Cf: float
    m_cm: int
    bar_regime: str
    alpha_crit: float
    omega_i_max: float
    lambda_crit_m: float
    alpha_unstable_min: float
    alpha_unstable_max: float
    n_unstable_frac: float
    lambda_unstable_min_m: float
    lambda_unstable_max_m: float


# =====================================================================
# Crosato-Mosselman (2009)
# =====================================================================

def predict_bar_mode_cm(
    beta: float,
    Cf: float,
    b_param: float = 1.7,
    sediment_exponent: float | None = None,
) -> int:
    """Crosato-Mosselman (2009) 。

    ::

        m = round( (2 * beta) / (b * pi) )

     b ≈ 1.7 (Crosato & Mosselman 2009, Table 1 )。
     sediment_exponent ::

        m = round( (beta / pi) * sqrt(Cf / epsilon) )

    
    ------
    beta : float
         B / H。
    Cf : float
        （）。
    b_param : float
         b， 1.7。
    sediment_exponent : float or None
         ε。 None 。

    
    ------
    int
         m（≥1）。

    
    ------
     β（ β > 100），m （/），
    。
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    if sediment_exponent is not None:
        if sediment_exponent <= 0:
            raise ValueError(
                f"sediment_exponent must be positive, got {sediment_exponent}"
            )
        m_raw = (beta / np.pi) * np.sqrt(Cf / sediment_exponent)
    else:
        m_raw = (2.0 * beta) / (b_param * np.pi)

    return max(1, int(round(m_raw)))


# =====================================================================
# OS
# =====================================================================

def compute_unstable_window(
    beta: float,
    Fr: float,
    Cf: float,
    nu_curvature: float = 0.0,
    alpha_range: tuple[float, float] = (0.01, 15.0),
    n_alpha: int = 200,
    D_m: float | None = None,
    **kwargs,
) -> AlphaSweepResult:
    """ OS （alpha-sweep ）。

    
    ------
    beta : float
        。
    Fr : float
        Froude number.
    Cf : float
        Friction coefficient.
    nu_curvature : float
         H/R， 0。
    alpha_range : tuple
        ， (0.01, 15.0)，。
    n_alpha : int
        ， 200（）。
    D_m : float or None
        （），。
    **kwargs
         diagnose_stability 。

    
    ------
    AlphaSweepResult
         omega_i(alpha) 。
    """
    result = diagnose_stability(
        beta=beta,
        Fr=Fr,
        Cf=Cf,
        nu_curvature=nu_curvature,
        alpha_range=alpha_range,
        n_alpha=n_alpha,
        D_m=D_m,
        return_alpha_sweep=True,
        **kwargs,
    )
    return result


# =====================================================================
# σ_width
# =====================================================================

def compute_sigma_width(
    s_m: np.ndarray,
    B_m: np.ndarray,
    smooth_window: int = 0,
) -> np.ndarray:
    """ σ_width = (1/B) dB/ds。

    
    ------
    s_m : np.ndarray
        （），。
    B_m : np.ndarray
        （）， s_m 。
    smooth_window : int
        （0 = ）。 > 0， B_m 
        。

    
    ------
    sigma_width : np.ndarray
        ，（ forward/backward diff）。

    
    ------
    σ_width > 0 ，σ_width < 0 。
    ，σ_width > 0 （≪ O(0.1)）
     bar–bend （γ_BA < 0），
    。 β  Cf。
    """
    if len(s_m) != len(B_m):
        raise ValueError(
            f"s_m and B_m must have same length, got {len(s_m)} vs {len(B_m)}"
        )
    if len(s_m) < 3:
        raise ValueError(f"Need at least 3 points, got {len(s_m)}")

    B = B_m.copy().astype(float)

    # Optional smoothing
    if smooth_window > 0:
        kernel = np.ones(smooth_window) / smooth_window
        B = np.convolve(B, kernel, mode="same")

    # Central difference for dB/ds (forward/backward at ends)
    ds = np.diff(s_m)
    dB = np.diff(B)

    # Guard against zero ds
    ds = np.where(ds == 0, np.nan, ds)

    dBds = np.empty_like(B)
    # Forward diff at start
    dBds[0] = dB[0] / ds[0]
    # Central diff in interior
    for i in range(1, len(B) - 1):
        dBds[i] = (B[i + 1] - B[i - 1]) / (s_m[i + 1] - s_m[i - 1])
    # Backward diff at end
    dBds[-1] = dB[-1] / ds[-1]

    # Normalize by B
    sigma_width = np.where(B > 0, dBds / B, np.nan)

    return sigma_width


def compute_sigma_width_stats(
    s_m: np.ndarray,
    B_m: np.ndarray,
    smooth_window: int = 5,
) -> dict:
    """ σ_width 。

    
    ------
    s_m : np.ndarray
        （）。
    B_m : np.ndarray
        （）。
    smooth_window : int
        ， 5 。

    
    ------
    dict
         mean, std, median, p10, p90, frac_positive, frac_in_cross_enh
        。
    """
    sigma = compute_sigma_width(s_m, B_m, smooth_window)
    valid = sigma[np.isfinite(sigma)]

    if len(valid) == 0:
        return {
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "p10": np.nan, "p90": np.nan,
            "frac_positive": np.nan, "frac_in_cross_enh": np.nan,
            "sigma_width": sigma,
        }

    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
        "frac_positive": float(np.sum(valid > 0) / len(valid)),
        # Fraction of points with small positive σ_width (gradual widening),
        # indicative of bar-bend cross-enhancement regime.
        # Threshold O(0.1) used as a conservative upper bound.
        "frac_in_cross_enh": float(
            np.sum((valid > 0) & (valid < 0.1)) / len(valid)
        ),
        "sigma_width": sigma,
    }


# =====================================================================
# Comprehensive Diagnostics
# =====================================================================

def diagnose_bar_regime(
    beta: float,
    Fr: float,
    Cf: float,
    nu_curvature: float = 0.0,
    D_m: float | None = None,
    B_m_phys: float | None = None,
    alpha_range: tuple[float, float] = (0.01, 15.0),
    n_alpha: int = 200,
    **kwargs,
) -> BarDiagnostic:
    """：Crosato-Mosselman + OS 。

    
    ------
    beta : float
         B / H。
    Fr : float
        Froude number.
    Cf : float
        Friction coefficient.
    nu_curvature : float
         H/R。
    D_m : float or None
        （）。
    B_m_phys : float or None
        （），。 None  beta * 2 * D_m 。
    alpha_range : tuple
        。
    n_alpha : int
        。
    **kwargs
         diagnose_stability 。

    
    ------
    BarDiagnostic
    """
    # 1. Crosato-Mosselman bar mode prediction
    m_cm = predict_bar_mode_cm(beta, Cf)

    # Bar regime classification
    if m_cm <= 1:
        bar_regime = "single_row"
    elif m_cm <= 3:
        bar_regime = "multi_row"
    else:
        bar_regime = "braided"

    # 2. OS unstable window
    asw = compute_unstable_window(
        beta=beta, Fr=Fr, Cf=Cf, nu_curvature=nu_curvature,
        alpha_range=alpha_range, n_alpha=n_alpha, D_m=D_m,
        **kwargs,
    )
    sr = asw.stability_result

    # 3. Convert wavelengths to meters
    #    λ(dimensional) = λ(non-dim) × D_m = (2π/α) × D_m
    #    But OS non-dimensionalization uses half-width h = D as length scale,
    #    so λ_m = (2π/α) × D_m
    #    Alternatively, using full width: λ_m = (2π/α) × (B/2) = (2π/α) × β × D_m
    #    The OS uses D (depth) as scale, so λ in OS = 2π/α × D
    #    Physical wavelength: λ_phys = λ_OS × D_m
    if D_m is not None:
        lambda_crit_m = (2.0 * np.pi / sr.alpha_crit * D_m
                         if sr.alpha_crit > 0 else np.inf)
        if np.isfinite(asw.alpha_unstable_min) and asw.alpha_unstable_min > 0:
            lambda_unstable_max_m = 2.0 * np.pi / asw.alpha_unstable_min * D_m
        else:
            lambda_unstable_max_m = np.inf
        if np.isfinite(asw.alpha_unstable_max) and asw.alpha_unstable_max > 0:
            lambda_unstable_min_m = 2.0 * np.pi / asw.alpha_unstable_max * D_m
        else:
            lambda_unstable_min_m = 0.0
    else:
        lambda_crit_m = np.nan
        lambda_unstable_min_m = np.nan
        lambda_unstable_max_m = np.nan

    return BarDiagnostic(
        beta=beta,
        Fr=Fr,
        Cf=Cf,
        m_cm=m_cm,
        bar_regime=bar_regime,
        alpha_crit=sr.alpha_crit,
        omega_i_max=sr.omega_i_max,
        lambda_crit_m=lambda_crit_m,
        alpha_unstable_min=asw.alpha_unstable_min,
        alpha_unstable_max=asw.alpha_unstable_max,
        n_unstable_frac=asw.n_unstable / len(asw.alpha_arr) if len(asw.alpha_arr) > 0 else 0.0,
        lambda_unstable_min_m=lambda_unstable_min_m,
        lambda_unstable_max_m=lambda_unstable_max_m,
    )




# =====================================================================
# σ_width （ trunk CSV）
# =====================================================================

def run_sigma_width_batch(
    trunk_dir: str | Path,
    pattern: str = "*_trunk_0.csv",
    smooth_window: int = 5,
    output_csv: str | Path | None = None,
) -> list[dict]:
    """   trunk  B(s)  σ_width 。

    
    ------
    trunk_dir : str or Path
        trunk CSV 。
    pattern : str
        ， "*_trunk_0.csv"。
    smooth_window : int
        。
    output_csv : str or Path or None
        ， CSV。

    
    ------
    list[dict]
        ， year, n_points, B_mean, σ_width 。
    """
    import csv as _csv
    import re

    trunk_dir = Path(trunk_dir)
    if not trunk_dir.exists():
        raise FileNotFoundError(f"Trunk directory not found: {trunk_dir}")

    files = sorted(trunk_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No trunk files matching '{pattern}' in {trunk_dir}"
        )

    all_results: list[dict] = []

    for fpath in files:
        # Extract year from filename like 'Gaocun-Sunkou_2016_trunk_0.csv'
        match = re.search(r'(\d{4})', fpath.stem)
        year = match.group(1) if match else fpath.stem

        # Read CSV
        s_list, B_list = [], []
        with open(fpath, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    s_val = float(row["s_m"])
                    B_val = float(row["B_m"])
                except (KeyError, ValueError):
                    continue
                s_list.append(s_val)
                B_list.append(B_val)

        if len(s_list) < 10:
            print(f"  [SKIP] {year}: only {len(s_list)} points")
            continue

        s_arr = np.array(s_list)
        B_arr = np.array(B_list)

        stats = compute_sigma_width_stats(s_arr, B_arr, smooth_window)
        row_out = {
            "year": year,
            "n_points": len(s_arr),
            "trunk_length_km": float(s_arr[-1] - s_arr[0]) / 1000.0,
            "B_mean_m": float(np.mean(B_arr)),
            "B_std_m": float(np.std(B_arr)),
            "sigma_mean": stats["mean"],
            "sigma_std": stats["std"],
            "sigma_median": stats["median"],
            "sigma_p10": stats["p10"],
            "sigma_p90": stats["p90"],
            "frac_positive": stats["frac_positive"],
            "frac_in_cross_enh": stats["frac_in_cross_enh"],
        }
        all_results.append(row_out)
        print(
            f"  {year}: n={len(s_arr):5d}, B_mean={row_out['B_mean_m']:.0f}m, "
            f"sigma_mean={row_out['sigma_mean']:.6f}, "
            f"frac_pos={row_out['frac_positive']:.1%}"
        )

    # Write CSV if requested
    if output_csv is not None and all_results:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_results[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Saved {len(all_results)} rows to {output_csv}")

    return all_results
# =====================================================================
# CLI demo
# =====================================================================

def main():
    """Demo: bar mode diagnostics for recent and early periods."""
    print("=" * 60)
    print("Bar Mode Diagnostic Module - Demo")
    print("=" * 60)

    # ── Crosato-Mosselman predictions ────────────────────────────
    print("\n--- Crosato-Mosselman bar mode predictions ---")
    for label, beta, Cf in [
        ("Recent (2011-2021)", 132.0, 0.00176),
        ("Early (2000-2003)", 253.0, 0.00070),
        ("Lab scale (Sub1/2)", 10.0, 0.01),
    ]:
        m = predict_bar_mode_cm(beta, Cf)
        print(f"  {label:25s}: beta={beta:6.1f}, Cf={Cf:.5f} -> m = {m}")

    # ── Full diagnostic for recent period ────────────────────────
    # D_m ≈ 1.5 m (typical bankfull depth for Gaocun-Sunkou)
    print("\n--- Full diagnostic: Recent period (curved OS) ---")
    diag = diagnose_bar_regime(
        beta=132.0, Fr=0.257, Cf=0.00176,
        nu_curvature=0.004, D_m=1.5,
    )
    print(f"  m_cm (C-M)     = {diag.m_cm}")
    print(f"  bar_regime     = {diag.bar_regime}")
    print(f"  alpha_crit     = {diag.alpha_crit:.4f}")
    print(f"  omega_i_max    = {diag.omega_i_max:.4f}")
    print(f"  lambda_crit    = {diag.lambda_crit_m:.1f} m")
    print(f"  unstable alpha = [{diag.alpha_unstable_min:.4f}, {diag.alpha_unstable_max:.4f}]")
    print(f"  unstable frac  = {diag.n_unstable_frac:.1%}")
    print(f"  unstable lambda= [{diag.lambda_unstable_min_m:.1f}, {diag.lambda_unstable_max_m:.1f}] m")

    # ── Full diagnostic for early period ─────────────────────────
    print("\n--- Full diagnostic: Early period (curved OS) ---")
    diag2 = diagnose_bar_regime(
        beta=253.0, Fr=0.448, Cf=0.00070,
        nu_curvature=0.004, D_m=0.8,
    )
    print(f"  m_cm (C-M)     = {diag2.m_cm}")
    print(f"  bar_regime     = {diag2.bar_regime}")
    print(f"  alpha_crit     = {diag2.alpha_crit:.4f}")
    print(f"  omega_i_max    = {diag2.omega_i_max:.4f}")
    print(f"  lambda_crit    = {diag2.lambda_crit_m:.1f} m")
    print(f"  unstable alpha = [{diag2.alpha_unstable_min:.4f}, {diag2.alpha_unstable_max:.4f}]")
    print(f"  unstable frac  = {diag2.n_unstable_frac:.1%}")
    print(f"  unstable lambda= [{diag2.lambda_unstable_min_m:.1f}, {diag2.lambda_unstable_max_m:.1f}] m")

    # ── Sigma_width demo from sample trunk data ──────────────────
    print("\n--- sigma_width demo (synthetic widening channel) ---")
    s_demo = np.linspace(0, 10000, 201)  # 10 km, 50m spacing
    B_demo = 300.0 + 50.0 * np.sin(2 * np.pi * s_demo / 5000) + 0.01 * s_demo
    stats = compute_sigma_width_stats(s_demo, B_demo, smooth_window=5)
    print(f"  sigma_width mean   = {stats['mean']:.6f}")
    print(f"  sigma_width std    = {stats['std']:.6f}")
    print(f"  sigma_width median = {stats['median']:.6f}")
    print(f"  sigma_width [p10, p90] = [{stats['p10']:.6f}, {stats['p90']:.6f}]")
    print(f"  frac_positive      = {stats['frac_positive']:.1%}")
    print(f"  frac_in_cross_enh  = {stats['frac_in_cross_enh']:.1%}")
    # ── Batch sigma_width from real trunk data ───────────────────
    print("\n--- Batch sigma_width from trunk data (2000-2023) ---")
    trunk_dir = _PROJECT_ROOT / "results" / "trunks"
    output_csv = _PROJECT_ROOT / "results" / "sigma_width_summary.csv"
    if trunk_dir.exists():
        run_sigma_width_batch(
            trunk_dir, smooth_window=5, output_csv=output_csv,
        )
    else:
        print(f"  [SKIP] Trunk directory not found: {trunk_dir}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()