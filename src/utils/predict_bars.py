"""Bar mode prediction and diagnostic module.

Combines Crosato-Mosselman (2009) empirical formula, OS instability wavenumber window,
and along-stream width gradient sigma_width diagnostics.

Typical usage::

    from src.utils import predict_bar_mode_cm, compute_unstable_window
    m = predict_bar_mode_cm(beta=132.0, Cf=0.00176)
    window = compute_unstable_window(beta=132.0, Fr=0.257, Cf=0.00176)
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

from src.utils.diagnose_stability import (  # noqa: E402
    AlphaSweepResult,
    diagnose_stability,
)


# =====================================================================
# Dataclass
# =====================================================================

@dataclasses.dataclass
class BarDiagnostic:
    """Comprehensive bar diagnostic result.

    Attributes
    ----------
    beta : float
        Aspect ratio B / (2H).
    Fr : float
        Froude number.
    Cf : float
        Friction coefficient.
    m_cm : int
        Crosato-Mosselman predicted bar mode number.
    bar_regime : str
        Bar type: single_row / multi_row / braided.
    alpha_crit : float
        OS most unstable wavenumber.
    omega_i_max : float
        OS maximum growth rate.
    lambda_crit_m : float
        OS most unstable wavelength (m); requires D_m.
    alpha_unstable_min : float
        Unstable wavenumber lower bound.
    alpha_unstable_max : float
        Unstable wavenumber upper bound.
    n_unstable_frac : float
        Fraction of sampled unstable wavenumbers.
    lambda_unstable_min_m : float
        Unstable wavelength lower bound (m).
    lambda_unstable_max_m : float
        Unstable wavelength upper bound (m).
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
# Crosato-Mosselman (2009) bar mode prediction
# =====================================================================

def predict_bar_mode_cm(
    beta: float,
    Cf: float,
    b_param: float = 1.7,
    sediment_exponent: float | None = None,
) -> int:
    """Crosato-Mosselman (2009) empirical formula for bar mode prediction.

    Simplified formula::

        m = round( (2 * beta) / (b * pi) )

    where b ≈ 1.7 (Crosato & Mosselman 2009, Table 1 recommended value).
    When sediment_exponent is provided, the full formula is used::

        m = round( (beta / pi) * sqrt(Cf / epsilon) )

    Parameters
    ----------
    beta : float
        Aspect ratio B / H.
    Cf : float
        Friction coefficient (for full formula).
    b_param : float
        Empirical constant for simplified formula, default 1.7.
    sediment_exponent : float or None
        Sediment transport exponent epsilon. If None, use simplified formula.

    Returns
    -------
    int
        Predicted bar mode number m (≥1).

    Note
    ----
    For high β (e.g., β > 100), m is large (multiple rows / braided),
    consistent with the historical braided character of the Lower Yellow River.
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
# OS unstable wavenumber window
# =====================================================================

def compute_unstable_window(
    beta: float,
    Fr: float,
    Cf: float,
    nu_curvature: float = 0.0,
    alpha_range: Tuple[float, float] = (0.01, 15.0),
    n_alpha: int = 200,
    D_m: float | None = None,
    **kwargs,
) -> AlphaSweepResult:
    """Compute OS unstable wavenumber window (full alpha-sweep curve).

    Parameters
    ----------
    beta : float
        Aspect ratio.
    Fr : float
        Froude number.
    Cf : float
        Friction coefficient.
    nu_curvature : float
        Curvature parameter H/R, default 0.
    alpha_range : Tuple
        Wavenumber sweep range, default (0.01, 15.0).
    n_alpha : int
        Number of wavenumber samples, default 200.
    D_m : float or None
        Physical water depth (m), for wavelength conversion.
    **kwargs
        Other parameters passed to diagnose_stability.

    Returns
    -------
    AlphaSweepResult
        Contains full omega_i(alpha) curve and unstable range.
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
# Along-stream width gradient sigma_width
# =====================================================================

def compute_sigma_width(
    s_m: np.ndarray,
    B_m: np.ndarray,
    smooth_window: int = 0,
) -> np.ndarray:
    """Compute normalized width gradient sigma_width = (1/B) dB/ds from along-stream profile.

    Parameters
    ----------
    s_m : np.ndarray
        Along-stream distance coordinate (m), monotonically increasing.
    B_m : np.ndarray
        Along-stream channel width (m), same length as s_m.
    smooth_window : int
        Smoothing window size (0 = no smoothing). If > 0, apply moving average
        to B_m before differentiation.

    Returns
    -------
    sigma_width : np.ndarray
        Normalized width gradient array, same length as input (endpoints
        use forward/backward diff).

    Note
    ----
    sigma_width > 0 indicates widening, sigma_width < 0 indicates narrowing.
    In non-uniform channels, zones with small positive sigma_width (≪ O(0.1))
    often correspond to bar-bend cross-enhancement regime (gamma_BA < 0), i.e.,
    gradual widening promotes bar growth in bends.
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
    """Compute statistics of along-stream sigma_width.

    Parameters
    ----------
    s_m : np.ndarray
        Along-stream distance (m).
    B_m : np.ndarray
        Along-stream width (m).
    smooth_window : int
        Smoothing window size, default 5 points.

    Returns
    -------
    dict
        Contains mean, std, median, p10, p90, frac_positive, frac_in_cross_enh
        and other statistics.
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
        # Fraction of points with small positive sigma_width (gradual widening),
        # indicative of bar-bend cross-enhancement regime.
        # Threshold O(0.1) used as a conservative upper bound.
        "frac_in_cross_enh": float(
            np.sum((valid > 0) & (valid < 0.1)) / len(valid)
        ),
        "sigma_width": sigma,
    }


# =====================================================================
# Comprehensive diagnostics
# =====================================================================

def diagnose_bar_regime(
    beta: float,
    Fr: float,
    Cf: float,
    nu_curvature: float = 0.0,
    D_m: float | None = None,
    B_m_phys: float | None = None,
    alpha_range: Tuple[float, float] = (0.01, 15.0),
    n_alpha: int = 200,
    **kwargs,
) -> BarDiagnostic:
    """Comprehensive bar mode diagnostics: Crosato-Mosselman + OS unstable window.

    Parameters
    ----------
    beta : float
        Aspect ratio B / H.
    Fr : float
        Froude number.
    Cf : float
        Friction coefficient.
    nu_curvature : float
        Curvature parameter H/R.
    D_m : float or None
        Physical water depth (m).
    B_m_phys : float or None
        Physical channel width (m), for wavelength-to-meter conversion.
        If None, inferred from beta * 2 * D_m.
    alpha_range : Tuple
        Wavenumber sweep range.
    n_alpha : int
        Number of wavenumber samples.
    **kwargs
        Other parameters passed to diagnose_stability.

    Returns
    -------
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
    #    λ(physical) = λ(non-dim) × D_m = (2π/α) × D_m
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

    # ── Sigma_width demo from synthetic data ──────────────────
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

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()