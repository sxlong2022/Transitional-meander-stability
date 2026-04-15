"""OS Stability Diagnosis Module: Diagnose the instability of river sandbanks from hydraulic parameters.

Encapsulate Route B's OS stability solution into a reproducible diagnostic interface.
Supports single point diagnosis, time series batch diagnosis, and spatial profile batch diagnosis.

Typical usage::

    from src.diagnostics import diagnose_stability
    result = diagnose_stability(beta=132.0, Fr=0.257, Cf=0.00176, nu_curvature=0.004)
    print(result.omega_i_max, result.alpha_crit, result.stability_class)
"""
from __future__ import annotations

import csv
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

from src.stability.run_beta_sweep import (  # noqa: E402
    alpha_sweep,
    alpha_sweep_curved,
    find_most_unstable,
    run_single,
    run_single_curved,
)


# =====================================================================
# data class
# =====================================================================

@dataclasses.dataclass
class StabilityResult:
    """OS stability diagnostic results.

    property
    ------
    omega_i_max : float
        Maximum time growth rate (dimensionless).
    alpha_crit : float
        Critical wave number (alpha corresponding to maximum growth rate).
    c_phase : float
        Phase velocity c_r at critical wave number.
    lambda_crit : float
        Critical wavelength (in water depth H) = 2*pi / alpha_crit.
    lambda_crit_m : float
        Critical wavelength (meters), D_m must be provided to calculate; otherwise NaN.
    stability_class : str
        Stability classification: strongly_unstable / weakly_unstable /
        near_marginal / stable。
    curvature_enhancement : float
        The percentage increase in growth rate due to curvature.
    nu_beta_product : float
        Bending parameter = nu * beta (= B / R).
    mode_type : str
        Modal types: short_wave/transitional/long_wave_bend.
    beta : float
        Width to depth ratio B/H.
    Fr : float
        Froude number.
    Cf : float
        Friction coefficient.
    nu_curvature : float
        Curvature parameter H/R.
    """

    omega_i_max: float
    alpha_crit: float
    c_phase: float
    lambda_crit: float
    lambda_crit_m: float
    stability_class: str
    curvature_enhancement: float
    nu_beta_product: float
    mode_type: str
    beta: float
    Fr: float
    Cf: float
    nu_curvature: float


@dataclasses.dataclass
class AlphaSweepResult:
    """OS alpha-sweep complete results, including omega_i(alpha) curve and unstable wavenumber range.

    property
    ------
    alpha_arr : np.ndarray
        Wavenumber array.
    omega_i_arr : np.ndarray
        The maximum time growth rate corresponding to each wave number.
    c_r_arr : np.ndarray
        The phase velocity corresponding to each wave number.
    alpha_unstable_min : float
        Lower bound on the unstable wavenumber range (minimum alpha for omega_i > 0), or NaN if none.
    alpha_unstable_max : float
        Upper bound on the unstable wavenumber range (maximum alpha for omega_i > 0), or NaN if none.
    n_unstable : int
        The number of unstable wavenumber sampling points.
    stability_result : StabilityResult
        Corresponding single point diagnosis results (peak information).
    """

    alpha_arr: np.ndarray
    omega_i_arr: np.ndarray
    c_r_arr: np.ndarray
    alpha_unstable_min: float
    alpha_unstable_max: float
    n_unstable: int
    stability_result: StabilityResult


# =====================================================================
# Classification threshold
# =====================================================================

_STABILITY_THRESHOLDS = {
    "strongly_unstable": 3.0,   # omega_i > 3.0
    "weakly_unstable": 0.0,     # 0 < omega_i <= 3.0
    "near_marginal": -0.5,      # -0.5 < omega_i <= 0
    # stable: omega_i <= -0.5
}


def _classify_stability(omega_i: float) -> str:
    """Classifies stability according to growth rate omega_i."""
    if not np.isfinite(omega_i):
        return "unknown"
    if omega_i > _STABILITY_THRESHOLDS["strongly_unstable"]:
        return "strongly_unstable"
    if omega_i > _STABILITY_THRESHOLDS["weakly_unstable"]:
        return "weakly_unstable"
    if omega_i > _STABILITY_THRESHOLDS["near_marginal"]:
        return "near_marginal"
    return "stable"


def _classify_mode(alpha_crit: float) -> str:
    """Classify mode types based on critical wave numbers."""
    if not np.isfinite(alpha_crit):
        return "unknown"
    if alpha_crit > 0.3:
        return "short_wave"
    if alpha_crit < 0.1:
        return "long_wave_bend"
    return "transitional"


# =====================================================================
# Core diagnostic functions
# =====================================================================

def diagnose_stability(
    beta: float,
    Fr: float,
    Cf: float,
    nu_curvature: float = 0.0,
    alpha_range: tuple[float, float] = (0.01, 15.0),
    n_alpha: int = 200,
    N_cheb: int = 80,
    profile_mode: str = "zs_turbulent",
    A_scour: float = 4.0,
    D_m: float | None = None,
    return_alpha_sweep: bool = False,
) -> StabilityResult | AlphaSweepResult:
    """Diagnosis of OS sandbar instability for given hydraulic parameters.

    parameter
    ------
    beta : float
        Width to depth ratio B/H.
    Fr : float
        Froude number.
    Cf : float
        Friction coefficient (energy method).
    nu_curvature : float
        Curvature parameter H/R, default 0 (pure straight river section OS).
    alpha_range : tuple
        Wavenumber scan range (alpha_min, alpha_max).
    n_alpha : int
        Wavenumber sampling points.
    N_cheb : int
        Order of Chebyshev's collocation method.
    profile_mode : str
        Baseflow profile mode.
    A_scour : float
        Erosion amplification factor.
    D_m : float or None
        Physical water depth (meters), used to convert wavelength to meters.
    return_alpha_sweep : bool
        If True, returns AlphaSweepResult (containing the complete omega_i(alpha) curve
        and instability wavenumber range); otherwise returns StabilityResult (peak values ​​only).

    return
    ------
    StabilityResult or AlphaSweepResult
    """
    if Cf <= 0:
        raise ValueError(f"Cf must be positive, got {Cf}")
    if Fr <= 0:
        raise ValueError(f"Fr must be positive, got {Fr}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    alpha_arr = np.linspace(alpha_range[0], alpha_range[1], n_alpha)

    # ── Step 1: Pure OS baseline (no curvature) ──────────────────────
    pure_results: list[dict] = []
    for alpha in alpha_arr:
        info = run_single(alpha, beta, Fr, Cf, N_cheb, profile_mode)
        pure_results.append(info)

    pure_oi = np.array([r["omega_i_max"] for r in pure_results])
    pure_valid = np.isfinite(pure_oi)

    if not np.any(pure_valid):
        # All eigenvalues spurious — return NaN result
        nan_sr = StabilityResult(
            omega_i_max=np.nan, alpha_crit=np.nan, c_phase=np.nan,
            lambda_crit=np.nan, lambda_crit_m=np.nan,
            stability_class="unknown", curvature_enhancement=0.0,
            nu_beta_product=nu_curvature * beta, mode_type="unknown",
            beta=beta, Fr=Fr, Cf=Cf, nu_curvature=nu_curvature,
        )
        if return_alpha_sweep:
            return AlphaSweepResult(
                alpha_arr=alpha_arr,
                omega_i_arr=pure_oi,
                c_r_arr=np.array([r["c_r"] for r in pure_results]),
                alpha_unstable_min=np.nan,
                alpha_unstable_max=np.nan,
                n_unstable=0,
                stability_result=nan_sr,
            )
        return nan_sr

    pure_best_idx = int(np.where(pure_valid)[0][np.argmax(pure_oi[pure_valid])])
    pure_best = pure_results[pure_best_idx]
    omega_i_pure = pure_best["omega_i_max"]

    # ── Step 2: Curved OS (if nu_curvature > 0) ─────────────────────
    curved_results: list[dict] | None = None
    if nu_curvature > 0:
        curved_results = []
        for alpha in alpha_arr:
            info = run_single_curved(
                alpha, beta, Fr, Cf, nu_curvature, N_cheb,
                profile_mode, "polynomial_zs", A_scour,
            )
            curved_results.append(info)

        curved_oi = np.array([r["omega_i_max"] for r in curved_results])
        curved_valid = np.isfinite(curved_oi)

        if np.any(curved_valid):
            best_idx = int(np.where(curved_valid)[0][
                np.argmax(curved_oi[curved_valid])
            ])
            best = curved_results[best_idx]
            omega_i_max = best["omega_i_max"]
            alpha_crit = best["alpha"]
            c_phase = best["c_r"]
            enhancement = (
                (omega_i_max - omega_i_pure) / abs(omega_i_pure) * 100.0
                if omega_i_pure != 0.0
                else 0.0
            )
        else:
            # Fall back to pure OS
            omega_i_max = omega_i_pure
            alpha_crit = pure_best["alpha"]
            c_phase = pure_best["c_r"]
            enhancement = 0.0
    else:
        omega_i_max = omega_i_pure
        alpha_crit = pure_best["alpha"]
        c_phase = pure_best["c_r"]
        enhancement = 0.0

    # ── Step 3: Derived quantities ───────────────────────────────────
    lambda_crit = 2.0 * np.pi / alpha_crit if alpha_crit > 0 else np.inf
    lambda_crit_m = lambda_crit * D_m if D_m is not None else np.nan

    sr = StabilityResult(
        omega_i_max=omega_i_max,
        alpha_crit=alpha_crit,
        c_phase=c_phase,
        lambda_crit=lambda_crit,
        lambda_crit_m=lambda_crit_m,
        stability_class=_classify_stability(omega_i_max),
        curvature_enhancement=enhancement,
        nu_beta_product=nu_curvature * beta,
        mode_type=_classify_mode(alpha_crit),
        beta=beta,
        Fr=Fr,
        Cf=Cf,
        nu_curvature=nu_curvature,
    )

    if not return_alpha_sweep:
        return sr

    # ── Step 4: Build full alpha-sweep result ────────────────────────
    # Use curved results if available, otherwise pure
    if curved_results is not None:
        final_results = curved_results

    sweep_oi = np.array([r["omega_i_max"] for r in final_results])
    sweep_cr = np.array([r["c_r"] for r in final_results])

    # Replace non-finite values with -inf for clean boundary detection
    sweep_oi_clean = np.where(np.isfinite(sweep_oi), sweep_oi, -np.inf)

    # Find unstable range (omega_i > 0)
    unstable_mask = sweep_oi_clean > 0.0
    if np.any(unstable_mask):
        unstable_indices = np.where(unstable_mask)[0]
        alpha_unstable_min = float(alpha_arr[unstable_indices[0]])
        alpha_unstable_max = float(alpha_arr[unstable_indices[-1]])
        n_unstable = int(np.sum(unstable_mask))
    else:
        alpha_unstable_min = np.nan
        alpha_unstable_max = np.nan
        n_unstable = 0

    return AlphaSweepResult(
        alpha_arr=alpha_arr,
        omega_i_arr=sweep_oi,
        c_r_arr=sweep_cr,
        alpha_unstable_min=alpha_unstable_min,
        alpha_unstable_max=alpha_unstable_max,
        n_unstable=n_unstable,
        stability_result=sr,
    )


# =====================================================================
# Batch Diagnostics: Time Series
# =====================================================================

def diagnose_timeseries(
    csv_path: str | Path,
    nu_curvature: float = 0.004,
    beta_col: str = "beta",
    Fr_col: str = "Fr",
    Cf_col: str = "Cf_energy",
    **kwargs,
) -> list[dict]:
    """Run OS diagnostics on a year-by-year basis for hydraulic parameter time series.

    parameter
    ------
    csv_path : str or Path
        Time series CSV path (requires year, beta, Fr, Cf_energy columns).
    nu_curvature : float
        Curvature parameter, default 0.004.
    beta_col, Fr_col, Cf_col : str
        Column name mapping.
    **kwargs
        Additional parameters passed to diagnose_stability.

    return
    ------
    list[dict]
        Each row contains one year of diagnostic results + year field.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    results: list[dict] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                year = row.get("year", "")
                b = float(row[beta_col])
                fr = float(row[Fr_col])
                cf = float(row[Cf_col])
            except (KeyError, ValueError, TypeError):
                continue

            if not (np.isfinite(b) and np.isfinite(fr) and np.isfinite(cf)):
                continue
            if b <= 0 or fr <= 0 or cf <= 0:
                continue

            print(f"  Diagnosing year={year}  beta={b:.1f}  Fr={fr:.4f}  Cf={cf:.6f}")
            sr = diagnose_stability(b, fr, cf, nu_curvature, **kwargs)
            d = dataclasses.asdict(sr)
            d["year"] = year
            results.append(d)

    return results


# =====================================================================
# Batch Diagnostics: Spatial Profiles
# =====================================================================

def diagnose_spatial(
    csv_path: str | Path,
    nu_curvature: float = 0.004,
    beta_col: str = "beta",
    Fr_col: str = "Fr",
    Cf_col: str = "Cf_energy",
    distance_col: str = "dist_km",
    **kwargs,
) -> list[dict]:
    """Run OS diagnostics on a section-by-section basis on spatial profiles.

    parameter
    ------
    csv_path : str or Path
        Spatial CSV path (requires dist_km, beta, Fr, Cf_energy columns).
    nu_curvature : float
        Curvature parameter, default 0.004.
    distance_col : str
        Distance column name.
    **kwargs
        Additional parameters passed to diagnose_stability.

    return
    ------
    list[dict]
        Each line is the diagnostic result of one section + dist_km field.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    results: list[dict] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dist = row.get(distance_col, "")
                b = float(row[beta_col])
                fr = float(row[Fr_col])
                cf = float(row[Cf_col])
            except (KeyError, ValueError, TypeError):
                continue

            if not (np.isfinite(b) and np.isfinite(fr) and np.isfinite(cf)):
                continue
            if b <= 0 or fr <= 0 or cf <= 0:
                continue

            print(f"  Diagnosing dist={dist}km  beta={b:.1f}  Fr={fr:.4f}  Cf={cf:.6f}")
            sr = diagnose_stability(b, fr, cf, nu_curvature, **kwargs)
            d = dataclasses.asdict(sr)
            d[distance_col] = dist
            results.append(d)

    return results


# =====================================================================
# CLI demo
# =====================================================================

def main():
    """Demo: diagnose field conditions (recent + early periods)."""
    print("=" * 60)
    print("OS Stability Diagnostic Module - Demo")
    print("=" * 60)

    # Recent period (2011-2021 mean)
    print("\n--- Recent period (2011-2021) ---")
    r = diagnose_stability(
        beta=132.0, Fr=0.257, Cf=0.00176, nu_curvature=0.004,
    )
    print(f"  omega_i_max    = {r.omega_i_max:.6f}")
    print(f"  alpha_crit     = {r.alpha_crit:.6f}")
    print(f"  c_phase        = {r.c_phase:.6f}")
    print(f"  lambda_crit    = {r.lambda_crit:.2f} (depths)")
    print(f"  stability      = {r.stability_class}")
    print(f"  curvature_enh  = {r.curvature_enhancement:.2f}%")
    print(f"  nu*beta        = {r.nu_beta_product:.4f}")
    print(f"  mode_type      = {r.mode_type}")

    # Early period (2000-2003 mean)
    print("\n--- Early period (2000-2003) ---")
    r2 = diagnose_stability(
        beta=253.0, Fr=0.448, Cf=0.00070, nu_curvature=0.004,
    )
    print(f"  omega_i_max    = {r2.omega_i_max:.6f}")
    print(f"  alpha_crit     = {r2.alpha_crit:.6f}")
    print(f"  c_phase        = {r2.c_phase:.6f}")
    print(f"  lambda_crit    = {r2.lambda_crit:.2f} (depths)")
    print(f"  stability      = {r2.stability_class}")
    print(f"  curvature_enh  = {r2.curvature_enhancement:.2f}%")
    print(f"  nu*beta        = {r2.nu_beta_product:.4f}")
    print(f"  mode_type      = {r2.mode_type}")

    # Pure OS (no curvature) for comparison
    print("\n--- Pure OS (no curvature) at recent conditions ---")
    r3 = diagnose_stability(
        beta=132.0, Fr=0.257, Cf=0.00176, nu_curvature=0.0,
    )
    print(f"  omega_i_max    = {r3.omega_i_max:.6f}")
    print(f"  alpha_crit     = {r3.alpha_crit:.6f}")
    print(f"  stability      = {r3.stability_class}")
    print("\n--- Alpha-sweep at recent conditions (curved OS) ---")
    asw = diagnose_stability(
        beta=132.0, Fr=0.257, Cf=0.00176, nu_curvature=0.004,
        return_alpha_sweep=True, n_alpha=100,
    )
    print(f"  alpha range    = [{asw.alpha_arr[0]:.2f}, {asw.alpha_arr[-1]:.2f}]")
    print(f"  n_alpha        = {len(asw.alpha_arr)}")
    print(f"  omega_i range  = [{np.nanmin(asw.omega_i_arr):.4f}, {np.nanmax(asw.omega_i_arr):.4f}]")
    print(f"  unstable range = [{asw.alpha_unstable_min:.4f}, {asw.alpha_unstable_max:.4f}]")
    print(f"  n_unstable     = {asw.n_unstable} / {len(asw.alpha_arr)}")
    print(f"  peak omega_i   = {asw.stability_result.omega_i_max:.6f}")
    print(f"  peak alpha     = {asw.stability_result.alpha_crit:.6f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
