"""Cf–Fr parameter space sweep: generates 2D SWE-Exner stability phase diagram data.

Sweeps the (Cf, Fr) parameter grid for a fixed aspect ratio (beta=130), finds the
most amplified longitudinal wavenumber k_max and peak growth rate sigma_r_max,
and saves the output to CSV.

Usage:
    python -m src.stability.run_phase_diagram
    python -m src.stability.run_phase_diagram --n-cf 25 --n-fr 25 --beta 130

Output:
    results/phase_diagram/phase_diagram_omega_i.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# third-party
import numpy as np
# ── sys.path hack ─────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── UTF-8 stdout ──────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from src.stability.solve_bar_stability import find_most_amplified_mode
def run_phase_diagram(
    cf_range: tuple[float, float] = (0.0003, 0.030),
    fr_range: tuple[float, float] = (0.08, 0.95),
    n_cf: int = 25,
    n_fr: int = 25,
    beta_fixed: float = 130.0,
    theta: float = 3.0,
    theta_c: float = 0.047,
    k_bounds: tuple[float, float] = (0.10, 20.00),
    N_cheb: int = 36,
    output_csv: Path | None = None,
) -> dict:
    """Sweep (Cf, Fr) parameter space and compute sigma_r_max using bounded scalar optimization.

    Parameters
    ----------
    cf_range : tuple
        (Cf_min, Cf_max) for the sweep, log-spaced.
    fr_range : tuple
        (Fr_min, Fr_max) for the sweep, linearly spaced.
    n_cf, n_fr : int
        Grid resolution in each dimension.
    beta_fixed : float
        Fixed aspect ratio (beta=130 for representative transitional river reach).
    theta : float
        Shields parameter (default 3.0).
    theta_c : float
        Critical Shields parameter (default 0.047).
    k_bounds : tuple
        Width-scaled wavenumber search bounds (default 0.10 to 20.00).
    N_cheb : int
        Chebyshev polynomial truncation degree (default 36).
    output_csv : Path or None
        If provided, save results to CSV.

    Returns
    -------
    dict with keys: cf_arr, fr_arr, sigma_r_grid, k_max_grid
    """
    cf_arr = np.logspace(np.log10(cf_range[0]), np.log10(cf_range[1]), n_cf)
    fr_arr = np.linspace(fr_range[0], fr_range[1], n_fr)
    sigma_r_grid = np.full((n_cf, n_fr), np.nan)
    k_max_grid = np.full((n_cf, n_fr), np.nan)

    total = n_cf * n_fr
    print(f"2D SWE-Exner Phase diagram sweep: {n_cf} x {n_fr} = {total} points")
    print(f"  Cf range: [{cf_range[0]:.5f}, {cf_range[1]:.5f}] (log-spaced)")
    print(f"  Fr range: [{fr_range[0]:.2f}, {fr_range[1]:.2f}] (linear)")
    print(f"  beta (fixed): {beta_fixed}, theta: {theta}, theta_c: {theta_c}")
    print(f"  k bounds: {k_bounds}, N_cheb: {N_cheb}")

    t0 = time.time()
    count = 0
    for i, cf in enumerate(cf_arr):
        for j, fr in enumerate(fr_arr):
            count += 1
            try:
                res = find_most_amplified_mode(
                    beta=beta_fixed,
                    Cf=cf,
                    Fr=fr,
                    theta=theta,
                    theta_c=theta_c,
                    k_bounds=k_bounds,
                    N_cheb=N_cheb,
                )
                sigma_r_grid[i, j] = res["sigma_r_max"]
                k_max_grid[i, j] = res["k_max"]
            except Exception:
                sigma_r_grid[i, j] = np.nan
                k_max_grid[i, j] = np.nan

            if count % 50 == 0 or count == total:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate if rate > 0 else 0
                print(
                    f"  [{count}/{total}] "
                    f"Cf={cf:.5f}, Fr={fr:.3f} -> "
                    f"sigma_r_max={sigma_r_grid[i,j]:.3f}  "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )
    elapsed = time.time() - t0
    print(f"\nSweep complete in {elapsed:.1f}s")

    # Summary statistics
    valid = sigma_r_grid[np.isfinite(sigma_r_grid)]
    if len(valid) > 0:
        print(f"  sigma_r_max range: [{valid.min():.3f}, {valid.max():.3f}]")
        print(f"  Fraction unstable (sigma_r_max > 0): {(valid > 0).sum()}/{len(valid)}")
    print(f"  NaN count: {np.isnan(sigma_r_grid).sum()}")

    # Save to CSV
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Cf", "Fr", "sigma_r_max", "k_max", "alpha_max"])
            for i, cf in enumerate(cf_arr):
                for j, fr in enumerate(fr_arr):
                    alpha_val = k_max_grid[i, j] / beta_fixed if np.isfinite(k_max_grid[i, j]) else np.nan
                    writer.writerow([
                        f"{cf:.6f}",
                        f"{fr:.4f}",
                        f"{sigma_r_grid[i,j]:.6f}",
                        f"{k_max_grid[i,j]:.6f}",
                        f"{alpha_val:.6f}",
                    ])
        print(f"  Saved to {output_csv}")

    return {
        "cf_arr": cf_arr,
        "fr_arr": fr_arr,
        "sigma_r_grid": sigma_r_grid,
        "k_max_grid": k_max_grid,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D SWE-Exner stability phase diagram on (Cf, Fr) grid"
    )
    parser.add_argument("--n-cf", type=int, default=25, help="Cf grid points")
    parser.add_argument("--n-fr", type=int, default=25, help="Fr grid points")
    parser.add_argument("--beta", type=float, default=130.0, help="Fixed beta")
    parser.add_argument("--theta", type=float, default=3.0, help="Shields parameter")
    parser.add_argument("--n-cheb", type=int, default=36, help="Chebyshev polynomial degree")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase_diagram" / "phase_diagram_omega_i.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    run_phase_diagram(
        n_cf=args.n_cf,
        n_fr=args.n_fr,
        beta_fixed=args.beta,
        theta=args.theta,
        N_cheb=args.n_cheb,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
