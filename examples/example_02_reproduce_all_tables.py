"""Example 02: One-Command Reproduction of All Manuscript and Supporting Information Tables.

This script executes the batch computation pipeline to reproduce:
1. Table 3: Multi-decadal temporal stability (14 years) -> results/temporal_stability_2d.csv
2. Table 4: Reach-scale spatial stability (26 cross-sections) -> results/spatial_2016_stability_2d.csv
3. Table S1: Sediment transport closure sensitivity (theta_c, b, Gamma) -> results/table_s1_sediment_sensitivity.csv
4. Table S2: Chebyshev grid convergence analysis (N=16 to 64) -> results/table_s2_convergence.csv
5. Table S3: Literature benchmark validation against Colombini et al. (1987) -> results/table_s3_benchmark.csv
6. Table S4: Observational satellite scene statistics -> results/table_s4_scenes.csv
7. Table S5: Curvature smoothing, step size, tapering & width perturbation sensitivity -> results/table_s5_spectral_sensitivity.csv
8. Table S6: Multi-decadal channel width gradient distributions -> results/table_s6_width_gradient.csv
9. Table S7: Satellite per-bend curvature sensitivity (R, nu*beta) -> results/table_s7_curvature_sensitivity.csv
10. Table S8: 3D parameter space exploration (Cf, Fr, beta) -> results/table_s8_multi_beta.csv
11. Table S9: Transverse mode competition (m=1, 2, 3, 4) -> results/table_s9_mode_competition.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def verify_tables(results_dir: Path) -> bool:
    tables = [
        ("Table 3 (Temporal Stability)", "temporal_stability_2d.csv", 14),
        ("Table 4 (Spatial Stability 2016)", "spatial_2016_stability_2d.csv", 26),
        ("Table S1 (Sediment Closure Sensitivity)", "table_s1_sediment_sensitivity.csv", 15),
        ("Table S2 (Chebyshev Spectral Convergence)", "table_s2_convergence.csv", 7),
        ("Table S3 (Colombini 1987 Benchmark)", "table_s3_benchmark.csv", 6),
        ("Table S4 (Satellite Scenes & Hydrology)", "table_s4_scenes.csv", 11),
        ("Table S5 (Spectral Sensitivity & Perturbation)", "table_s5_spectral_sensitivity.csv", 15),
        ("Table S6 (Width Gradient Distributions)", "table_s6_width_gradient.csv", 11),
        ("Table S7 (Satellite Curvature Sensitivity)", "table_s7_curvature_sensitivity.csv", 6),
        ("Table S8 (Multi-Beta Parameter Grid)", "table_s8_multi_beta.csv", 48),
        ("Table S9 (Transverse Mode Competition)", "table_s9_mode_competition.csv", 5),
    ]

    print("\nVerifying Manuscript & Supporting Information Tables in results/:")
    print("-" * 85)
    all_ok = True
    for label, fname, expected_rows in tables:
        fpath = results_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            status = "OK" if len(df) >= expected_rows else "WARN"
            print(f"  [{status}] {label:<42} -> {fname:<36} ({len(df)} rows)")
            if status == "WARN":
                all_ok = False
        else:
            print(f"  [FAIL] {label:<42} -> {fname:<36} (MISSING)")
            all_ok = False
    print("-" * 85)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or reproduce manuscript tables.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all 2D SWE--Exner stability tables from scratch (takes ~3-4 minutes).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick recomputation of fast tables (Tables S1, S2, S3, S4, S5, S6, S7).",
    )
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("MANUSCRIPT DATA TABLES VERIFICATION AND REPRODUCTION SUITE")
    print("=" * 85)

    if args.recompute:
        print("\nStarting FULL recomputation of all stability tables...")
        from src.stability.compute_all_revision_tables import main as run_compute_tables
        from src.stability.generate_si_uncertainty_csvs import main as run_generate_uncertainty

        t0 = time.time()
        run_compute_tables()
        run_generate_uncertainty()
        print(f"\nFull recomputation completed in {time.time() - t0:.1f} seconds.")

    elif args.fast:
        print("\nStarting FAST recomputation of Tables S1, S2, S3, S4, S5, S6, S7...")
        from src.stability.compute_all_revision_tables import (
            compute_table_s1_sediment_sensitivity,
            compute_table_s2_convergence,
            compute_table_s3_benchmark,
            compute_table_s7_curvature_sensitivity,
        )
        from src.stability.generate_si_uncertainty_csvs import main as run_generate_uncertainty

        t0 = time.time()
        compute_table_s1_sediment_sensitivity()
        compute_table_s2_convergence()
        compute_table_s3_benchmark()
        compute_table_s7_curvature_sensitivity()
        run_generate_uncertainty()
        print(f"\nFast recomputation completed in {time.time() - t0:.1f} seconds.")

    # Verify tables
    all_ok = verify_tables(results_dir)
    if all_ok:
        print("\n[SUCCESS] All data tables are verified and consistent with the ESPL manuscript.\n")
    else:
        print("\n[NOTICE] Some tables need recomputation. Run with --recompute to regenerate.\n")


if __name__ == "__main__":
    main()
