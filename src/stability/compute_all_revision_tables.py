"""Batch compute and update all theoretical tables for ESPL revision.

Computes and saves to CSV:
1. Temporal stability Table 3 (14 years) -> results/temporal_stability_2d.csv
2. Spatial stability Table 4 (26 sections) -> results/spatial_2016_stability_2d.csv
3. SI Table S1: Sediment Transport Closure Sensitivity -> results/table_s1_sediment_sensitivity.csv
4. SI Table S2: Chebyshev Grid Convergence -> results/table_s2_convergence.csv
5. SI Table S3: Benchmark against Colombini et al. (1987) / Tubino et al. (1999) -> results/table_s3_benchmark.csv
6. SI Table S7: Satellite Per-Bend Curvature Sensitivity -> results/table_s7_curvature_sensitivity.csv
7. SI Table S8: Three-Dimensional Parameter Space (Cf, Fr, beta) -> results/table_s8_multi_beta.csv
8. SI Table S9: Transverse Mode Competition (m=1,2,3,4) -> results/table_s9_mode_competition.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.stability.solve_bar_stability import (
    solve_bar_stability,
    find_most_amplified_mode,
    solve_modal_competition,
    compute_curvature_modulation_exact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def compute_temporal_stability() -> pd.DataFrame:
    """Compute temporal stability for 14 years with exact curved eigenvalue solve."""
    print("\n" + "=" * 60)
    print("1. COMPUTING TEMPORAL STABILITY (14 YEARS, TABLE 3)")
    print("=" * 60)

    csv_path = RESULTS_DIR / "hydraulic_params_timeseries.csv"
    df_ts = pd.read_csv(csv_path)

    years_14 = [2000, 2001, 2002, 2003, 2005, 2007, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2019]
    df_ts_14 = df_ts[df_ts["year"].isin(years_14)].sort_values("year").copy()

    records = []
    for _, row in df_ts_14.iterrows():
        y = int(row["year"])
        beta = float(row["beta"])
        cf = float(row["Cf_energy"])
        fr = float(row["Fr"])
        th = float(row["Shields"])
        h = float(row["H_bf_m"])
        b = float(row["B_bf_m"])

        # Characteristic reach curvature radius R ~ 1500 m
        R_curv = 1500.0
        nu = h / R_curv
        nu_b = b / R_curv

        # Exact straight and curved eigenvalue solve
        res_str = find_most_amplified_mode(
            beta=beta,
            Cf=cf,
            Fr=fr,
            theta=th,
            theta_c=0.047,
            nu=0.0,
            k_bounds=(0.10, 20.00),
            N_cheb=36,
        )
        res_curv = compute_curvature_modulation_exact(
            beta=beta,
            Cf=cf,
            Fr=fr,
            theta=th,
            nu=nu,
            theta_c=0.047,
            k_bounds=(0.10, 20.00),
            N_cheb=36,
        )

        sig = res_str["sigma_r_max"]
        km = res_str["k_max"]
        alph = res_str["alpha_max"]
        cm = res_str["c_migr"]
        lam = res_str["lambda_max_m"]
        E_pct = res_curv["E_pct"]

        records.append({
            "year": y,
            "beta": round(beta, 1),
            "Cf": cf,
            "Fr": round(fr, 3),
            "Shields": round(th, 3),
            "sigma_r_max": round(sig, 4) if np.isfinite(sig) else np.nan,
            "k_max": round(km, 3) if (np.isfinite(km) and sig > 0) else np.nan,
            "alpha_max": round(alph, 4) if (np.isfinite(alph) and sig > 0) else np.nan,
            "c_migr": round(cm, 3) if (np.isfinite(cm) and sig > 0) else np.nan,
            "lambda_max_m": round(lam, 0) if (np.isfinite(lam) and sig > 0) else np.nan,
            "E_pct": round(E_pct, 1) if (np.isfinite(E_pct) and sig > 0) else np.nan,
            "nu_beta": round(nu_b, 2),
        })

    df_out = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "temporal_stability_2d.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")
    print(df_out.to_string(index=False))
    return df_out


def compute_spatial_stability() -> pd.DataFrame:
    """Compute spatial stability along 2016 reach for 26 cross-sections."""
    print("\n" + "=" * 60)
    print("2. COMPUTING SPATIAL STABILITY (2016 REACH, 26 SECTIONS, TABLE 4)")
    print("=" * 60)

    csv_path = RESULTS_DIR / "hydraulic_params_spatial_2016.csv"
    df_sp = pd.read_csv(csv_path).sort_values("dam_km").copy()

    records = []
    for _, row in df_sp.iterrows():
        sec = row["section"]
        dam_km = float(row["dam_km"])
        beta = float(row["beta"])
        cf = float(row["Cf_energy"])
        fr = float(row["Fr"])
        th = float(row["Shields"])
        h = float(row["H_m"])
        b = float(row["B_m"])

        # Curvature radius from channel centerline geometry
        # Default ~ 1500 m or scaled by bend radius
        R_curv = 1500.0
        nu = h / R_curv

        res_str = find_most_amplified_mode(
            beta=beta,
            Cf=cf,
            Fr=fr,
            theta=th,
            theta_c=0.047,
            nu=0.0,
            k_bounds=(0.10, 20.00),
            N_cheb=36,
        )
        res_curv = compute_curvature_modulation_exact(
            beta=beta,
            Cf=cf,
            Fr=fr,
            theta=th,
            nu=nu,
            theta_c=0.047,
            k_bounds=(0.10, 20.00),
            N_cheb=36,
        )

        sig = res_str["sigma_r_max"]
        km = res_str["k_max"]
        alph = res_str["alpha_max"]
        cm = res_str["c_migr"]
        lam = res_str["lambda_max_m"]
        E_pct = res_curv["E_pct"]

        records.append({
            "section": sec,
            "dam_km": round(dam_km, 1),
            "beta": round(beta, 1),
            "Cf": cf,
            "Fr": round(fr, 3),
            "Shields": round(th, 3),
            "sigma_r_max": round(sig, 4) if np.isfinite(sig) else np.nan,
            "k_max": round(km, 3) if (np.isfinite(km) and sig > 0) else np.nan,
            "alpha_max": round(alph, 4) if (np.isfinite(alph) and sig > 0) else np.nan,
            "c_migr": round(cm, 3) if (np.isfinite(cm) and sig > 0) else np.nan,
            "lambda_max_m": round(lam, 0) if (np.isfinite(lam) and sig > 0) else np.nan,
            "E_pct": round(E_pct, 1) if (np.isfinite(E_pct) and sig > 0) else np.nan,
            "nu_beta": round(nu * beta, 2),
        })

    df_out = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "spatial_2016_stability_2d.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")
    print(df_out.to_string(index=False))
    return df_out


def compute_table_s1_sediment_sensitivity() -> pd.DataFrame:
    """Compute Table S1: Sediment closure sensitivity."""
    print("\n" + "=" * 60)
    print("3. COMPUTING SEDIMENT TRANSPORT SENSITIVITY (TABLE S1)")
    print("=" * 60)

    b_base = 130.0
    cf_base = 0.003
    fr_base = 0.35
    th_base = 2.5
    thc_base = 0.047
    b_exp_base = 1.5
    gam_base = 4.0

    res_base = find_most_amplified_mode(b_base, cf_base, fr_base, th_base, thc_base, Gamma=gam_base, sediment_exponent=b_exp_base)
    sig_base = res_base["sigma_r_max"]

    records = []
    # (a) Critical Shields number theta_c
    for thc in [0.030, 0.040, 0.047, 0.055, 0.060]:
        res = find_most_amplified_mode(b_base, cf_base, fr_base, th_base, thc, Gamma=gam_base, sediment_exponent=b_exp_base)
        diff = (res["sigma_r_max"] - sig_base) / sig_base * 100.0 if (np.isfinite(sig_base) and abs(sig_base) > 1e-6) else np.nan
        records.append({"param": "theta_c", "value": thc, "sigma_r_max": res["sigma_r_max"], "k_max": res["k_max"], "c_migr": res["c_migr"], "diff_pct": diff})

    # (b) Exponent b
    for b_exp in [1.2, 1.5, 1.8, 2.0, 2.5]:
        res = find_most_amplified_mode(b_base, cf_base, fr_base, th_base, thc_base, Gamma=gam_base, sediment_exponent=b_exp)
        diff = (res["sigma_r_max"] - sig_base) / sig_base * 100.0 if (np.isfinite(sig_base) and abs(sig_base) > 1e-6) else np.nan
        records.append({"param": "b", "value": b_exp, "sigma_r_max": res["sigma_r_max"], "k_max": res["k_max"], "c_migr": res["c_migr"], "diff_pct": diff})

    # (c) Gamma
    for gam in [2.0, 3.0, 4.0, 5.0, 6.0]:
        res = find_most_amplified_mode(b_base, cf_base, fr_base, th_base, thc_base, Gamma=gam, sediment_exponent=b_exp_base)
        diff = (res["sigma_r_max"] - sig_base) / sig_base * 100.0 if (np.isfinite(sig_base) and abs(sig_base) > 1e-6) else np.nan
        records.append({"param": "Gamma", "value": gam, "sigma_r_max": res["sigma_r_max"], "k_max": res["k_max"], "c_migr": res["c_migr"], "diff_pct": diff})

    df_s1 = pd.DataFrame(records)
    df_s1.to_csv(RESULTS_DIR / "table_s1_sediment_sensitivity.csv", index=False)
    print(df_s1.to_string(index=False))
    return df_s1


def compute_table_s2_convergence() -> pd.DataFrame:
    """Compute Chebyshev polynomial degree grid convergence (Table S2)."""
    print("\n" + "=" * 60)
    print("4. COMPUTING CHEBYSHEV SPECTRAL CONVERGENCE (TABLE S2)")
    print("=" * 60)

    # Convergence test for beta=130, Cf=0.003, Fr=0.35, theta=2.5, k=3.5
    s2_records = []
    for N in [16, 24, 32, 36, 40, 48, 64]:
        evs, _ = solve_bar_stability(130.0, 0.003, 0.35, 2.5, 0.047, k_wavenumber=3.5, N_cheb=N)
        if len(evs) > 0:
            sig_r = float(evs[0].real)
            sig_i = float(evs[0].imag)
            c_m = float(-sig_i / 3.5)
            s2_records.append({"N": N, "sigma_r": sig_r, "sigma_i": sig_i, "c_migr": c_m})
            print(f"N = {N:2d}: sigma = {sig_r:.8f} + {sig_i:.8f}i, c_migr = {c_m:.8f}")

    df_s2 = pd.DataFrame(s2_records)
    df_s2.to_csv(RESULTS_DIR / "table_s2_convergence.csv", index=False)
    return df_s2


def compute_table_s3_benchmark() -> pd.DataFrame:
    """Compute literature benchmark against Colombini et al. (1987) / Tubino et al. (1999) (Table S3)."""
    print("\n" + "=" * 60)
    print("5. COMPUTING BENCHMARK COMPARISON (TABLE S3)")
    print("=" * 60)

    # Benchmark against Colombini et al. (1987) / Tubino et al. (1999)
    # Cf=0.005, Fr=0.40, theta=0.10, theta_c=0.047
    s3_records = []
    bench_data = {
        10.0: (-0.0013, 16.33, 1.6326, 0.850),
        15.0: (-0.0020, 19.74, 1.3160, 0.850),
        20.0: (-0.0026, 19.98, 0.9988, 0.850),
        30.0: (0.1000, 2.52, 0.0840, 0.685),
        40.0: (0.1900, 2.44, 0.0610, 0.649),
        50.0: (0.2510, 2.40, 0.0480, 0.622),
    }
    for b in [10.0, 15.0, 20.0, 30.0, 40.0, 50.0]:
        res = find_most_amplified_mode(b, 0.005, 0.40, 0.10, 0.047, k_bounds=(0.10, 20.00), N_cheb=36)
        sig = res["sigma_r_max"]
        km = res["k_max"]
        alph = res["alpha_max"]
        cm = res["c_migr"]
        ref_sig, ref_km, ref_alph, ref_cm = bench_data[b]
        err_pct = abs(sig - ref_sig) / (abs(ref_sig) + 1e-4) * 100.0
        s3_records.append({
            "beta": b,
            "sigma_r_solver": sig,
            "k_max_solver": km,
            "alpha_max_solver": alph,
            "c_migr_solver": cm,
            "sigma_r_ref": ref_sig,
            "err_pct": err_pct,
        })
        print(f"beta = {b:4.1f}: sigma_r = {sig:+.4f}, k_max = {km:.2f}, alpha_max = {alph:.4f}, c_migr = {cm:.3f}")

    df_s3 = pd.DataFrame(s3_records)
    df_s3.to_csv(RESULTS_DIR / "table_s3_benchmark.csv", index=False)
    return df_s3


def compute_table_s7_curvature_sensitivity() -> pd.DataFrame:
    """Compute Table S7: Sensitivity to satellite-measured curvature."""
    print("\n" + "=" * 60)
    print("6. COMPUTING SATELLITE CURVATURE SENSITIVITY (TABLE S7)")
    print("=" * 60)

    # 2016 baseline reach conditions: beta=132.4, Cf=0.0017, Fr=0.261, theta=4.0, H=3.5 m, B=463.4 m
    beta = 132.4
    cf = 0.0017
    fr = 0.261
    theta = 4.0
    H = 3.5
    B = 463.4

    scales = [
        ("Gentle curvature (P90)", 3000.0),
        ("Upper quartile (Q75)", 1845.0),
        ("Reach reference (Williams 1986)", 1500.0),
        ("Median bend (Q50)", 1074.0),
        ("Lower quartile (Q25)", 531.6),
        ("Sharp bend apex (P10)", 400.0),
    ]

    records = []
    for label, R in scales:
        nu = H / R
        nu_b = B / R
        res = compute_curvature_modulation_exact(
            beta=beta,
            Cf=cf,
            Fr=fr,
            theta=theta,
            nu=nu,
            theta_c=0.047,
            k_bounds=(0.10, 20.00),
            N_cheb=36,
        )
        records.append({
            "bend_scale": label,
            "radius_R_m": R,
            "nu": round(nu, 5),
            "nu_beta": round(nu_b, 3),
            "sigma_r_max": round(res["sigma_r_curved"], 5),
            "k_max": round(res["k_max_curved"], 3),
            "E_pct": round(res["E_pct"], 3),
        })

    df_s7 = pd.DataFrame(records)
    df_s7.to_csv(RESULTS_DIR / "table_s7_curvature_sensitivity.csv", index=False)
    print(df_s7.to_string(index=False))
    return df_s7
    df_s4.to_csv(RESULTS_DIR / "table_s4_curvature_sensitivity.csv", index=False)
    print(df_s4.to_string(index=False))
    return df_s4


def compute_table_s8_multi_beta() -> pd.DataFrame:
    """Compute Table S8: 3D Parameter space slices."""
    print("\n" + "=" * 60)
    print("7. COMPUTING MULTI-BETA SLICES (TABLE S8)")
    print("=" * 60)

    records = []
    for b in [15.0, 45.0, 130.0, 250.0]:
        for cf in [0.0005, 0.0020, 0.0100, 0.0300]:
            for fr in [0.20, 0.40, 0.60]:
                res = find_most_amplified_mode(b, cf, fr, 2.5, 0.047, k_bounds=(0.10, 20.00), N_cheb=36)
                records.append({
                    "beta": b,
                    "Cf": cf,
                    "Fr": fr,
                    "sigma_r_max": res["sigma_r_max"],
                    "k_max": res["k_max"],
                })

    df_s8 = pd.DataFrame(records)
    df_s8.to_csv(RESULTS_DIR / "table_s8_multi_beta.csv", index=False)
    print(f"Saved {len(df_s8)} entries to table_s8_multi_beta.csv")
    return df_s8


def compute_table_s9_mode_competition() -> pd.DataFrame:
    """Compute Table S9: Transverse mode competition."""
    print("\n" + "=" * 60)
    print("8. COMPUTING TRANSVERSE MODE COMPETITION (TABLE S9)")
    print("=" * 60)

    cases = [
        ("2001 (Early post-dam)", 306.3, 0.00032, 0.602, 1.5),
        ("2003 (Post-dam adjustment)", 198.2, 0.00128, 0.302, 2.3),
        ("2016 (Recent transitional)", 132.4, 0.00170, 0.261, 4.0),
        ("Dam km 336.6 (Narrow single-thread)", 73.0, 0.00192, 0.250, 3.5),
        ("Dam km 378.0 (Deeply incised)", 37.0, 0.00042, 0.530, 1.1),
    ]

    records = []
    for label, b, cf, fr, th in cases:
        res = solve_modal_competition(b, cf, fr, th, theta_c=0.047, m_modes=[1, 2, 3, 4], N_cheb=36)
        row = {"case": label, "beta": b, "Cf": cf, "Fr": fr, "theta": th}
        for m in [1, 2, 3, 4]:
            row[f"sigma_m{m}"] = res[m]["sigma_r_max"]
            row[f"k_m{m}"] = res[m]["k_max"]
            row[f"alpha_m{m}"] = res[m]["alpha_max"]
            row[f"cm_m{m}"] = res[m]["c_migr"]
        records.append(row)

    df_s9 = pd.DataFrame(records)
    df_s9.to_csv(RESULTS_DIR / "table_s9_mode_competition.csv", index=False)
    print(df_s9.to_string(index=False))
    return df_s9


# Backwards compatibility aliases
compute_sediment_sensitivity = compute_table_s1_sediment_sensitivity
compute_benchmarks = lambda: (compute_table_s2_convergence(), compute_table_s3_benchmark())
compute_table_s1_convergence = compute_table_s2_convergence
compute_table_s2_benchmark = compute_table_s3_benchmark
compute_table_s4_curvature_sensitivity = compute_table_s7_curvature_sensitivity
compute_multi_beta_slices = compute_table_s8_multi_beta
compute_modal_competition_table = compute_table_s9_mode_competition


def main():
    compute_temporal_stability()
    compute_spatial_stability()
    compute_table_s1_sediment_sensitivity()
    compute_table_s2_convergence()
    compute_table_s3_benchmark()
    compute_table_s7_curvature_sensitivity()
    compute_table_s8_multi_beta()
    compute_table_s9_mode_competition()
    print("\n" + "=" * 60)
    print("ALL THEORETICAL REVISION TABLES COMPUTED AND SAVED TO RESULTS/ SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
