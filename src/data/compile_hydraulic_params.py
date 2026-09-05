"""Compile reach-averaged and spatial hydraulic parameter tables for the Gaocun–Sunkou reach.

Integrates hydrometric observation data:
  - Qin (2025): B_bf(t), H_bf(t) reach-averaged annual timeseries (2000-2021)
  - Zhang (2021): Cross-sectional B(s), H(s), A(s) and median grain size D50(s)
  - Chen et al. (2022): Bankfull discharge Q_bf(t) at Gaocun and Sunkou stations
  - Niu (2024): Manning roughness coefficient n at Gaocun and Sunkou (2002-2020)
  - Reach slope: S = 1.16e-4 (cross-validated from survey profiles)

Computes derived physical parameters:
  beta (aspect ratio), U (mean velocity), Fr (Froude number),
  Cf (friction coefficient), Shields stress theta, and Reynolds number Re*.

Usage:
    python -m src.data.compile_hydraulic_params
"""
from __future__ import annotations

import argparse
import sys as _sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from src.config import (
    G, DELTA, RHO_S, RHO_W,
    RESULTS_DIR,
    GAOCUN_DAM_KM, SUNKOU_DAM_KM,
)

# Optional literature source directory (workspace only)
LIT_DATA_DIR = Path(_PROJECT_ROOT).parent / "文献" / "+数据收集"
if not LIT_DATA_DIR.exists():
    LIT_DATA_DIR = Path(_PROJECT_ROOT) / "data" / "literature_data"

# ═════════════════════════════════════════════════════════════════════════════
# Physical & Reach Constants
# ═════════════════════════════════════════════════════════════════════════════
S_REACH = 1.16e-4   # Reach energy slope (m/m), cross-validated from survey profiles


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """Read CSV with auto-detection of text encodings (utf-8-sig, gbk, gb18030)."""
    for enc in ('utf-8-sig', 'gbk', 'gb18030'):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeDecodeError(f'Failed to decode {path} using utf-8-sig/gbk/gb18030')


# ═════════════════════════════════════════════════════════════════════════════
# Data Loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_qin2025() -> pd.DataFrame:
    """Qin (2025): Reach-averaged bankfull width B_bf(t) and depth H_bf(t)."""
    fp = LIT_DATA_DIR / "秦2025" / "水面宽&水深.csv"
    if not fp.exists():
        fp = LIT_DATA_DIR / "2025" / "&.csv"
    df = _read_csv_auto(fp)
    df.columns = ["year", "B_bf_m", "H_bf_m"]
    df["year"] = df["year"].astype(int)
    return df.set_index("year").sort_index()


def load_chen2022_qbf() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chen et al. (2022): Bankfull discharge Q_bf(t) at Gaocun & Sunkou stations."""
    base = LIT_DATA_DIR / "Chen et al. (2022)"

    f_gc = base / "高村Qbf.csv" if (base / "高村Qbf.csv").exists() else base / "Qbf.csv"
    df_gc = _read_csv_auto(f_gc)
    df_gc.columns = ["year", "Qbf_gc_m3s"]
    df_gc["year"] = df_gc["year"].astype(int)

    f_sk = base / "孙口Qbf.csv" if (base / "孙口Qbf.csv").exists() else base / "Qbf.csv"
    df_sk = _read_csv_auto(f_sk)
    df_sk.columns = ["year", "Qbf_sk_m3s"]
    df_sk["year"] = df_sk["year"].astype(int)

    return df_gc.set_index("year"), df_sk.set_index("year")


def load_niu2024_manning() -> pd.DataFrame:
    """Niu (2024) Table 3: Manning roughness n (2002-2020) pre- and post-flood."""
    data = {
        2002: [0.0082, 0.0099, 0.0185, 0.0125],
        2003: [0.0218, 0.0137, 0.0190, 0.0172],
        2004: [0.0128, 0.0249, 0.0126, 0.0178],
        2005: [0.0138, 0.0181, 0.0126, 0.0285],
        2006: [0.0090, 0.0117, 0.0107, 0.0143],
        2007: [0.0128, 0.0134, 0.0118, 0.0118],
        2008: [0.0104, 0.0098, 0.0135, 0.0182],
        2009: [0.0162, 0.0144, 0.0151, 0.0194],
        2010: [0.0098, 0.0194, 0.0151, 0.0137],
        2011: [0.0146, 0.0126, 0.0136, 0.0146],
        2012: [0.0106, 0.0132, 0.0124, 0.0197],
        2013: [0.0127, 0.0192, 0.0126, 0.0154],
        2014: [0.0141, 0.0207, 0.0141, 0.0282],
        2015: [0.0186, 0.0235, 0.0115, 0.0271],
        2016: [0.0203, 0.0226, 0.0165, 0.0231],
        2017: [0.0145, 0.0191, 0.0179, 0.0221],
        2018: [0.0176, 0.0125, 0.0161, 0.0133],
        2019: [0.0156, 0.0151, 0.0115, 0.0282],
        2020: [0.0111, 0.0137, 0.0123, 0.0160],
    }
    df = pd.DataFrame.from_dict(
        data, orient="index",
        columns=["n_gc_pre", "n_gc_post", "n_sk_pre", "n_sk_post"],
    )
    df.index.name = "year"
    return df


def load_zhang2021_d50() -> Dict[str, pd.DataFrame]:
    """Zhang (2021): D50 along-stream distribution (main channel & floodplain)."""
    base = LIT_DATA_DIR / "张2021"
    result = {}
    for tag, fname in [("channel", "D50-主槽.csv"), ("floodplain", "D50-滩地.csv")]:
        fp = base / fname
        if not fp.exists():
            fp = base / f"D50-{tag}.csv"
        df = _read_csv_auto(fp)
        df.columns = ["dam_km", "D50_mm"]
        result[tag] = df
    return result


def load_zhang2021_spatial(variable: str, year: int) -> pd.DataFrame:
    """Zhang (2021): Cross-sectional geometry profiles along reach."""
    base = LIT_DATA_DIR / "张2021"
    fname = f"{variable}{year}.csv"
    df = _read_csv_auto(base / fname)
    df.columns = ["dam_km", "value"]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Parameter Derivation Engine
# ═════════════════════════════════════════════════════════════════════════════

def compute_derived(
    B: float | np.ndarray,
    H: float | np.ndarray,
    Q: float | np.ndarray,
    S: float,
    D50_m: float,
    n: float | np.ndarray | None = None,
) -> Dict[str, float | np.ndarray]:
    """Compute derived hydraulic, friction, and sediment transport parameters."""
    A = B * H
    U = Q / A
    beta = B / H

    # Froude number
    Fr = U / np.sqrt(G * H)

    # Friction coefficient (energy slope definition: Cf = g H S / U^2)
    Cf_energy = G * H * S / U**2

    # Manning-based friction coefficient
    R = H
    Cf_manning = np.nan
    if n is not None:
        C_chezy = (1.0 / n) * R**(1.0 / 6.0)
        Cf_manning = G / C_chezy**2

    # Shear velocity
    u_star = np.sqrt(G * H * S)

    # Particle Reynolds number
    nu = 1.0e-6
    Re_star = u_star * D50_m / nu

    # Dimensionless Shields stress
    tau_b = RHO_W * G * H * S
    Shields = tau_b / ((RHO_S - RHO_W) * G * D50_m)

    # Sediment mobility parameter psi
    psi = u_star**2 / (DELTA * G * D50_m)

    # Flow rate check via Manning formula
    Q_manning = np.nan
    if n is not None:
        Q_manning = (1.0 / n) * A * (R**(2.0 / 3.0)) * np.sqrt(S)

    return {
        "A_m2": A,
        "U_ms": U,
        "beta": beta,
        "Fr": Fr,
        "Cf_energy": Cf_energy,
        "Cf_manning": Cf_manning,
        "u_star_ms": u_star,
        "Re_star": Re_star,
        "Shields": Shields,
        "psi": psi,
        "Q_manning_m3s": Q_manning,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Compilation Pipelines
# ═════════════════════════════════════════════════════════════════════════════

def compile_timeseries_table() -> pd.DataFrame:
    """Compile reach-averaged annual parameter timeseries (2000-2021)."""
    df_bh = load_qin2025()
    df_qbf_gc, df_qbf_sk = load_chen2022_qbf()
    df_n = load_niu2024_manning()

    d50_data = load_zhang2021_d50()["channel"]
    mask = (d50_data["dam_km"] >= GAOCUN_DAM_KM) & (d50_data["dam_km"] <= SUNKOU_DAM_KM)
    D50_mm = d50_data.loc[mask, "D50_mm"].median()
    D50_m = D50_mm * 1e-3

    years = df_bh.index.tolist()
    rows = []

    for yr in years:
        B = df_bh.loc[yr, "B_bf_m"]
        H = df_bh.loc[yr, "H_bf_m"]

        q_gc = df_qbf_gc.loc[yr, "Qbf_gc_m3s"] if yr in df_qbf_gc.index else np.nan
        q_sk = df_qbf_sk.loc[yr, "Qbf_sk_m3s"] if yr in df_qbf_sk.index else np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            Q_bf = np.nanmean([q_gc, q_sk])

        if yr in df_n.index:
            n_vals = df_n.loc[yr].values
            n_mean = np.nanmean(n_vals)
        else:
            n_mean = np.nan

        derived = compute_derived(B, H, Q_bf, S_REACH, D50_m, n=n_mean)

        rows.append({
            "year": yr,
            "B_bf_m": round(B, 1),
            "H_bf_m": round(H, 3),
            "Qbf_gc_m3s": round(q_gc, 0) if not np.isnan(q_gc) else np.nan,
            "Qbf_sk_m3s": round(q_sk, 0) if not np.isnan(q_sk) else np.nan,
            "Qbf_reach_m3s": round(Q_bf, 0),
            "n_manning": round(n_mean, 4) if not np.isnan(n_mean) else np.nan,
            "S": S_REACH,
            "D50_mm": round(D50_mm, 3),
            "beta": round(derived["beta"], 1),
            "A_m2": round(derived["A_m2"], 0),
            "U_ms": round(derived["U_ms"], 3),
            "Fr": round(derived["Fr"], 4),
            "Cf_energy": round(derived["Cf_energy"], 6),
            "Cf_manning": round(derived["Cf_manning"], 6) if not np.isnan(derived["Cf_manning"]) else np.nan,
            "u_star_ms": round(derived["u_star_ms"], 4),
            "Re_star": round(derived["Re_star"], 2),
            "Shields": round(derived["Shields"], 2),
            "Q_manning_m3s": round(derived["Q_manning_m3s"], 0) if not np.isnan(derived["Q_manning_m3s"]) else np.nan,
        })

    return pd.DataFrame(rows)


def compile_spatial_table(year: int = 2016) -> pd.DataFrame:
    """Compile cross-sectional spatial hydraulic parameters along the reach for a given year."""
    df_b = load_zhang2021_spatial("平滩水面宽", year)
    df_h = load_zhang2021_spatial("平滩水深", year)
    df_a = load_zhang2021_spatial("平滩面积", year)

    d50_data = load_zhang2021_d50()["channel"]

    df = df_b.rename(columns={"value": "B_m"})
    df["H_m"] = df_h["value"]
    df["A_m2"] = df_a["value"]

    mask_reach = (df["dam_km"] >= GAOCUN_DAM_KM) & (df["dam_km"] <= SUNKOU_DAM_KM)
    df = df[mask_reach].copy().reset_index(drop=True)

    df["D50_mm"] = np.interp(df["dam_km"], d50_data["dam_km"], d50_data["D50_mm"])

    df_ts = compile_timeseries_table()
    if year in df_ts["year"].values:
        q_rep = df_ts.loc[df_ts["year"] == year, "Qbf_reach_m3s"].values[0]
        n_rep = df_ts.loc[df_ts["year"] == year, "n_manning"].values[0]
    else:
        q_rep = 5218.0
        n_rep = 0.0206

    D50_m_arr = df["D50_mm"].values * 1e-3
    B_arr = df["B_m"].values
    H_arr = df["H_m"].values

    derived = compute_derived(B_arr, H_arr, q_rep, S_REACH, D50_m_arr, n=n_rep)

    df["year"] = year
    df["Qbf_m3s"] = q_rep
    df["n_manning"] = n_rep
    df["S"] = S_REACH
    df["beta"] = np.round(derived["beta"], 1)
    df["U_ms"] = np.round(derived["U_ms"], 3)
    df["Fr"] = np.round(derived["Fr"], 4)
    df["Cf_energy"] = np.round(derived["Cf_energy"], 6)
    df["Cf_manning"] = np.round(derived["Cf_manning"], 6)
    df["u_star_ms"] = np.round(derived["u_star_ms"], 4)
    df["Re_star"] = np.round(derived["Re_star"], 2)
    df["Shields"] = np.round(derived["Shields"], 2)
    df["Q_manning_m3s"] = np.round(derived["Q_manning_m3s"], 0)

    cols = [
        "year", "dam_km", "B_m", "H_m", "A_m2", "D50_mm",
        "Qbf_m3s", "n_manning", "S",
        "beta", "U_ms", "Fr", "Cf_energy", "Cf_manning",
        "u_star_ms", "Re_star", "Shields", "Q_manning_m3s",
    ]
    return df[cols]


def print_diagnostic_summary(df_ts: pd.DataFrame, df_sp: pd.DataFrame) -> None:
    """Print comparative statistical summary of reach hydraulic parameters."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  HYDRAULIC PARAMETERS DIAGNOSTIC SUMMARY (GAOCUN–SUNKOU REACH)")
    print(f"{sep}\n")

    print("【1】 Temporal Evolution: Early vs. Recent Periods (2000-2021)")
    print("-" * 60)

    early = df_ts[df_ts["year"] <= 2003]
    recent = df_ts[df_ts["year"] >= 2011]

    for label, sub in [("Early Period (2000-2003)", early), ("Recent Period (2011-2021)", recent)]:
        print(f"\n  {label}:")
        print(f"    beta = B/H   : {sub['beta'].min():.1f} - {sub['beta'].max():.1f} (mean {sub['beta'].mean():.1f})")
        print(f"    Fr           : {sub['Fr'].min():.3f} - {sub['Fr'].max():.3f} (mean {sub['Fr'].mean():.3f})")
        print(f"    Cf (energy)  : {sub['Cf_energy'].min():.5f} - {sub['Cf_energy'].max():.5f} (mean {sub['Cf_energy'].mean():.5f})")
        print(f"    B_bf (m)     : {sub['B_bf_m'].min():.1f} - {sub['B_bf_m'].max():.1f}")
        print(f"    H_bf (m)     : {sub['H_bf_m'].min():.2f} - {sub['H_bf_m'].max():.2f}")
        print(f"    Q_bf (m3/s)  : {sub['Qbf_reach_m3s'].min():.0f} - {sub['Qbf_reach_m3s'].max():.0f}")

    valid = df_ts.dropna(subset=["Q_manning_m3s", "Qbf_reach_m3s"])
    if len(valid) > 0:
        print(f"\n\n【2】 Manning Flow Rate Consistency Check")
        print("-" * 60)
        ratio = valid["Q_manning_m3s"] / valid["Qbf_reach_m3s"]
        print(f"  Q_manning / Q_bf: {ratio.min():.2f} - {ratio.max():.2f} (mean {ratio.mean():.2f})")

    print(f"\n\n【3】 Spatial Variation along 2016 Reach (118 km, 26 Cross-Sections)")
    print("-" * 60)
    print(f"  beta range: {df_sp['beta'].min():.1f} - {df_sp['beta'].max():.1f} (median {df_sp['beta'].median():.1f})")
    print(f"  Fr   range: {df_sp['Fr'].min():.3f} - {df_sp['Fr'].max():.3f}")

    print(f"\n{sep}\n")


def main():
    import io
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="Compile and verify reach hydraulic parameters.")
    parser.add_argument("--no-save", action="store_true", help="Do not save CSVs")
    args = parser.parse_args()

    out_ts = RESULTS_DIR / "hydraulic_params_timeseries.csv"
    out_sp = RESULTS_DIR / "hydraulic_params_spatial_2016.csv"

    qin_fp = LIT_DATA_DIR / "秦2025" / "水面宽&水深.csv"
    chen_fp = LIT_DATA_DIR / "Chen et al. (2022)" / "高村Qbf.csv"
    raw_available = qin_fp.exists() and chen_fp.exists()

    if raw_available:
        print("Compiling annual timeseries from raw survey sources...")
        df_ts = compile_timeseries_table()
        print("Compiling 2016 spatial cross-sections from raw survey sources...")
        df_sp = compile_spatial_table(year=2016)

        if not args.no_save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            df_ts.to_csv(out_ts, index=False, float_format="%.6g")
            df_sp.to_csv(out_sp, index=False, float_format="%.6g")
            print(f"[OK] Saved timeseries -> {out_ts}")
            print(f"[OK] Saved spatial    -> {out_sp}")
    else:
        print("Raw literature sources not present locally; verifying pre-compiled release datasets:")
        print(f"  Loading: {out_ts}")
        print(f"  Loading: {out_sp}")
        df_ts = pd.read_csv(out_ts)
        df_sp = pd.read_csv(out_sp)

    print_diagnostic_summary(df_ts, df_sp)

    print("\n── First 5 rows of Annual Reach-Averaged Hydraulics ──")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4g}".format):
        print(df_ts.head().to_string(index=False))


if __name__ == "__main__":
    main()
