"""-。

：
  - 2025: B_bf(t), H_bf(t)  (2000-2021)
  - 2021: B(s), H(s), √B/H(s), A(s) 4; D50(s) /
  - Chen et al. (2022): Q_bf(t) / (1952-2020 / 1964-2020)
  - (2024): Manning n / / (2002-2020)
  - : S = 1.16e-4 (，-)

：β, U, Fr, Cf, Shields θ, ψ

：
    conda activate riverpiv
    python -m src.data.compile_hydraulic_params
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import pandas as pd

# python -m src.data.compile_hydraulic_params  python src/data/compile_hydraulic_params.py
import sys as _sys
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from src.config import (
    G, DELTA, RHO_S, RHO_W,
    LIT_DATA_DIR, RESULTS_DIR,
    GAOCUN_DAM_KM, SUNKOU_DAM_KM,
)

# ══════════════════════════════════════════════════════════════
# 
# ══════════════════════════════════════════════════════════════
S_REACH = 1.16e-4   # - (m/m)， + 2021


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """ utf-8-sig → gbk  CSV。"""
    for enc in ('utf-8-sig', 'gbk', 'gb18030'):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeDecodeError(f' {path}， utf-8-sig/gbk/gb18030')


# ══════════════════════════════════════════════════════════════
# 
# ══════════════════════════════════════════════════════════════

def load_qin2025() -> pd.DataFrame:
    """2025:  B_bf(t), H_bf(t)。"""
    fp = LIT_DATA_DIR / "2025" / "&.csv"
    df = _read_csv_auto(fp)
    df.columns = ["year", "B_bf_m", "H_bf_m"]
    df["year"] = df["year"].astype(int)
    return df.set_index("year").sort_index()


def load_chen2022_qbf() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Chen et al. (2022): Q_bf(t)  & 。"""
    base = LIT_DATA_DIR / "Chen et al. (2022)"

    df_gc = _read_csv_auto(base / "Qbf.csv")
    df_gc.columns = ["year", "Qbf_gc_m3s"]
    df_gc["year"] = df_gc["year"].astype(int)

    df_sk = _read_csv_auto(base / "Qbf.csv")
    df_sk.columns = ["year", "Qbf_sk_m3s"]
    df_sk["year"] = df_sk["year"].astype(int)

    return df_gc.set_index("year"), df_sk.set_index("year")


def load_niu2024_manning() -> pd.DataFrame:
    """(2024) Table 3: Manning n (2002-2020)。

     DataFrame, index=year, columns:
        n_gc_pre, n_gc_post, n_sk_pre, n_sk_post
    """
# look_at
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
    """2021: D50 （ & ）。"""
    base = LIT_DATA_DIR / "2021"
    result = {}
    for tag, fname in [("channel", "D50-.csv"), ("floodplain", "D50-.csv")]:
        df = _read_csv_auto(base / fname)
        df.columns = ["dam_km", "D50_mm"]
        result[tag] = df
    return result


def load_zhang2021_spatial(variable: str, year: int) -> pd.DataFrame:
    """2021 。

    
    ------
    variable : str
        "" | "" | "" | ""
    year : int
        2000 | 2005 | 2011 | 2016
    """
    base = LIT_DATA_DIR / "2021"
    fname = f"{variable}{year}.csv"
    df = _read_csv_auto(base / fname)
# CSV ，
    cols = df.columns.tolist()
    df.columns = ["dam_km", "value"]
    return df


# ══════════════════════════════════════════════════════════════
# 
# ══════════════════════════════════════════════════════════════

def compute_derived(
    B: float | np.ndarray,
    H: float | np.ndarray,
    Q: float | np.ndarray,
    S: float,
    D50_m: float,
    n: float | np.ndarray | None = None,
) -> Dict[str, float | np.ndarray]:
    """。

    
    ------
    B :  (m)
    H :  (m)
    Q :  (m³/s)
    S :  (-)
    D50_m :  (m)
    n : Manning  (-)，

    
    ------
    dict :  beta, A, U, Fr, Cf_energy, Cf_manning, Re_star, Shields, psi
    """
    A = B * H
    U = Q / A
    beta = B / H

# Froude
    Fr = U / np.sqrt(G * H)

# —  ()
    Cf_energy = G * H * S / U**2

# — Manning  ()
    R = H  # R ≈ H
    Cf_manning = np.nan
    if n is not None:
        C_chezy = (1.0 / n) * R**(1.0/6.0)
        Cf_manning = G / C_chezy**2

# 
    u_star = np.sqrt(G * H * S)

# Reynolds
    nu = 1.0e-6  # (m²/s)
    Re_star = u_star * D50_m / nu

# Shields
    tau_b = RHO_W * G * H * S
    Shields = tau_b / ((RHO_S - RHO_W) * G * D50_m)

# Manning （）
    Q_manning = np.nan
    if n is not None:
        U_manning = (1.0 / n) * R**(2.0/3.0) * S**0.5
        Q_manning = U_manning * A

    return {
        "beta": beta,
        "A_m2": A,
        "U_ms": U,
        "Fr": Fr,
        "Cf_energy": Cf_energy,
        "Cf_manning": Cf_manning,
        "u_star_ms": u_star,
        "Re_star": Re_star,
        "Shields": Shields,
        "Q_manning_m3s": Q_manning,
    }


# ══════════════════════════════════════════════════════════════
# ：
# ══════════════════════════════════════════════════════════════

def compile_timeseries_table() -> pd.DataFrame:
    """ (2000-2021)。

    ：
    - B, H: 2025 (2000-2021, , 15)
    - Q_bf: 
    - n: 
    - S, D50:  ()
    """
# 
    df_bh = load_qin2025()
    df_qbf_gc, df_qbf_sk = load_chen2022_qbf()
    df_n = load_niu2024_manning()

# D50: - (2021)
    d50_data = load_zhang2021_d50()["channel"]
    mask = (d50_data["dam_km"] >= GAOCUN_DAM_KM) & (d50_data["dam_km"] <= SUNKOU_DAM_KM)
    D50_mm = d50_data.loc[mask, "D50_mm"].median()
    D50_m = D50_mm * 1e-3

# 2025
    years = df_bh.index.tolist()
    rows = []

    for yr in years:
        B = df_bh.loc[yr, "B_bf_m"]
        H = df_bh.loc[yr, "H_bf_m"]

# Q_bf: /；
        q_gc = df_qbf_gc.loc[yr, "Qbf_gc_m3s"] if yr in df_qbf_gc.index else np.nan
        q_sk = df_qbf_sk.loc[yr, "Qbf_sk_m3s"] if yr in df_qbf_sk.index else np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            Q_bf = np.nanmean([q_gc, q_sk])

# Manning n: /
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

    df = pd.DataFrame(rows)
    return df


def compile_spatial_table(year: int = 2016) -> pd.DataFrame:
    """ ()。

    2021 B(s), H(s) 。
    Q_bf  n 。
    """
    df_b = load_zhang2021_spatial("", year)
    df_h = load_zhang2021_spatial("", year)

# （ dam_km）
    d50_data = load_zhang2021_d50()["channel"]

# Q_bf
    df_qbf_gc, df_qbf_sk = load_chen2022_qbf()
    q_gc = df_qbf_gc.loc[year, "Qbf_gc_m3s"] if year in df_qbf_gc.index else np.nan
    q_sk = df_qbf_sk.loc[year, "Qbf_sk_m3s"] if year in df_qbf_sk.index else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        Q_reach = np.nanmean([q_gc, q_sk])

# Manning n
    df_n = load_niu2024_manning()
    if year in df_n.index:
        n_mean = np.nanmean(df_n.loc[year].values)
    else:
        n_mean = np.nan

# D50 —
    def get_d50_at_km(km: float) -> float:
        dists = np.abs(d50_data["dam_km"].values - km)
        mask = dists < 15.0  # 15km
        if mask.any():
            return d50_data.loc[mask, "D50_mm"].median()
        return np.nan

# （BH，B）
    rows = []
    for _, row_b in df_b.iterrows():
        km = row_b["dam_km"]
        B = row_b["value"]

# H
        dists_h = np.abs(df_h["dam_km"].values - km)
        idx_h = np.argmin(dists_h)
        if dists_h[idx_h] > 5.0:  # 5km
            continue
        H = df_h.iloc[idx_h]["value"]

        D50_mm = get_d50_at_km(km)
        D50_m = D50_mm * 1e-3 if not np.isnan(D50_mm) else 0.075e-3

        derived = compute_derived(B, H, Q_reach, S_REACH, D50_m, n=n_mean)

        rows.append({
            "dam_km": round(km, 1),
            "B_m": round(B, 1),
            "H_m": round(H, 3),
            "D50_mm": round(D50_mm, 3) if not np.isnan(D50_mm) else np.nan,
            "beta": round(derived["beta"], 1),
            "U_ms": round(derived["U_ms"], 3),
            "Fr": round(derived["Fr"], 4),
            "Cf_energy": round(derived["Cf_energy"], 6),
            "Shields": round(derived["Shields"], 2),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 
# ══════════════════════════════════════════════════════════════

def print_diagnostic_summary(df_ts: pd.DataFrame, df_sp: pd.DataFrame) -> None:
    """。"""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  -")
    print(f"{sep}\n")

# ──  ──
    print("【1】 ·  (2000-2021)")
    print("-" * 60)

# 
    early = df_ts[df_ts["year"] <= 2003]
    recent = df_ts[df_ts["year"] >= 2011]

    for label, sub in [(" (2000-2003)", early), (" (2011-2021)", recent)]:
        print(f"\n  {label}:")
        print(f"    beta = B/H   : {sub['beta'].min():.0f} - {sub['beta'].max():.0f}"
              f"  ( {sub['beta'].mean():.0f})")
        print(f"    Fr           : {sub['Fr'].min():.3f} - {sub['Fr'].max():.3f}"
              f"  ( {sub['Fr'].mean():.3f})")
        print(f"    Cf (energy)  : {sub['Cf_energy'].min():.5f} - {sub['Cf_energy'].max():.5f}"
              f"  ( {sub['Cf_energy'].mean():.5f})")
        print(f"    B_bf (m)     : {sub['B_bf_m'].min():.0f} - {sub['B_bf_m'].max():.0f}")
        print(f"    H_bf (m)     : {sub['H_bf_m'].min():.2f} - {sub['H_bf_m'].max():.2f}")
        print(f"    Q_bf (m3/s)  : {sub['Qbf_reach_m3s'].min():.0f} - {sub['Qbf_reach_m3s'].max():.0f}")

# ── Manning  ──
    valid = df_ts.dropna(subset=["Q_manning_m3s", "Qbf_reach_m3s"])
    if len(valid) > 0:
        print(f"\n\n【2】Manning ")
        print("-" * 60)
        ratio = valid["Q_manning_m3s"] / valid["Qbf_reach_m3s"]
        print(f"  Q_manning / Q_bf: {ratio.min():.2f} - {ratio.max():.2f}"
              f"  (mean {ratio.mean():.2f})")
        print(f"   = 1.0； n  B/H ")

# ──  ──
    print(f"\n\n【3】 (2016)")
    print("-" * 60)
    gc_sp = df_sp[df_sp["dam_km"] <= GAOCUN_DAM_KM + 5]
    sk_sp = df_sp[(df_sp["dam_km"] >= SUNKOU_DAM_KM - 5) & (df_sp["dam_km"] <= SUNKOU_DAM_KM + 5)]
    print(f"  beta range: {df_sp['beta'].min():.0f} - {df_sp['beta'].max():.0f}"
          f"  ( {df_sp['beta'].median():.0f})")
    print(f"  Fr   range: {df_sp['Fr'].min():.3f} - {df_sp['Fr'].max():.3f}")
    if len(gc_sp) > 0:
        print(f"  : β={gc_sp['beta'].values[0]:.0f}, Fr={gc_sp['Fr'].values[0]:.3f}")
    if len(sk_sp) > 0:
        print(f"  : β={sk_sp['beta'].values[0]:.0f}, Fr={sk_sp['Fr'].values[0]:.3f}")

# ──  Sub1  ──
    print(f"\n\n【4】1")
    print("-" * 60)
    print(f"  Sub1 Termini S1:  β=16.7,  Fr=0.73,  Cf=0.0050")
    print(f"  Sub1 Termini S2:  β=9.1,   Fr=0.90,  Cf=0.0042")
    print(f"  Sub1 Van Dijk:    β=20.0,  Fr=0.58,  Cf=0.0050")
    print(f"  {'-'*45}")
    latest = df_ts.dropna(subset=['Fr', 'Cf_energy']).iloc[-1]
    print(f"   ():    β={latest['beta']:.0f},  Fr={latest['Fr']:.3f},"
          f"  Cf={latest['Cf_energy']:.4f}")
    print(f"  ** beta :  = Sub1 x {latest['beta']/15:.0f} ")
    print(f"  ** Fr :  = Sub1 x {latest['Fr']/0.7:.1f} ")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
# stdout  utf-8， Windows GBK
    import io
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="-")
    parser.add_argument("--no-save", action="store_true", help="，CSV")
    args = parser.parse_args()

    print("...")
    df_ts = compile_timeseries_table()

    print(" (2016)...")
    df_sp = compile_spatial_table(year=2016)

# 
    print_diagnostic_summary(df_ts, df_sp)

# 
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_ts = RESULTS_DIR / "hydraulic_params_timeseries.csv"
        out_sp = RESULTS_DIR / "hydraulic_params_spatial_2016.csv"

        df_ts.to_csv(out_ts, index=False, float_format="%.6g")
        df_sp.to_csv(out_sp, index=False, float_format="%.6g")

        print(f"[OK]  -> {out_ts}")
        print(f"[OK]  -> {out_sp}")

# 
    print("\n──  ──")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4g}".format):
        print(df_ts.to_string(index=False))


if __name__ == "__main__":
    main()