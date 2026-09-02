"""Step 3.2 — trunk B(s)/C(s) 。

 24 （2000-2023） trunk ：
  * FFT （ Welch ）
  * （B  C ）
  * B-C  & 
  * （e-folding & ）

  : results/trunks/Gaocun-Sunkou_{year}_trunk_0.csv
  : results/spectral/spectral_summary.csv          — 24 
         results/spectral/{year}_B_spectrum.csv         —  B 
         results/spectral/{year}_C_spectrum.csv         —  C 

 C&G  quantitative_relationships.py（fft_spectrum / dominant_wavelength
/ phase_difference_at_frequency / cross_correlation_lag / autocorr_length_scales）。


----
    python -m src.spectral.analyze_profiles          # 24
    python -m src.spectral.analyze_profiles --years 2016 2020
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import detrend as _scipy_detrend

# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUNK_DIR = PROJECT_ROOT / "results" / "trunks"
SPECTRAL_DIR = PROJECT_ROOT / "results" / "spectral"

# ---------------------------------------------------------------------------
# （adapted from C&G _as_1d_float / _valid_mask / _fill_nan_linear）
# ---------------------------------------------------------------------------

def _as_1d_float(a: np.ndarray) -> np.ndarray:
    """ 1-D float64。"""
    return np.asarray(a, dtype=float).ravel()


def _valid_mask(*arrs: np.ndarray) -> np.ndarray:
    """。"""
    if not arrs:
        raise ValueError("arrs must not be empty")
    m = np.ones_like(_as_1d_float(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(_as_1d_float(a))
    return m


def _fill_nan_linear(x: np.ndarray) -> np.ndarray:
    """ NaN。"""
    x = _as_1d_float(x).copy()
    if x.size == 0:
        return x
    m = np.isfinite(x)
    if np.all(m):
        return x
    idx = np.arange(x.size, dtype=float)
    if int(np.sum(m)) >= 2:
        x[~m] = np.interp(idx[~m], idx[m], x[m])
    else:
        x[~m] = float(np.nanmean(x[m])) if np.any(m) else 0.0
    return x


def _interpolate_uniform(s: np.ndarray, y: np.ndarray,
                         step_m: float | None = None,
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """。

    
    ------
    s :  (m)，（）
    y : 
    step_m : ，None  s 

    
    ------
    s_uni, y_uni, step : 、、
    """
    s = _as_1d_float(s)
    y = _as_1d_float(y)
# （）
    _, unique_idx = np.unique(s, return_index=True)
# np.unique  index， s
# pandas drop_duplicates
    mask_keep = np.zeros(s.size, dtype=bool)
    seen = {}
    for i in range(s.size):
        seen[s[i]] = i
    mask_keep[list(seen.values())] = True
    s = s[mask_keep]
    y = y[mask_keep]
# 
    order = np.argsort(s)
    s = s[order]
    y = y[order]
    if step_m is None:
        ds = np.diff(s)
        step_m = float(np.median(ds[ds > 0]))
    s_uni = np.arange(s[0], s[-1], step_m)
    y_uni = np.interp(s_uni, s, y)
    return s_uni, y_uni, step_m


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

def fft_spectrum(x: np.ndarray, step_m: float,
                 detrend: bool = True) -> Dict[str, np.ndarray]:
    """ FFT 。

    
    ------
    x : 1-D （）
    step_m :  (m)
    detrend : （）， True

    
    ------
    {"freq":  (1/m), "amp": , "phase":  (rad),
     "psd":  (amp^2 * step_m / N)}
    """
    x = _as_1d_float(x)
    x = x[np.isfinite(x)]
    if x.size < 4:
        empty = np.array([])
        return {"freq": empty, "amp": empty, "phase": empty, "psd": empty}
    if detrend:
        x = _scipy_detrend(x, type='linear')
    N = x.size
    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(N, d=float(step_m))
    amp = np.abs(X)
    phase = np.angle(X)
# : |X|^2 / (N * df), df = 1/(N*step_m)
    psd = (amp ** 2) * float(step_m) / float(N)
# （ DC  Nyquist  ×2）
    if psd.size > 2:
        psd[1:-1] *= 2.0
        amp[1:-1] *= np.sqrt(2.0)
    return {"freq": freq, "amp": amp, "phase": phase, "psd": psd}


def dominant_wavelength(x: np.ndarray, step_m: float,
                        n_peaks: int = 3,
                        max_wavelength_m: float | None = None,
                        ) -> Dict[str, object]:
    """（ n_peaks ）。

    
    ------
    x : 1-D 
    step_m :  (m)
    n_peaks :  n 
    max_wavelength_m :  (m)，；
        None  1/3（）

    
    ------
    {"lambda_m": , "freq": , "amp": , "phase": ,
     "top_lambdas_m": array of top-n ,
     "top_freqs": array, "top_amps": array}
    """
    spec = fft_spectrum(x, step_m=step_m, detrend=True)
    freq = spec["freq"]
    amp = spec["amp"]
    phase = spec["phase"]
    if freq.size < 3:
        nan = float("nan")
        return {"lambda_m": nan, "freq": nan, "amp": nan, "phase": nan,
                "top_lambdas_m": np.array([]), "top_freqs": np.array([]),
                "top_amps": np.array([])}
# （ L/3）
    signal_length = float(x.size * step_m) if np.isfinite(x).sum() > 1 else 0.0
    if max_wavelength_m is None and signal_length > 0:
        max_wavelength_m = signal_length / 3.0
# 
    min_freq = 1.0 / max_wavelength_m if (max_wavelength_m and max_wavelength_m > 0) else 0.0
# DC
    amp_no_dc = amp[1:].copy()
    freq_no_dc = freq[1:]
    phase_no_dc = phase[1:]
# （）
    valid_freq = freq_no_dc >= min_freq
    if not np.any(valid_freq):
# ，
        valid_freq = np.ones_like(freq_no_dc, dtype=bool)
    amp_valid = amp_no_dc.copy()
    amp_valid[~valid_freq] = 0.0  # ，
# n_peaks （）
    n_avail = min(n_peaks, amp_valid.size)
    top_idx = np.argsort(amp_valid)[::-1][:n_avail]
    top_idx_sorted = top_idx[np.argsort(freq_no_dc[top_idx])]
# 
    peak_idx = int(np.argmax(amp_valid))
    f_peak = float(freq_no_dc[peak_idx])
    lam = float(1.0 / f_peak) if f_peak > 0 else float("nan")
    top_freqs = freq_no_dc[top_idx_sorted]
    top_lams = np.where(top_freqs > 0, 1.0 / top_freqs, np.nan)
    return {
        "lambda_m": lam,
        "freq": f_peak,
        "amp": float(amp_no_dc[peak_idx]),  # ，
        "phase": float(phase_no_dc[peak_idx]),
        "top_lambdas_m": top_lams,
        "top_freqs": top_freqs,
        "top_amps": amp_no_dc[top_idx_sorted],
    }


def phase_difference_at_frequency(
    x: np.ndarray, y: np.ndarray, step_m: float, freq: float,
) -> float:
    """ ()。

     phi_y - phi_x， [-180, 180]。
    """
    sx = fft_spectrum(x, step_m=step_m, detrend=True)
    sy = fft_spectrum(y, step_m=step_m, detrend=True)
    fx = sx["freq"]
    if fx.size == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(fx - float(freq))))
    dphi = float(sy["phase"][idx]) - float(sx["phase"][idx])
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.degrees(dphi))


def cross_correlation_lag(
    x: np.ndarray, y: np.ndarray, step_m: float,
    max_lag_m: float | None = None,
) -> Dict[str, float]:
    """。

    
    ------
    x, y :  1-D 
    step_m :  (m)
    max_lag_m : 

    
    ------
    {"lag_m":  (m), "corr": }
    """
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    m = _valid_mask(x, y)
    x, y = x[m], y[m]
    if x.size < 4:
        return {"lag_m": float("nan"), "corr": float("nan")}
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    ccf = np.correlate(x, y, mode="full")
    lags = np.arange(-x.size + 1, x.size) * float(step_m)
    if max_lag_m is not None:
        mm = np.abs(lags) <= float(max_lag_m)
        ccf, lags = ccf[mm], lags[mm]
    denom = float(np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)))
    ccf_norm = ccf / denom if denom > 0 else ccf
    idx = int(np.argmax(ccf_norm))
    return {"lag_m": float(lags[idx]), "corr": float(ccf_norm[idx])}


def autocorr_length_scales(x: np.ndarray, step_m: float) -> Dict[str, float]:
    """。

    
    ------
    {"e_folding_m": e-, "integral_scale_m": , "n": }
    """
    x = _as_1d_float(x)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return {"e_folding_m": float("nan"),
                "integral_scale_m": float("nan"), "n": int(x.size)}
    x = x - float(np.mean(x))
    var = float(np.sum(x ** 2))
    if var <= 0:
        return {"e_folding_m": float("nan"),
                "integral_scale_m": float("nan"), "n": int(x.size)}
    acf = np.correlate(x, x, mode="full")[x.size - 1:]
    acf = acf / float(acf[0])
# e-folding
    thr = float(np.exp(-1.0))
    below = np.where(acf <= thr)[0]
    e_fold = float(below[0]) * float(step_m) if below.size > 0 else float("nan")
# （）
    neg = np.where(acf < 0)[0]
    kmax = int(neg[0]) if neg.size > 0 else int(acf.size)
    integral = float(np.sum(acf[1:kmax])) * float(step_m) if kmax > 1 else float("nan")
    return {"e_folding_m": e_fold, "integral_scale_m": integral, "n": int(x.size)}


# ---------------------------------------------------------------------------
# ： trunk
# ---------------------------------------------------------------------------

@dataclass
class SpectralResult:
    """ trunk 。"""
    year: int
    trunk_length_km: float
    n_points: int
    step_m: float
# B(s)
    B_lambda_m: float       # B
    B_freq: float           # B
    B_amp: float            # B
    B_efold_m: float        # B e-folding
    B_integral_m: float     # B
    B_mean: float           # B
    B_std: float            # B
# C(s)
    C_lambda_m: float       # C
    C_freq: float           # C
    C_amp: float            # C
    C_efold_m: float        # C e-folding
    C_integral_m: float     # C
    C_mean: float           # C  ( 0)
    C_std: float            # C
# B-C
    BC_phase_deg: float     # C  B-C  ()
    BC_lag_m: float         # B-C  (m)
    BC_corr: float          # B-C


def analyze_single_trunk(csv_path: Path | str,
                         step_m: float | None = None,
                         max_lag_km: float = 30.0,
                         ) -> SpectralResult:
    """ trunk CSV 。

    
    ------
    csv_path : trunk CSV  (columns: s_m, lon, lat, B_m, C_1m)
    step_m :  (m)，None 
    max_lag_km :  (km)

    
    ------
    SpectralResult dataclass
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
# 
    stem = csv_path.stem  # e.g. Gaocun-Sunkou_2016_trunk_0
    year = int(stem.split("_")[1])

    s_raw = np.asarray(df["s_m"].values, dtype=float)
    B_raw = np.asarray(df["B_m"].values, dtype=float)
    C_raw = np.asarray(df["C_1m"].values, dtype=float)

# 
    s_B, B_uni, step = _interpolate_uniform(s_raw, B_raw, step_m)
    _, C_uni, _ = _interpolate_uniform(s_raw, C_raw, step)

# （）
    n = min(B_uni.size, C_uni.size)
    B_uni, C_uni = B_uni[:n], C_uni[:n]
    s_uni = s_B[:n]

    trunk_km = float(s_uni[-1] - s_uni[0]) / 1000.0 if n > 1 else 0.0

# B
    domB = dominant_wavelength(B_uni, step_m=step)
    scB = autocorr_length_scales(B_uni, step_m=step)

# C
    domC = dominant_wavelength(C_uni, step_m=step)
    scC = autocorr_length_scales(C_uni, step_m=step)

# B-C
# C  B-C
    freq_C = float(domC["freq"])
    if np.isfinite(freq_C) and freq_C > 0:
        bc_phase = phase_difference_at_frequency(
            C_uni, B_uni, step_m=step, freq=freq_C)
    else:
        bc_phase = float("nan")

    bc_lag = cross_correlation_lag(
        B_uni, C_uni, step_m=step,
        max_lag_m=max_lag_km * 1000.0)

    return SpectralResult(
        year=year,
        trunk_length_km=trunk_km,
        n_points=n,
        step_m=step,
        B_lambda_m=float(domB["lambda_m"]),
        B_freq=float(domB["freq"]),
        B_amp=float(domB["amp"]),
        B_efold_m=float(scB["e_folding_m"]),
        B_integral_m=float(scB["integral_scale_m"]),
        B_mean=float(np.nanmean(B_uni)),
        B_std=float(np.nanstd(B_uni)),
        C_lambda_m=float(domC["lambda_m"]),
        C_freq=float(domC["freq"]),
        C_amp=float(domC["amp"]),
        C_efold_m=float(scC["e_folding_m"]),
        C_integral_m=float(scC["integral_scale_m"]),
        C_mean=float(np.nanmean(C_uni)),
        C_std=float(np.nanstd(C_uni)),
        BC_phase_deg=float(bc_phase),
        BC_lag_m=float(bc_lag["lag_m"]),
        BC_corr=float(bc_lag["corr"]),
    )


def _save_spectrum_csv(spec: Dict[str, np.ndarray], out_path: Path,
                       label: str) -> None:
    """ FFT  CSV。"""
    if spec["freq"].size == 0:
        return
    df = pd.DataFrame({
        "freq_1m": spec["freq"],
        "wavelength_m": np.where(spec["freq"] > 0, 1.0 / spec["freq"], np.inf),
        f"{label}_amp": spec["amp"],
        f"{label}_psd": spec["psd"],
        f"{label}_phase_rad": spec["phase"],
    })
    df.to_csv(out_path, index=False, float_format="%.6g")


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

def analyze_all_trunks(
    trunk_dir: Path | None = None,
    out_dir: Path | None = None,
    years: List[int] | None = None,
    step_m: float | None = None,
    save_spectra: bool = True,
) -> pd.DataFrame:
    """ trunk 。

    
    ------
    trunk_dir : trunk CSV 
    out_dir : 
    years : ，None=
    step_m : 
    save_spectra :  CSV

    
    ------
    DataFrame with one row per year, columns from SpectralResult fields
    """
    trunk_dir = trunk_dir or TRUNK_DIR
    out_dir = out_dir or SPECTRAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

# trunk
    pattern = "Gaocun-Sunkou_*_trunk_0.csv"
    csvs = sorted(trunk_dir.glob(pattern))
    if not csvs:
        raise FileNotFoundError(
            f"No trunk CSVs found in {trunk_dir} matching {pattern}")

    if years:
        year_set = set(years)
        csvs = [p for p in csvs
                if int(p.stem.split("_")[1]) in year_set]

    rows: list[dict[str, object]] = []
    for csv_path in csvs:
        year = int(csv_path.stem.split("_")[1])
        print(f"  [{year}] analyzing {csv_path.name} ...", flush=True)
        try:
            result = analyze_single_trunk(csv_path, step_m=step_m)
            rows.append({f.name: getattr(result, f.name)
                         for f in fields(result)})

# 
            if save_spectra:
                df_raw = pd.read_csv(csv_path)
                s_raw = np.asarray(df_raw["s_m"].values, dtype=float)
                _, B_uni, st = _interpolate_uniform(
                    s_raw, np.asarray(df_raw["B_m"].values, dtype=float),
                    step_m)
                _, C_uni, _ = _interpolate_uniform(
                    s_raw, np.asarray(df_raw["C_1m"].values, dtype=float),
                    st)
                spec_B = fft_spectrum(B_uni, step_m=st)
                spec_C = fft_spectrum(C_uni, step_m=st)
                _save_spectrum_csv(
                    spec_B, out_dir / f"{year}_B_spectrum.csv", "B")
                _save_spectrum_csv(
                    spec_C, out_dir / f"{year}_C_spectrum.csv", "C")

        except Exception as exc:
            print(f"  [{year}] ERROR: {exc}", flush=True)
# NaN
            row = {"year": year}
            for f in fields(SpectralResult):
                if f.name != "year":
                    row[f.name] = float("nan") if f.type != "int" else 0
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("year").reset_index(drop=True)
    summary_path = out_dir / "spectral_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"\nSummary saved to {summary_path}", flush=True)
    print(f"  {len(rows)} years analyzed, "
          f"{summary['B_lambda_m'].notna().sum()} with valid spectra",
          flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """。"""
# Windows UTF-8
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        description="Spectral analysis of trunk B(s)/C(s) profiles")
    parser.add_argument("--years", nargs="*", type=int, default=None,
                        help="Year(s) to analyze; default=all")
    parser.add_argument("--step-m", type=float, default=None,
                        help="Uniform sampling step (m); default=median")
    parser.add_argument("--no-spectra", action="store_true",
                        help="Skip per-year spectrum CSV output")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("Step 3.2: Spectral Analysis of Trunk Profiles", flush=True)
    print("=" * 60, flush=True)

    summary = analyze_all_trunks(
        years=args.years,
        step_m=args.step_m,
        save_spectra=not args.no_spectra,
    )

# 
    print("\n--- Summary Statistics ---", flush=True)
    for col in ["B_lambda_m", "C_lambda_m", "BC_lag_m", "BC_corr",
                "B_efold_m", "C_efold_m", "B_mean", "B_std"]:
        if col in summary.columns:
            vals = summary[col].dropna()
            if vals.size > 0:
                print(f"  {col:18s}: "
                      f"mean={vals.mean():10.2f}  "
                      f"std={vals.std():10.2f}  "
                      f"range=[{vals.min():.2f}, {vals.max():.2f}]",
                      flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()