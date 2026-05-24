"""Step 3.2 — trunk B(s)/C(s) 剖面的谱分析模块。

对 24 年（2000-2023）主河道 trunk 数据执行：
  * FFT 功率谱（含 Welch 平滑可选）
  * 主波长提取（B 和 C 的峰值频率）
  * B-C 互相关滞后 & 互谱相位差
  * 自相关长度尺度（e-folding & 积分尺度）

输入  : results/trunks/Gaocun-Sunkou_{year}_trunk_0.csv
输出  : results/spectral/spectral_summary.csv          — 24 行年度汇总
         results/spectral/{year}_B_spectrum.csv         — 单年 B 功率谱
         results/spectral/{year}_C_spectrum.csv         — 单年 C 功率谱

适配自 C&G 项目 quantitative_relationships.py（fft_spectrum / dominant_wavelength
/ phase_difference_at_frequency / cross_correlation_lag / autocorr_length_scales）。

用法
----
    python -m src.spectral.analyze_profiles          # 分析全部 24 年
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
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUNK_DIR = PROJECT_ROOT / "results" / "trunks"
SPECTRAL_DIR = PROJECT_ROOT / "results" / "spectral"

# ---------------------------------------------------------------------------
# 辅助函数（adapted from C&G _as_1d_float / _valid_mask / _fill_nan_linear）
# ---------------------------------------------------------------------------

def _as_1d_float(a: np.ndarray) -> np.ndarray:
    """将任意数组转为 1-D float64。"""
    return np.asarray(a, dtype=float).ravel()


def _valid_mask(*arrs: np.ndarray) -> np.ndarray:
    """返回所有输入数组中均为有限值位置的布尔掩膜。"""
    if not arrs:
        raise ValueError("arrs must not be empty")
    m = np.ones_like(_as_1d_float(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(_as_1d_float(a))
    return m


def _fill_nan_linear(x: np.ndarray) -> np.ndarray:
    """线性插值填充 NaN。"""
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
    """将非均匀采样数据插值到等间距网格。

    参数
    ------
    s : 沿程坐标 (m)，单调递增（允许重复）
    y : 对应信号值
    step_m : 目标间距，None 时使用 s 的中位间距

    返回
    ------
    s_uni, y_uni, step : 均匀坐标、插值信号、实际间距
    """
    s = _as_1d_float(s)
    y = _as_1d_float(y)
    # 去除重复坐标（保留最后一个）
    _, unique_idx = np.unique(s, return_index=True)
    # np.unique 返回排序后的 index，但原始 s 可能有重复
    # 使用 pandas drop_duplicates 保留最后出现的
    mask_keep = np.zeros(s.size, dtype=bool)
    seen = {}
    for i in range(s.size):
        seen[s[i]] = i
    mask_keep[list(seen.values())] = True
    s = s[mask_keep]
    y = y[mask_keep]
    # 确保单调递增
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
# 核心谱函数
# ---------------------------------------------------------------------------

def fft_spectrum(x: np.ndarray, step_m: float,
                 detrend: bool = True) -> Dict[str, np.ndarray]:
    """计算实信号的单边 FFT 功率谱。

    参数
    ------
    x : 1-D 信号（均匀采样）
    step_m : 采样间距 (m)
    detrend : 是否线性去趋势（消除沿程线性变化），默认 True

    返回
    ------
    {"freq": 空间频率 (1/m), "amp": 幅度, "phase": 相位 (rad),
     "psd": 功率谱密度 (amp^2 * step_m / N)}
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
    # 功率谱密度: |X|^2 / (N * df), df = 1/(N*step_m)
    psd = (amp ** 2) * float(step_m) / float(N)
    # 单边谱修正（除 DC 和 Nyquist 外 ×2）
    if psd.size > 2:
        psd[1:-1] *= 2.0
        amp[1:-1] *= np.sqrt(2.0)
    return {"freq": freq, "amp": amp, "phase": phase, "psd": psd}


def dominant_wavelength(x: np.ndarray, step_m: float,
                        n_peaks: int = 3,
                        max_wavelength_m: float | None = None,
                        ) -> Dict[str, object]:
    """提取信号的主波长（及前 n_peaks 个峰）。

    参数
    ------
    x : 1-D 信号
    step_m : 采样间距 (m)
    n_peaks : 返回前 n 个峰
    max_wavelength_m : 最大允许波长 (m)，超过此值的峰被忽略；
        None 时取信号长度的 1/3（防止低频伪峰）

    返回
    ------
    {"lambda_m": 主波长, "freq": 主频率, "amp": 主幅度, "phase": 主相位,
     "top_lambdas_m": array of top-n 波长,
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
    # 最大波长截断（默认 L/3）
    signal_length = float(x.size * step_m) if np.isfinite(x).sum() > 1 else 0.0
    if max_wavelength_m is None and signal_length > 0:
        max_wavelength_m = signal_length / 3.0
    # 最小频率截断
    min_freq = 1.0 / max_wavelength_m if (max_wavelength_m and max_wavelength_m > 0) else 0.0
    # 排除 DC 和低于最小频率的分量
    amp_no_dc = amp[1:].copy()
    freq_no_dc = freq[1:]
    phase_no_dc = phase[1:]
    # 应用最小频率截断（屏蔽超长波长伪峰）
    valid_freq = freq_no_dc >= min_freq
    if not np.any(valid_freq):
        # 所有分量都低于截断频率，回退到无截断
        valid_freq = np.ones_like(freq_no_dc, dtype=bool)
    amp_valid = amp_no_dc.copy()
    amp_valid[~valid_freq] = 0.0  # 低频分量幅值置零，不参与峰值搜索
    # 前 n_peaks 个最大幅度（基于截断后的幅值）
    n_avail = min(n_peaks, amp_valid.size)
    top_idx = np.argsort(amp_valid)[::-1][:n_avail]
    top_idx_sorted = top_idx[np.argsort(freq_no_dc[top_idx])]
    # 最大峰
    peak_idx = int(np.argmax(amp_valid))
    f_peak = float(freq_no_dc[peak_idx])
    lam = float(1.0 / f_peak) if f_peak > 0 else float("nan")
    top_freqs = freq_no_dc[top_idx_sorted]
    top_lams = np.where(top_freqs > 0, 1.0 / top_freqs, np.nan)
    return {
        "lambda_m": lam,
        "freq": f_peak,
        "amp": float(amp_no_dc[peak_idx]),  # 返回原始幅值，非截断后
        "phase": float(phase_no_dc[peak_idx]),
        "top_lambdas_m": top_lams,
        "top_freqs": top_freqs,
        "top_amps": amp_no_dc[top_idx_sorted],
    }


def phase_difference_at_frequency(
    x: np.ndarray, y: np.ndarray, step_m: float, freq: float,
) -> float:
    """计算两信号在指定频率处的相位差 (度)。

    返回 phi_y - phi_x，范围 [-180, 180]。
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
    """计算两信号的互相关峰值滞后。

    参数
    ------
    x, y : 等长 1-D 信号
    step_m : 采样间距 (m)
    max_lag_m : 最大搜索滞后距离

    返回
    ------
    {"lag_m": 峰值滞后 (m), "corr": 峰值归一化互相关系数}
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
    """计算自相关长度尺度。

    返回
    ------
    {"e_folding_m": e-折叠长度, "integral_scale_m": 积分尺度, "n": 有效点数}
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
    # e-folding 长度
    thr = float(np.exp(-1.0))
    below = np.where(acf <= thr)[0]
    e_fold = float(below[0]) * float(step_m) if below.size > 0 else float("nan")
    # 积分尺度（到第一个零交叉）
    neg = np.where(acf < 0)[0]
    kmax = int(neg[0]) if neg.size > 0 else int(acf.size)
    integral = float(np.sum(acf[1:kmax])) * float(step_m) if kmax > 1 else float("nan")
    return {"e_folding_m": e_fold, "integral_scale_m": integral, "n": int(x.size)}


# ---------------------------------------------------------------------------
# 高级分析：单年 trunk
# ---------------------------------------------------------------------------

@dataclass
class SpectralResult:
    """单年 trunk 谱分析结果。"""
    year: int
    trunk_length_km: float
    n_points: int
    step_m: float
    # B(s) 谱
    B_lambda_m: float       # B 主波长
    B_freq: float           # B 主频率
    B_amp: float            # B 主幅度
    B_efold_m: float        # B e-folding 自相关长度
    B_integral_m: float     # B 积分自相关尺度
    B_mean: float           # B 均值
    B_std: float            # B 标准差
    # C(s) 谱
    C_lambda_m: float       # C 主波长
    C_freq: float           # C 主频率
    C_amp: float            # C 主幅度
    C_efold_m: float        # C e-folding 自相关长度
    C_integral_m: float     # C 积分自相关尺度
    C_mean: float           # C 均值 (应接近 0)
    C_std: float            # C 标准差
    # B-C 关系
    BC_phase_deg: float     # C 主频率处 B-C 相位差 (度)
    BC_lag_m: float         # B-C 互相关峰值滞后 (m)
    BC_corr: float          # B-C 互相关峰值系数


def analyze_single_trunk(csv_path: Path | str,
                         step_m: float | None = None,
                         max_lag_km: float = 30.0,
                         ) -> SpectralResult:
    """对单个 trunk CSV 执行完整谱分析。

    参数
    ------
    csv_path : trunk CSV 路径 (columns: s_m, lon, lat, B_m, C_1m)
    step_m : 均匀化间距 (m)，None 则自动取中位间距
    max_lag_km : 互相关最大搜索滞后 (km)

    返回
    ------
    SpectralResult dataclass
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    # 提取年份
    stem = csv_path.stem  # e.g. Gaocun-Sunkou_2016_trunk_0
    year = int(stem.split("_")[1])

    s_raw = np.asarray(df["s_m"].values, dtype=float)
    B_raw = np.asarray(df["B_m"].values, dtype=float)
    C_raw = np.asarray(df["C_1m"].values, dtype=float)

    # 均匀化插值
    s_B, B_uni, step = _interpolate_uniform(s_raw, B_raw, step_m)
    _, C_uni, _ = _interpolate_uniform(s_raw, C_raw, step)

    # 确保等长（取较短者）
    n = min(B_uni.size, C_uni.size)
    B_uni, C_uni = B_uni[:n], C_uni[:n]
    s_uni = s_B[:n]

    trunk_km = float(s_uni[-1] - s_uni[0]) / 1000.0 if n > 1 else 0.0

    # B 谱分析
    domB = dominant_wavelength(B_uni, step_m=step)
    scB = autocorr_length_scales(B_uni, step_m=step)

    # C 谱分析
    domC = dominant_wavelength(C_uni, step_m=step)
    scC = autocorr_length_scales(C_uni, step_m=step)

    # B-C 互谱
    # 在 C 的主频率处计算 B-C 相位差
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
    """保存单个信号的 FFT 频谱到 CSV。"""
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
# 批量分析
# ---------------------------------------------------------------------------

def analyze_all_trunks(
    trunk_dir: Path | None = None,
    out_dir: Path | None = None,
    years: List[int] | None = None,
    step_m: float | None = None,
    save_spectra: bool = True,
) -> pd.DataFrame:
    """批量分析所有年份的 trunk 数据。

    参数
    ------
    trunk_dir : trunk CSV 目录
    out_dir : 输出目录
    years : 要分析的年份列表，None=全部
    step_m : 均匀化间距
    save_spectra : 是否保存每年的功率谱 CSV

    返回
    ------
    DataFrame with one row per year, columns from SpectralResult fields
    """
    trunk_dir = trunk_dir or TRUNK_DIR
    out_dir = out_dir or SPECTRAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 发现 trunk 文件
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

            # 保存单年频谱
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
            # 填充 NaN 行
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
    """命令行入口。"""
    # Windows UTF-8 安全输出
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

    # 打印摘要统计
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
