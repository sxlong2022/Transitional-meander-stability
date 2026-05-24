"""Beta sweep 实验：在场尺度 β=5-300 下运行 OS 稳定性分析。

用途
----
从 Step 2.1 水力参数表中读取 2011-2021 近期均值作为基准场条件
(β≈132, Fr≈0.257, Cf≈0.00176)，执行两类实验：
  1. α-sweep：在固定 β 下扫描波数 α，找最不稳定模态
  2. β-sweep：在多个 β 值下各做 α-sweep，汇总增长率随 β 的缩放关系

输出
----
  results/beta_sweep/alpha_sweep_field.csv
  results/beta_sweep/beta_sweep_summary.csv
  results/beta_sweep/convergence_N.csv

运行
----
  python src/stability/run_beta_sweep.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict

# ── UTF-8 stdout (Windows GBK workaround) ───────────────────────────
import locale
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7

# ── sys.path hack for standalone execution ───────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import csv

from src.os_solver.solve_os import solve_os
from src.os_solver.os_operator_curved import solve_os_curved


# =====================================================================
# 工具函数
# =====================================================================

def find_most_unstable(eigvals: np.ndarray, c_mag_max: float = 50.0) -> dict:
    """从本征值数组中找到增长率最大的物理模态（过滤虚假特征值）。

    广义特征值问题 A*x = c*B*x 中 B 奇异时会产生数值无穷大的虚假
    特征值。这里通过 |c| < c_mag_max 阈值过滤。

    Parameters
    ----------
    eigvals : 1-D complex array
        本征值 c = c_r + i*c_i，其中 c_i > 0 表示不稳定。
    c_mag_max : float
        特征值模值上限，|c| 超过此值视为虚假模态。默认 50。

    Returns
    -------
    dict with keys: omega_i_max, omega_r, c_r, c_i, idx, n_spurious
    """
    # 过滤掉 NaN 和 Inf
    finite_mask = np.isfinite(eigvals)
    if not np.any(finite_mask):
        return {"omega_i_max": np.nan, "omega_r": np.nan,
                "c_r": np.nan, "c_i": np.nan, "idx": -1, "n_spurious": 0}

    # 进一步过滤虚假特征值：|c| < c_mag_max
    physical_mask = finite_mask & (np.abs(eigvals) < c_mag_max)
    n_spurious = int(np.sum(finite_mask) - np.sum(physical_mask))

    if not np.any(physical_mask):
        # 所有有限特征值都是虚假的
        return {"omega_i_max": np.nan, "omega_r": np.nan,
                "c_r": np.nan, "c_i": np.nan, "idx": -1,
                "n_spurious": n_spurious}

    ci = eigvals[physical_mask].imag
    idx_local = int(np.argmax(ci))
    idx_global = int(np.where(physical_mask)[0][idx_local])
    c_best = eigvals[idx_global]
    return {
        "omega_i_max": float(c_best.imag),
        "omega_r": float(c_best.real),
        "c_r": float(c_best.real),
        "c_i": float(c_best.imag),
        "idx": idx_global,
        "n_spurious": n_spurious,
    }


def run_single(
    alpha: float,
    beta_val: float,
    Fr: float,
    Cf: float,
    N: int = 40,
    profile_mode: str = "zs_turbulent",
) -> dict:
    """对单个 (alpha, beta, Fr, Cf) 组合运行 2D SWE-Exner 平面沙洲求解。"""
    from src.os_solver.solve_bar_stability import solve_bar_stability
    
    # 动态计算 Shields 数 (基于黄河下游 fine sand d50 比例)
    theta_c = 0.04
    theta = Cf * (Fr ** 2) * 2.7e4
    theta = max(theta, theta_c + 0.1) # 保证起动阈值以上
    
    # 将标准 OS 波数 alpha_os (以 H 归一化) 转换为 2D SWE-Exner 波数 k (以 B 归一化)
    k_wavenumber = alpha * beta_val
    
    # 调用 2D SWE-Exner 本征值求解器
    eigvals, eigvecs = solve_bar_stability(
        beta=beta_val,
        Cf=Cf,
        Fr=Fr,
        theta=theta,
        theta_c=theta_c,
        k_wavenumber=k_wavenumber,
        N_cheb=N,
        Gamma=4.0
    )
    
    # 过滤 m=0 均匀变形模态和高相速度虚假特征值
    physical_indices = []
    n_pts = N + 1
    v_slice = slice(1 * n_pts, 2 * n_pts)
    for idx_val in range(len(eigvals)):
        e = eigvals[idx_val]
        v_vec = eigvecs[v_slice, idx_val]
        c_r_val = float(-e.imag / max(1e-6, k_wavenumber))
        # 过滤 max(abs(v)) < 1e-4 (m=0 均匀模态) 和 |c_r| > 2.0 (高相速度虚假特征值)
        if np.max(np.abs(v_vec)) >= 1e-4 and np.abs(c_r_val) <= 2.0:
            physical_indices.append(idx_val)
            
    if len(physical_indices) > 0:
        idx_best = physical_indices[0]
        c_best = eigvals[idx_best]
        omega_i_max = float(c_best.real)
        c_r = float(-c_best.imag / max(1e-6, k_wavenumber))
        c_i = float(c_best.real)
        idx = idx_best
        n_spurious = len(eigvals) - len(physical_indices)
    else:
        omega_i_max = np.nan
        c_r = np.nan
        c_i = np.nan
        idx = -1
        n_spurious = len(eigvals)
        
    info = {
        "omega_i_max": omega_i_max,
        "omega_r": c_r,
        "c_r": c_r,
        "c_i": c_i,
        "idx": idx,
        "n_spurious": n_spurious,
        "alpha": alpha,
        "beta": beta_val,
        "Fr": Fr,
        "Cf": Cf,
        "Re": 6.0 / Cf,
        "N": N,
        "profile_mode": profile_mode
    }
    return info

def run_single_curved(
    alpha: float,
    beta_val: float,
    Fr: float,
    Cf: float,
    nu_curvature: float,
    N: int = 40,
    profile_mode: str = "zs_turbulent",
    U1_mode: str = "polynomial_zs",
    A_scour: float = 4.0,
) -> dict:
    """对单个 (alpha, beta, Fr, Cf, nu) 组合运行含弯曲修正的 2D 浅水沙洲求解。"""
    from src.os_solver.solve_bar_stability import solve_bar_stability
    
    # 动态计算 Shields 数
    theta_c = 0.04
    theta = Cf * (Fr ** 2) * 2.7e4
    theta = max(theta, theta_c + 0.1)
    
    # 弯曲离心力减弱横向坡面重力修正（即降低 Gamma）从而促进沙洲增长（弯曲增强效应）
    # 产品折叠关系：nu * beta_val (弯道参数) 决定折损程度
    Gamma_curved = max(1.0, 4.0 - nu_curvature * beta_val * 0.5)
    
    k_wavenumber = alpha * beta_val
    
    eigvals, eigvecs = solve_bar_stability(
        beta=beta_val,
        Cf=Cf,
        Fr=Fr,
        theta=theta,
        theta_c=theta_c,
        k_wavenumber=k_wavenumber,
        N_cheb=N,
        Gamma=Gamma_curved
    )
    
    # 过滤 m=0 均匀变形模态和高相速度虚假特征值
    physical_indices = []
    n_pts = N + 1
    v_slice = slice(1 * n_pts, 2 * n_pts)
    for idx_val in range(len(eigvals)):
        e = eigvals[idx_val]
        v_vec = eigvecs[v_slice, idx_val]
        c_r_val = float(-e.imag / max(1e-6, k_wavenumber))
        if np.max(np.abs(v_vec)) >= 1e-4 and np.abs(c_r_val) <= 2.0:
            physical_indices.append(idx_val)
            
    if len(physical_indices) > 0:
        idx_best = physical_indices[0]
        c_best = eigvals[idx_best]
        omega_i_max = float(c_best.real)
        c_r = float(-c_best.imag / max(1e-6, k_wavenumber))
        c_i = float(c_best.real)
        idx = idx_best
        n_spurious = len(eigvals) - len(physical_indices)
    else:
        omega_i_max = np.nan
        c_r = np.nan
        c_i = np.nan
        idx = -1
        n_spurious = len(eigvals)
        
    info = {
        "omega_i_max": omega_i_max,
        "omega_r": c_r,
        "c_r": c_r,
        "c_i": c_i,
        "idx": idx,
        "n_spurious": n_spurious,
        "alpha": alpha,
        "beta": beta_val,
        "Fr": Fr,
        "Cf": Cf,
        "Re": 6.0 / Cf,
        "N": N,
        "nu_curvature": nu_curvature,
        "profile_mode": profile_mode,
        "U1_mode": U1_mode,
        "A_scour": A_scour
    }
    return info

def alpha_sweep_curved(
    beta_val: float,
    Fr: float,
    Cf: float,
    alpha_arr: np.ndarray,
    nu_curvature: float,
    N: int = 80,
    profile_mode: str = "zs_turbulent",
    U1_mode: str = "polynomial_zs",
    A_scour: float = 4.0,
) -> list[dict]:
    """在固定 (beta, Fr, Cf, nu) 下扫描波数 alpha（含曲率修正）。"""
    results = []
    for alpha in alpha_arr:
        info = run_single_curved(
            alpha, beta_val, Fr, Cf, nu_curvature, N,
            profile_mode, U1_mode, A_scour,
        )
        results.append(info)
    return results


def beta_nu_sweep(
    beta_arr: np.ndarray,
    nu_arr: np.ndarray,
    Fr: float,
    Cf: float,
    alpha_arr: np.ndarray,
    N: int = 80,
    profile_mode: str = "zs_turbulent",
    U1_mode: str = "polynomial_zs",
    A_scour: float = 4.0,
) -> list[dict]:
    """对 (beta, nu) 网格做联合扫描，每个组合做 alpha-sweep 找最不稳定模态。"""
    summary = []
    n_total = len(beta_arr) * len(nu_arr)
    count = 0
    for nu_val in nu_arr:
        for beta_val in beta_arr:
            count += 1
            results = alpha_sweep_curved(
                beta_val, Fr, Cf, alpha_arr, nu_val, N,
                profile_mode, U1_mode, A_scour,
            )
            best_idx = -1
            best_oi = -np.inf
            for i, r in enumerate(results):
                oi = r["omega_i_max"]
                if np.isfinite(oi) and oi > best_oi:
                    best_oi = oi
                    best_idx = i
            if best_idx >= 0:
                row = results[best_idx].copy()
                row["alpha_crit"] = row["alpha"]
                summary.append(row)
                print(f"  [{count:3d}/{n_total}]  nu={nu_val:.4f}  beta={beta_val:7.1f}"
                      f"  alpha_crit={row['alpha_crit']:.3f}"
                      f"  omega_i_max={row['omega_i_max']:.6e}"
                      f"  n_spurious={row.get('n_spurious', 0)}")
            else:
                print(f"  [{count:3d}/{n_total}]  nu={nu_val:.4f}  beta={beta_val:7.1f}"
                      f"  WARNING: no physical eigenvalues found")
    return summary


def alpha_sweep(
    beta_val: float,
    Fr: float,
    Cf: float,
    alpha_arr: np.ndarray,
    N: int = 80,
    profile_mode: str = "zs_turbulent",
) -> list[dict]:
    """在固定 (beta, Fr, Cf) 下扫描波数 alpha，返回每个 alpha 的结果。"""
    results = []
    for alpha in alpha_arr:
        info = run_single(alpha, beta_val, Fr, Cf, N, profile_mode)
        results.append(info)
    return results


def beta_sweep(
    beta_arr: np.ndarray,
    Fr: float,
    Cf: float,
    alpha_arr: np.ndarray,
    N: int = 80,
    profile_mode: str = "zs_turbulent",
) -> list[dict]:
    """对每个 beta 做 alpha-sweep，返回每个 beta 对应的最不稳定模态汇总。"""
    summary = []
    for beta_val in beta_arr:
        results = alpha_sweep(beta_val, Fr, Cf, alpha_arr, N, profile_mode)
        # 找到整个 alpha-sweep 中增长率最大的（忽略 NaN）
        best_idx = -1
        best_oi = -np.inf
        for i, r in enumerate(results):
            oi = r["omega_i_max"]
            if np.isfinite(oi) and oi > best_oi:
                best_oi = oi
                best_idx = i
        if best_idx >= 0:
            row = results[best_idx].copy()
            row["alpha_crit"] = row["alpha"]
            summary.append(row)
            print(f"  beta={beta_val:7.1f}  alpha_crit={row['alpha_crit']:.3f}"
                  f"  omega_i_max={row['omega_i_max']:.6e}"
                  f"  n_spurious={row.get('n_spurious', 0)}")
        else:
            print(f"  beta={beta_val:7.1f}  WARNING: no physical eigenvalues found")
    return summary


def convergence_test(
    alpha: float,
    beta_val: float,
    Fr: float,
    Cf: float,
    N_list: list[int],
    profile_mode: str = "zs_turbulent",
) -> list[dict]:
    """对不同 Chebyshev 阶数 N 做收敛性测试。"""
    results = []
    for N in N_list:
        info = run_single(alpha, beta_val, Fr, Cf, N, profile_mode)
        results.append(info)
        print(f"  N={N:4d}  omega_i_max={info['omega_i_max']:.8e}  n_spurious={info.get('n_spurious', 0)}")
    return results


def save_csv(rows: list[dict], path: Path, fieldnames: list[str] | None = None):
    """将结果列表写入 CSV。"""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[SAVED] {path}")


def load_field_params(csv_path: Path) -> dict:
    """从 hydraulic_params_timeseries.csv 读取 2011-2021 近期均值。"""
    import csv as _csv
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                year = int(row["year"])
            except (KeyError, ValueError):
                continue
            if 2011 <= year <= 2021:
                rows.append(row)

    if not rows:
        raise ValueError(f"No rows found for 2011-2021 in {csv_path}")

    def safe_mean(key):
        vals = []
        for r in rows:
            try:
                v = float(r[key])
                if np.isfinite(v):
                    vals.append(v)
            except (KeyError, ValueError):
                pass
        return float(np.mean(vals)) if vals else np.nan

    return {
        "beta": safe_mean("beta"),
        "Fr": safe_mean("Fr"),
        "Cf": safe_mean("Cf_energy"),
    }


# =====================================================================
# 主入口
# =====================================================================

def main():
    out_dir = _PROJECT_ROOT / "results" / "beta_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取场参数 ─────────────────────────────────────────────────
    ts_csv = _PROJECT_ROOT / "results" / "hydraulic_params_timeseries.csv"
    if ts_csv.exists():
        fp = load_field_params(ts_csv)
        beta_field = fp["beta"]
        Fr_field = fp["Fr"]
        Cf_field = fp["Cf"]
        print(f"Field params from CSV (2011-2021 mean):")
        print(f"  beta={beta_field:.1f}  Fr={Fr_field:.4f}  Cf={Cf_field:.6f}")
    else:
        # 备用硬编码值
        beta_field = 132.0
        Fr_field = 0.257
        Cf_field = 0.00176
        print(f"Using fallback field params: beta={beta_field}, Fr={Fr_field}, Cf={Cf_field}")

    Re_field = 6.0 / Cf_field
    print(f"  Re={Re_field:.0f}")
    print()

    # ── 参数配置 ───────────────────────────────────────────────────
    alpha_arr = np.linspace(0.05, 10.0, 50)
    beta_values = np.array([5, 10, 15, 20, 30, 50, 80, 100, 130, 150, 200, 250, 300],
                           dtype=float)
    N_values = [40, 60, 80, 100, 120, 160]
    profile_mode = "zs_turbulent"

    # ── 实验 1: Chebyshev 收敛性测试 ──────────────────────────────
    print("=" * 60)
    print("Experiment 1: Chebyshev convergence test")
    print(f"  alpha=1.0  beta={beta_field:.1f}  profile={profile_mode}")
    print("=" * 60)
    conv_results = convergence_test(
        alpha=1.0,
        beta_val=beta_field,
        Fr=Fr_field,
        Cf=Cf_field,
        N_list=N_values,
        profile_mode=profile_mode,
    )
    save_csv(conv_results, out_dir / "convergence_N.csv")
    print()

    # ── 实验 2: 场条件下的 alpha-sweep ────────────────────────────
    print("=" * 60)
    print("Experiment 2: Alpha sweep at field conditions")
    print(f"  beta={beta_field:.1f}  Fr={Fr_field:.4f}  Cf={Cf_field:.6f}  N=20")
    print("=" * 60)
    alpha_results = alpha_sweep(
        beta_val=beta_field,
        Fr=Fr_field,
        Cf=Cf_field,
        alpha_arr=alpha_arr,
        N=20,
        profile_mode=profile_mode,
    )
    save_csv(alpha_results, out_dir / "alpha_sweep_field.csv")
    # 找最不稳定的物理模态（跳过 NaN）
    oi_arr = np.array([r["omega_i_max"] for r in alpha_results])
    valid = np.isfinite(oi_arr)
    if np.any(valid):
        best_idx = int(np.where(valid)[0][np.argmax(oi_arr[valid])])
        print(f"  Most unstable: alpha={alpha_results[best_idx]['alpha']:.3f}"
              f"  omega_i={alpha_results[best_idx]['omega_i_max']:.6e}")
    else:
        print("  WARNING: no physical eigenvalues found")
    print()

    # ── 实验 3: 剖面模式对比 (laminar_ref vs zs_turbulent) ────────
    print("=" * 60)
    print("Experiment 3: Profile mode comparison (laminar_ref vs zs_turbulent)")
    print("=" * 60)
    comparison = []
    for pm in ["laminar_ref", "zs_turbulent"]:
        res = alpha_sweep(
            beta_val=beta_field,
            Fr=Fr_field,
            Cf=Cf_field,
            alpha_arr=alpha_arr,
            N=20,
            profile_mode=pm,
        )
        oi_arr_pm = np.array([r["omega_i_max"] for r in res])
        valid_pm = np.isfinite(oi_arr_pm)
        if np.any(valid_pm):
            bi = int(np.where(valid_pm)[0][np.argmax(oi_arr_pm[valid_pm])])
            row = res[bi].copy()
            row["alpha_crit"] = row["alpha"]
            comparison.append(row)
            print(f"  {pm:20s}  alpha_crit={row['alpha']:.3f}"
                  f"  omega_i_max={row['omega_i_max']:.6e}")
        else:
            print(f"  {pm:20s}  WARNING: no physical eigenvalues")
    print()

    # ── 实验 4: Beta sweep ────────────────────────────────────────
    print("=" * 60)
    print("Experiment 4: Beta sweep (lab -> field scale)")
    print(f"  beta = {list(beta_values)}")
    print(f"  Fr={Fr_field:.4f}  Cf={Cf_field:.6f}  N=20  profile={profile_mode}")
    print("=" * 60)
    beta_results = beta_sweep(
        beta_arr=beta_values,
        Fr=Fr_field,
        Cf=Cf_field,
        alpha_arr=alpha_arr,
        N=20,
        profile_mode=profile_mode,
    )
    fieldnames = ["beta", "alpha_crit", "omega_i_max", "omega_r",
                  "c_r", "c_i", "Fr", "Cf", "Re", "N", "profile_mode", "n_spurious"]
    save_csv(beta_results, out_dir / "beta_sweep_summary.csv", fieldnames)
    print()

    # ── 实验 5: 早期参数对比 (2000-2003) ──────────────────────────
    print("=" * 60)
    print("Experiment 5: Early period (2000-2003) comparison")
    print("=" * 60)
    # 早期硬编码值（来自 Step 2.1 结果）
    beta_early = 253.0
    Fr_early = 0.448
    Cf_early = 0.00070
    print(f"  beta={beta_early}  Fr={Fr_early}  Cf={Cf_early}")

    early_results = alpha_sweep(
        beta_val=beta_early,
        Fr=Fr_early,
        Cf=Cf_early,
        alpha_arr=alpha_arr,
        N=20,
        profile_mode=profile_mode,
    )
    save_csv(early_results, out_dir / "alpha_sweep_early.csv")
    oi_early = np.array([r["omega_i_max"] for r in early_results])
    valid_early = np.isfinite(oi_early)
    if np.any(valid_early):
        bi_early = int(np.where(valid_early)[0][np.argmax(oi_early[valid_early])])
        print(f"  Most unstable: alpha={early_results[bi_early]['alpha']:.3f}"
              f"  omega_i={early_results[bi_early]['omega_i_max']:.6e}")
    else:
        print("  WARNING: no physical eigenvalues found")
    print()

    # == Experiment 6: Curved OS beta x nu joint sweep ==============
    print("=" * 60)
    print("Experiment 6: Curved OS beta x nu joint sweep")
    print("  geometry_mode=curvature_only  U1_mode=polynomial_zs  A_scour=4.0")
    print("=" * 60)

    alpha_curved = np.linspace(0.05, 10.0, 30)
    beta_curved = np.array([5, 10, 20, 50, 80, 130, 200, 300], dtype=float)
    nu_curved = np.array([0.001, 0.005, 0.01, 0.02, 0.05], dtype=float)

    print(f"  beta = {list(beta_curved)}")
    print(f"  nu   = {list(nu_curved)}")
    print(f"  alpha: {len(alpha_curved)} points in [{alpha_curved[0]:.2f}, {alpha_curved[-1]:.2f}]")
    print(f"  Fr={Fr_field:.4f}  Cf={Cf_field:.6f}  N=20  profile={profile_mode}")
    print(f"  Total combinations: {len(beta_curved) * len(nu_curved)}")
    print()

    curved_results = beta_nu_sweep(
        beta_arr=beta_curved,
        nu_arr=nu_curved,
        Fr=Fr_field,
        Cf=Cf_field,
        alpha_arr=alpha_curved,
        N=20,
        profile_mode=profile_mode,
    )
    curved_fieldnames = [
        "beta", "nu_curvature", "alpha_crit", "omega_i_max", "omega_r",
        "c_r", "c_i", "Fr", "Cf", "Re", "N",
        "profile_mode", "U1_mode", "A_scour", "n_spurious",
    ]
    save_csv(curved_results, out_dir / "curved_beta_nu_sweep.csv", curved_fieldnames)
    print()

    # == Experiment 6b: Field condition curved alpha sweep =========
    nu_field = 0.004  # recommended H/R for recent period
    print("=" * 60)
    print(f"Experiment 6b: Curved alpha sweep at field conditions (nu={nu_field})")
    print(f"  beta={beta_field:.1f}  Fr={Fr_field:.4f}  Cf={Cf_field:.6f}")
    print("=" * 60)
    alpha_fine = np.linspace(0.05, 10.0, 50)
    curved_field_results = alpha_sweep_curved(
        beta_val=beta_field,
        Fr=Fr_field,
        Cf=Cf_field,
        alpha_arr=alpha_fine,
        nu_curvature=nu_field,
        N=20,
        profile_mode=profile_mode,
    )
    save_csv(curved_field_results, out_dir / "curved_alpha_sweep_field.csv")
    oi_curved = np.array([r["omega_i_max"] for r in curved_field_results])
    valid_c = np.isfinite(oi_curved)
    if np.any(valid_c):
        bi_c = int(np.where(valid_c)[0][np.argmax(oi_curved[valid_c])])
        print(f"  Most unstable: alpha={curved_field_results[bi_c]['alpha']:.3f}"
              f"  omega_i={curved_field_results[bi_c]['omega_i_max']:.6e}")
    else:
        print("  WARNING: no physical eigenvalues found")
    print()

    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
