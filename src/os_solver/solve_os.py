"""Orr-Sommerfeld 稳定性分析求解器（移植自 Subproject-1，适配 Subproject-3）.

本模块只定义接口与基本数据流，用于在给定 1D 基流
(s, c(s), B(s), beta(s), Cf(s)) 情况下，求解类似 O-S 的
本征值问题，输出增长率谱与模态结构。

注意：已移除 MCMM Fortran 接口依赖，仅保留纯 Python 求解器核心。
"""

from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

import numpy as np


def _estimate_re_from_cf(Cf: float) -> float:
    if Cf <= 0.0:
        raise ValueError(f"Cf must be positive to estimate Re, got {Cf}")
    return 12.0 / Cf


def _estimate_re_from_cf_open(Cf: float) -> float:
    if Cf <= 0.0:
        raise ValueError(f"Cf must be positive to estimate Re, got {Cf}")
    return 6.0 / Cf


def make_openchannel_profile(
    y: np.ndarray,
    D: np.ndarray,
    D2: np.ndarray,
    beta_scalar: float | None = None,
    Cf_scalar: float | None = None,
    Fr_scalar: float | None = None,
    Re_scalar: float | None = None,
    curvature: float | None = None,
    width: float | None = None,
    mode: str = "laminar_ref",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造开渠道基流 U(y) 及其一二阶导数（阶段 2：基流工厂原型）。

    当前实现：
    - mode="laminar_ref"：对应 solve_os 中原有的开渠道层流型剖面
      U(y) = -0.25*y**2 + 0.5*y + 0.75；
    - mode="empirical_power"：一个简单的经验型剖面族，利用 Re_scalar 控制
      剖面“顶平程度”（Re 越大，越接近顶平），满足 U(-1)=0, U'(1)=0, U(1)=1。
    - mode="zs_turbulent"：基于 Zolezzi & Seminara (2001) 湍流解析剖面
      U0(z; Cf)（通过 Cf 或 Re 估计），并在 y∈[-1,1] 上归一化使 U(-1)≈0, U(1)=1。
    - mode="laminar_curved"：对应 solve_os 中原有的开渠道层流型剖面
      U(y) = (1.0 + gamma) * (-0.25*y**2 + 0.5*y + 0.75)，其中 gamma 为曲率参数。

    其他物理参数和模式先作为占位，在后续工作中根据 3.1.3 大纲和文献推导
    再具体使用。
    """

    if mode == "laminar_ref":
        U = -0.25 * y ** 2 + 0.5 * y + 0.75
        U_y = D @ U
        U_yy = D2 @ U
        return U, U_y, U_yy

    if mode == "laminar_curved":
        U_base = -0.25 * y ** 2 + 0.5 * y + 0.75
        if curvature is None:
            gamma = 0.0
        else:
            gamma = float(curvature)
        f_c = y * (1.0 - y ** 2)
        U = U_base + gamma * f_c
        U_y = D @ U
        U_yy = D2 @ U
        return U, U_y, U_yy

    if mode == "empirical_power":
        # 使用一个简单的一参数族 U(z) = z^p * ((1+p) - p z), z \in [0,1]
        # 满足 U(0)=0, U(1)=1, U'(1)=0。参数 p 随 Re_scalar 单调变化：
        # Re 小时较“抛物线”，Re 大时更趋向顶平。
        if Re_scalar is None or Re_scalar <= 0.0:
            p = 2.0
        else:
            Re0 = 500.0
            Re1 = 5000.0
            logR = np.log10(Re_scalar)
            t = (logR - np.log10(Re0)) / (np.log10(Re1) - np.log10(Re0))
            t = float(max(0.0, min(1.0, t)))
            p = 2.0 + 3.0 * t

        z = 0.5 * (y + 1.0)
        z_clipped = np.clip(z, 0.0, 1.0)
        U = z_clipped ** p * ((1.0 + p) - p * z_clipped)
        U_y = D @ U
        U_yy = D2 @ U
        return U, U_y, U_yy

    if mode == "zs_turbulent":
        # Zolezzi & Seminara (2001) 湍流解析剖面 U0(z; Cf)
        # 需要 Cf_scalar 或 Re_scalar 用于构造 Cf。
        Cf_eff: float | None = None
        if Cf_scalar is not None and Cf_scalar > 0.0:
            Cf_eff = float(Cf_scalar)
        elif Re_scalar is not None and Re_scalar > 0.0:
            # 与 _estimate_re_from_cf_open 保持一致：Re ≈ 6 / Cf
            Cf_eff = 6.0 / float(Re_scalar)

        if Cf_eff is None or Cf_eff <= 0.0:
            raise ValueError(
                "make_openchannel_profile(mode='zs_turbulent') 需要正的 Cf_scalar 或 Re_scalar。"
            )

        kappa = 0.41
        A_zs = 1.84
        B_zs = -1.56
        z0 = np.exp(-kappa / np.sqrt(Cf_eff) - 0.777)

        # 将 y ∈ [-1,1] 线性映射到 z ∈ [z0, 1]
        z = 0.5 * (y + 1.0)
        z = z0 + (1.0 - z0) * np.clip(z, 0.0, 1.0)

        aux1 = np.log(z / z0)
        aux2 = A_zs * (z**2 - z0**2)
        aux3 = B_zs * (z**3 - z0**3)
        U0 = (np.sqrt(Cf_eff) / kappa) * (aux1 + aux2 + aux3)

        # 线性平移 + 归一化，使 U(-1)≈0, U(1)=1，保证与其它 profile_mode 可比
        U0_min = float(np.min(U0))
        U0_max = float(np.max(U0))
        if U0_max > U0_min:
            U = (U0 - U0_min) / (U0_max - U0_min)
        else:
            U = np.zeros_like(U0)

        U_y = D @ U
        U_yy = D2 @ U
        return U, U_y, U_yy

    if mode == "mcmm_profile":
        # 从 MCMM 垂向剖面文件读取 U(y)，并插值到当前 Chebyshev 网格
        # 需要在 params 中传入 mcmm_profile_path 或 mcmm_profile_data
        # 这里先实现一个占位：若无数据则回退到 laminar_ref
        #
        # 未来完整实现：
        # 1. 从 load_os_profile_from_mcmm 读取 (y_mcmm, U_mcmm)
        # 2. 用 scipy.interpolate 插值到 Chebyshev 网格 y
        # 3. 用 D, D2 计算 U_y, U_yy
        #
        # 当前占位：直接使用 laminar_ref
        U = -0.25 * y ** 2 + 0.5 * y + 0.75
        U_y = D @ U
        U_yy = D2 @ U
        return U, U_y, U_yy

    raise ValueError(f"make_openchannel_profile: 未实现的 mode={mode!r}")


def assemble_open_free_surface_matrices(
    D: np.ndarray,
    D2: np.ndarray,
    D3: np.ndarray,
    y: np.ndarray,
    alpha: float,
    Re: float,
    Fr: float,
    profile_mode: str,
    curvature_param: float | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    N = D.shape[0] - 1
    n_v = N + 1
    I_v = np.eye(n_v)
    k = alpha

    U, U_y, U_yy = make_openchannel_profile(
        y,
        D,
        D2,
        beta_scalar=None,
        Cf_scalar=None,
        Fr_scalar=float(Fr),
        Re_scalar=float(Re),
        curvature=curvature_param,
        width=None,
        mode=profile_mode,
    )

    U_surf = U[0]
    U_y_surf = U_y[0]
    U_yy_surf = U_yy[0]

    L = D2 - (k ** 2) * I_v
    A_full = L @ L - 1j * alpha * Re * (np.diag(U) @ L - np.diag(U_yy))
    B_full = -1j * alpha * Re * L

    n_tot = n_v + 1
    A_ext = np.zeros((n_tot, n_tot), dtype=complex)
    B_ext = np.zeros((n_tot, n_tot), dtype=complex)

    A_ext[:n_v, :n_v] = A_full
    B_ext[:n_v, :n_v] = B_full

    v_slice = slice(0, n_v)
    eta_col = n_v

    A_ext[0, :] = 0.0
    B_ext[0, :] = 0.0
    A_ext[0, v_slice] = D2[0, :] + (alpha ** 2) * I_v[0, :]
    A_ext[0, eta_col] = 0.0

    A_ext[1, :] = 0.0
    B_ext[1, :] = 0.0
    term_viscous = (1j / (alpha * Re)) * (D3[0, :] - 3.0 * (alpha ** 2) * D[0, :])
    term_conv = U_surf * D[0, :] - U_y_surf * I_v[0, :]
    A_ext[1, v_slice] = term_viscous + term_conv
    A_ext[1, eta_col] = -1j * alpha / (Fr ** 2)
    B_ext[1, v_slice] = D[0, :]

    bottom = n_v - 1
    bottom2 = n_v - 2

    A_ext[bottom2, :] = 0.0
    B_ext[bottom2, :] = 0.0
    A_ext[bottom2, v_slice] = D[bottom, :]

    A_ext[bottom, :] = 0.0
    B_ext[bottom, :] = 0.0
    A_ext[bottom, v_slice] = I_v[bottom, :]

    rowK = n_tot - 1
    A_ext[rowK, :] = 0.0
    B_ext[rowK, :] = 0.0
    A_ext[rowK, v_slice] = I_v[0, :]
    A_ext[rowK, eta_col] = -1j * alpha * U_surf
    B_ext[rowK, eta_col] = -1j * alpha

    aux = {
        "y": y,
        "D": D,
        "D2": D2,
        "D3": D3,
        "U": U,
        "U_y": U_y,
        "U_yy": U_yy,
        "L": L,
        "alpha": alpha,
        "Re": Re,
        "Fr": Fr,
        "B_ext": B_ext,
    }

    return A_ext, B_ext, aux


def solve_os(
    s: np.ndarray,
    c: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    Cf: np.ndarray,
    bc: Dict,
    params: Dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """求解给定基流下的 Orr–Sommerfeld 型本征值问题（接口草案）。

    Parameters
    ----------
    s : array_like
        弧长坐标 s 的一维数组。
    c : array_like
        曲率 c(s) 的一维数组。
    B : array_like
        河宽 B(s) 的一维数组。
    beta : array_like
        宽深比 β(s) 的一维数组。
    Cf : array_like
        阻力系数 Cf(s) 的一维数组。
    bc : dict
        边界条件设置，如 {"type": "periodic"} 等。
    params : dict, optional
        其他物理/数值参数（如 Re, Fr, 波数范围、离散点数等）。

    Returns
    -------
    eigvals : np.ndarray
        本征值数组（可对应增长率 σ 或复频率）。
    eigvecs : np.ndarray
        本征向量矩阵，每一列对应一个模态。

    Notes
    -----
    - 当前函数仅为“接口定义”，尚未实现具体离散与求解；
    - 实现时可在函数内部按需导入 scipy.sparse / scipy.sparse.linalg；
    - 需与 3.1.3 子课题一大纲中“4.3 O–S 特征值分析”部分保持一致。
    """
    if params is None:
        params = {}

    alpha = float(params.get("alpha", 1.0))
    Re = float(params.get("Re", 10000.0))
    N = int(params.get("N", 80))

    bc_type = str(params.get("bc_type", "rigid"))
    Fr = params.get("Fr", None)
    profile_mode = str(params.get("profile_mode", "laminar_ref"))
    curvature_param = params.get("curvature", None)
    delta_Ac_assembler = params.get("_delta_Ac_assembler", None)

    if bc_type not in {"rigid", "open_rigid_lid", "open_free_surface"}:
        raise ValueError(
            f"未知的 bc_type={bc_type!r}，必须为 'rigid'、'open_rigid_lid' 或 'open_free_surface'。"
        )

    def _cheb(N: int) -> tuple[np.ndarray, np.ndarray]:
        if N <= 0:
            return np.array([[0.0]]), np.array([1.0])
        k = np.arange(0, N + 1)
        x = np.cos(np.pi * k / N)
        c = np.ones(N + 1)
        c[0] = 2.0
        c[-1] = 2.0
        c = c * ((-1.0) ** k)
        X = np.tile(x, (N + 1, 1))
        dX = X - X.T
        D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
        D = D - np.diag(np.sum(D, axis=1))
        return D, x

    try:
        import scipy.linalg as la
    except ImportError as exc:
        raise ImportError("solve_os 需要 SciPy (scipy.linalg)，请先安装再使用。") from exc

    D, y = _cheb(N)
    D2 = D @ D
    D3 = D2 @ D

    if bc_type == "open_free_surface":
        if Fr is None:
            raise ValueError("bc_type='open_free_surface' 需要在 params 中提供 Fr。")
        A_ext, B_ext, aux = assemble_open_free_surface_matrices(
            D,
            D2,
            D3,
            y,
            alpha,
            Re,
            float(Fr),
            profile_mode,
            curvature_param,
        )

        if callable(delta_Ac_assembler):
            delta_A = delta_Ac_assembler(aux, params)
            if delta_A is not None:
                if delta_A.shape != A_ext.shape:
                    raise ValueError(
                        f"_delta_Ac_assembler 返回的矩阵形状 {delta_A.shape} 与 A_ext 形状 {A_ext.shape} 不一致。"
                    )
                A_ext = A_ext + delta_A

        eigvals, eigvecs = la.eig(A_ext, B_ext)
        return eigvals, eigvecs

    k = alpha
    D2i = D2[1:-1, 1:-1]
    U = 1.0 - y[1:-1] ** 2
    Up2 = -2.0 * np.ones_like(U)
    I = np.eye(N - 1)
    A0 = D2i - (k ** 2) * I
    A = A0 @ A0 - 1j * alpha * Re * (np.diag(U) @ A0 - np.diag(Up2))
    B = A0
    w, v = la.eig(A, B)
    return w, v
