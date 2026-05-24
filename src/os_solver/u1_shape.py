"""曲率形状函数 U1(y) 工厂模块.

基于 Gemini 推导 3-4，实现两种 U1(y) 形状函数：
1. self_similar：U1 ∝ U0（曲率修正保持与基流相同的垂向形状）
2. polynomial_zs：满足 Ψ(0)=0, Ψ(1)=1, Ψ'(1)=0 的立方多项式

参考文献：
- Zolezzi & Seminara (2001)
- Ikeda, Parker & Sawai (1981)
- Johannesson & Parker (1989)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict


def compute_u1_amplitude(
    beta: float,
    Fr: float,
    Cf: float | None = None,
    A_scour: float = 4.0,
) -> float:
    """计算 U1 幅值标度因子 S_amp（O(1) 量）.

    基于 IPS 平衡态标度：
        S_amp = (beta / 2) * (A_scour + Fr^2)

    Parameters
    ----------
    beta : float
        宽深比 B/H
    Fr : float
        Froude 数
    Cf : float, optional
        摩擦系数，当前未使用（预留给更精细的标度）
    A_scour : float
        冲刷因子，典型值 4.0（IPS 模型）

    Returns
    -------
    float
        幅值标度因子 S_amp
    """
    return (beta / 2.0) * (A_scour + Fr ** 2)


def make_u1_self_similar(
    y: np.ndarray,
    U0: np.ndarray,
    U0_y: np.ndarray,
    U0_yy: np.ndarray,
    beta: float,
    Fr: float,
    Cf: float | None = None,
    A_scour: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Self-Similar 形状函数：U1 ∝ U0.

    物理假设：曲率引起的速度增量在各深度等比例缩放，
    即高速核向外岸偏移但保持相同的垂向分布。

    U1(y) = S_amp * U0(y) / mean(U0)

    Parameters
    ----------
    y : np.ndarray
        Chebyshev 网格 y ∈ [-1, 1]
    U0, U0_y, U0_yy : np.ndarray
        基流及其一阶、二阶导数
    beta : float
        宽深比 B/H
    Fr : float
        Froude 数
    Cf : float, optional
        摩擦系数
    A_scour : float
        冲刷因子

    Returns
    -------
    U1, U1_y, U1_yy : np.ndarray
        曲率形状函数及其导数
    """
    S_amp = compute_u1_amplitude(beta, Fr, Cf, A_scour)

    # 深度平均（使用简单均值近似 Clenshaw-Curtis）
    U0_mean = np.mean(U0)
    if np.abs(U0_mean) < 1e-12:
        U0_mean = 1.0  # 避免除零

    scale = S_amp / U0_mean
    U1 = scale * U0
    U1_y = scale * U0_y
    U1_yy = scale * U0_yy

    return U1, U1_y, U1_yy


def make_u1_polynomial_zs(
    y: np.ndarray,
    beta: float,
    Fr: float,
    Cf: float | None = None,
    z0: float | None = None,
    A_scour: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """修正立方多项式形状函数（Gemini 推导4）.

    满足边界条件：
    - Ψ(0) = 0（河床无滑移）
    - Ψ(1) = 1（归一化）
    - Ψ'(1) = 0（表面零剪切扰动）

    形状函数：
        Ψ(ζ) = (3/2)ζ - (1/2)ζ³
        其中 ζ = (y + 1) / 2 ∈ [0, 1]

    物理解释：该剖面在床面附近线性增长，在水面附近趋于平坦，
    模拟二次流将动量向表面/外岸输运的效应。

    Parameters
    ----------
    y : np.ndarray
        Chebyshev 网格 y ∈ [-1, 1]
    beta : float
        宽深比 B/H
    Fr : float
        Froude 数
    Cf : float, optional
        摩擦系数
    z0 : float, optional
        粗糙度高度（ZS 模型），当前实现中简化为映射到 ζ ∈ [0, 1]
    A_scour : float
        冲刷因子

    Returns
    -------
    U1, U1_y, U1_yy : np.ndarray
        曲率形状函数及其导数
    """
    S_amp = compute_u1_amplitude(beta, Fr, Cf, A_scour)

    # 归一化坐标：y ∈ [-1, 1] → ζ ∈ [0, 1]
    # 简化处理：忽略 z0，直接线性映射
    # ζ = (y + 1) / 2, dζ/dy = 0.5
    zeta = 0.5 * (y + 1.0)
    zeta = np.clip(zeta, 0.0, 1.0)

    # 形状函数 Ψ(ζ) = 1.5ζ - 0.5ζ³
    Psi = 1.5 * zeta - 0.5 * zeta ** 3
    dPsi_dzeta = 1.5 - 1.5 * zeta ** 2
    d2Psi_dzeta2 = -3.0 * zeta

    # 链式法则：d/dy = (dζ/dy) * d/dζ = 0.5 * d/dζ
    # d²/dy² = 0.5² * d²/dζ²
    dzeta_dy = 0.5

    U1 = S_amp * Psi
    U1_y = S_amp * dPsi_dzeta * dzeta_dy
    U1_yy = S_amp * d2Psi_dzeta2 * dzeta_dy ** 2

    return U1, U1_y, U1_yy


def make_u1_shape_function(
    y: np.ndarray,
    D: np.ndarray,
    D2: np.ndarray,
    params: Dict,
    U0: np.ndarray | None = None,
    U0_y: np.ndarray | None = None,
    U0_yy: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """U1 形状函数工厂（统一入口）.

    Parameters
    ----------
    y : np.ndarray
        Chebyshev 网格
    D, D2 : np.ndarray
        一阶、二阶微分矩阵
    params : Dict
        参数字典，需包含：
        - u1_shape_mode: "self_similar" | "polynomial_zs" | "analytic"
        - beta: 宽深比
        - Fr: Froude 数
        - Cf: 摩擦系数（可选）
        - A_scour: 冲刷因子（可选，默认 4.0）
    U0, U0_y, U0_yy : np.ndarray, optional
        基流及其导数（self_similar 模式需要）

    Returns
    -------
    U1, U1_y, U1_yy : np.ndarray
        曲率形状函数及其导数
    """
    mode = str(params.get("u1_shape_mode", "analytic"))
    beta = float(params.get("beta", 10.0))
    Fr = float(params.get("Fr", 0.7))
    Cf = params.get("Cf", None)
    if Cf is not None:
        Cf = float(Cf)
    A_scour = float(params.get("A_scour", 4.0))

    if mode == "self_similar":
        if U0 is None or U0_y is None or U0_yy is None:
            raise ValueError(
                "u1_shape_mode='self_similar' 需要提供 U0, U0_y, U0_yy"
            )
        return make_u1_self_similar(y, U0, U0_y, U0_yy, beta, Fr, Cf, A_scour)

    if mode == "polynomial_zs":
        z0 = params.get("z0", None)
        return make_u1_polynomial_zs(y, beta, Fr, Cf, z0, A_scour)

    # 默认：简单解析形式 U1(y) = y * (1 - y²)
    # 这是原有的 analytic 模式，幅值为 O(1)
    U1 = y * (1.0 - y ** 2)
    U1_y = D @ U1
    U1_yy = D2 @ U1
    return U1, U1_y, U1_yy
