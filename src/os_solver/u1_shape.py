""" U1(y) .

Based on Gemini derivation 3-4， U1(y) ：
1. self_similar：U1 ∝ U0（）
2. polynomial_zs： Ψ(0)=0, Ψ(1)=1, Ψ'(1)=0 

：
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
    """ U1  S_amp（O(1) ）.

     IPS ：
        S_amp = (beta / 2) * (A_scour + Fr^2)

    Parameters
    ----------
    beta : float
        Aspect ratio B/H
    Fr : float
        Froude number
    Cf : float, optional
        Friction coefficient，（）
    A_scour : float
        ， 4.0（IPS ）

    Returns
    -------
    float
         S_amp
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
    """Self-Similar ：U1 ∝ U0.

    ：，
    。

    U1(y) = S_amp * U0(y) / mean(U0)

    Parameters
    ----------
    y : np.ndarray
        Chebyshev  y ∈ [-1, 1]
    U0, U0_y, U0_yy : np.ndarray
        、
    beta : float
        Aspect ratio B/H
    Fr : float
        Froude number
    Cf : float, optional
        Friction coefficient
    A_scour : float
        

    Returns
    -------
    U1, U1_y, U1_yy : np.ndarray
        
    """
    S_amp = compute_u1_amplitude(beta, Fr, Cf, A_scour)

    # （ Clenshaw-Curtis）
    U0_mean = np.mean(U0)
    if np.abs(U0_mean) < 1e-12:
        U0_mean = 1.0  # 

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
    """（Gemini 4）.

    Boundary conditions：
    - Ψ(0) = 0（）
    - Ψ(1) = 1（）
    - Ψ'(1) = 0（）

    ：
        Ψ(ζ) = (3/2)ζ - (1/2)ζ³
         ζ = (y + 1) / 2 ∈ [0, 1]

    ：，，
    /。

    Parameters
    ----------
    y : np.ndarray
        Chebyshev  y ∈ [-1, 1]
    beta : float
        Aspect ratio B/H
    Fr : float
        Froude number
    Cf : float, optional
        Friction coefficient
    z0 : float, optional
        （ZS ）， ζ ∈ [0, 1]
    A_scour : float
        

    Returns
    -------
    U1, U1_y, U1_yy : np.ndarray
        
    """
    S_amp = compute_u1_amplitude(beta, Fr, Cf, A_scour)

    # ：y ∈ [-1, 1] → ζ ∈ [0, 1]
    # ： z0，
    # ζ = (y + 1) / 2, dζ/dy = 0.5
    zeta = 0.5 * (y + 1.0)
    zeta = np.clip(zeta, 0.0, 1.0)

    # Ψ(ζ) = 1.5ζ - 0.5ζ³
    Psi = 1.5 * zeta - 0.5 * zeta ** 3
    dPsi_dzeta = 1.5 - 1.5 * zeta ** 2
    d2Psi_dzeta2 = -3.0 * zeta

    # ：d/dy = (dζ/dy) * d/dζ = 0.5 * d/dζ
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
    """U1 （）.

    Parameters
    ----------
    y : np.ndarray
        Chebyshev 
    D, D2 : np.ndarray
        、
    params : Dict
        Parameters，：
        - u1_shape_mode: "self_similar" | "polynomial_zs" | "analytic"
        - beta: 
        - Fr: Froude number
        - Cf: Friction coefficient（）
        - A_scour: （， 4.0）
    U0, U0_y, U0_yy : np.ndarray, optional
        （self_similar ）

    Returns
    -------
    U1, U1_y, U1_yy : np.ndarray
        
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
            raise ValueError("u1_shape_mode='self_similar' requires U0, U0_y, and U0_yy")
        return make_u1_self_similar(y, U0, U0_y, U0_yy, beta, Fr, Cf, A_scour)

    if mode == "polynomial_zs":
        z0 = params.get("z0", None)
        return make_u1_polynomial_zs(y, beta, Fr, Cf, z0, A_scour)

    # ： U1(y) = y * (1 - y²)
    # analytic ， O(1)
    U1 = y * (1.0 - y ** 2)
    U1_y = D @ U1
    U1_yy = D2 @ U1
    return U1, U1_y, U1_yy
