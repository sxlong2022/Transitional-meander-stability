"""Orr-Sommerfeld stability analysis solver (ported from Subproject-1, adapted for Subproject-3).

This module only defines interfaces and basic data flows for use in a given 1D base flow
(s, c(s), B(s), beta(s), Cf(s)), solve similar O-S
Eigenvalue problem, output growth rate spectrum and modal structure.

Note: The MCMM Fortran interface dependency has been removed, leaving only the pure Python solver core.
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
    """Construct the channeled base flow U(y) and its first and second derivatives (Phase 2: Base flow factory prototype).

    Current implementation:
    - mode="laminar_ref": corresponds to the original open channel laminar flow profile in solve_os
      U(y) = -0.25*y**2 + 0.5*y + 0.75；
    - mode="empirical_power": a simple empirical profile family, controlled by Re_scalar
      The "top flatness" of the profile (the larger Re, the closer it is to the top flat) satisfies U(-1)=0, U'(1)=0, U(1)=1.
    - mode="zs_turbulent": based on Zolezzi & Seminara (2001) turbulence analysis profile
      U0(z; Cf) (estimated by Cf or Re), and normalized on y∈[-1,1] such that U(-1)≈0, U(1)=1.
    - mode="laminar_curved": corresponds to the original open channel laminar flow profile in solve_os
      U(y) = (1.0 + gamma) * (-0.25*y**2 + 0.5*y + 0.75), where gamma is the curvature parameter.

    Other physical parameters and models are first used as placeholders and will be derived in subsequent work based on the outline of 3.1.3 and the literature.
    Use it more specifically.
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
        # Use a simple one-parameter family U(z) = z^p * ((1+p) - p z), z \in [0,1]
        # Satisfy U(0)=0, U(1)=1, U'(1)=0. The parameter p varies monotonically with Re_scalar:
        # When Re is small, it is more "parabolic", and when Re is large, it tends to be flat.
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
        # Zolezzi & Seminara (2001) Turbulence analytical profile U0(z; Cf)
        # Requires Cf_scalar or Re_scalar for constructing Cf.
        Cf_eff: float | None = None
        if Cf_scalar is not None and Cf_scalar > 0.0:
            Cf_eff = float(Cf_scalar)
        elif Re_scalar is not None and Re_scalar > 0.0:
            # Consistent with _estimate_re_from_cf_open: Re ≈ 6 / Cf
            Cf_eff = 6.0 / float(Re_scalar)

        if Cf_eff is None or Cf_eff <= 0.0:
            raise ValueError(
                "make_openchannel_profile(mode='zs_turbulent')  Cf_scalar  Re_scalar。"
            )

        kappa = 0.41
        A_zs = 1.84
        B_zs = -1.56
        z0 = np.exp(-kappa / np.sqrt(Cf_eff) - 0.777)

        # linearly maps y ∈ [-1,1] to z ∈ [z0, 1]
        z = 0.5 * (y + 1.0)
        z = z0 + (1.0 - z0) * np.clip(z, 0.0, 1.0)

        aux1 = np.log(z / z0)
        aux2 = A_zs * (z**2 - z0**2)
        aux3 = B_zs * (z**3 - z0**3)
        U0 = (np.sqrt(Cf_eff) / kappa) * (aux1 + aux2 + aux3)

        # Linear translation + normalization, so that U(-1)≈0, U(1)=1, guaranteed to be comparable with other profile_mode
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
        # Read U(y) from MCMM vertical profile file and interpolate to current Chebyshev grid
        # You need to pass in mcmm_profile_path or mcmm_profile_data in params
        # Here first implement a placeholder: if there is no data, fall back to laminar_ref
        #
        # Complete implementation in the future:
        # 1. Read (y_mcmm, U_mcmm) from load_os_profile_from_mcmm
        # 2. Interpolate to Chebyshev grid y using scipy.interpolate
        # 3. Use D, D2 to calculate U_y, U_yy
        #
        # Current placeholder: use laminar_ref directly
        U = -0.25 * y ** 2 + 0.5 * y + 0.75
        U_y = D @ U
        U_yy = D2 @ U
        return U, U_y, U_yy

    raise ValueError(f"make_openchannel_profile:  mode={mode!r}")


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
    """Solve an Orr–Sommerfeld type eigenvalue problem for a given base flow (draft interface).

    Parameters
    ----------
    s : array_like
        One-dimensional array of arc length coordinates s.
    c : array_like
        One-dimensional array of curvatures c(s).
    B : array_like
        One-dimensional array of river widths B(s).
    beta : array_like
        One-dimensional array of width-to-depth ratio β(s).
    Cf : array_like
        One-dimensional array of drag coefficients Cf(s).
    bc : dict
        Boundary condition settings, such as {"type": "periodic"} etc.
    params : dict, optional
        Other physical/numerical parameters (such as Re, Fr, wave number range, number of discrete points, etc.).

    Returns
    -------
    eigvals : np.ndarray
        Array of eigenvalues ​​(can correspond to growth rate σ or complex frequency).
    eigvecs : np.ndarray
        Eigenvector matrix, each column corresponds to a mode.

    Notes
    -----
    - The current function is only an "interface definition" and has not yet implemented specific discretization and solution;
    - During implementation, scipy.sparse / scipy.sparse.linalg can be imported as needed inside the function;
    - It needs to be consistent with the “4.3 O–S Eigenvalue Analysis” section in the syllabus of 3.1.3 Subtopic 1.
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
            f" bc_type={bc_type!r}， 'rigid'、'open_rigid_lid'  'open_free_surface'。"
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
        raise ImportError("solve_os  SciPy (scipy.linalg)，。") from exc

    D, y = _cheb(N)
    D2 = D @ D
    D3 = D2 @ D

    if bc_type == "open_free_surface":
        if Fr is None:
            raise ValueError("bc_type='open_free_surface'  params  Fr。")
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
                        f"_delta_Ac_assembler  {delta_A.shape}  A_ext  {A_ext.shape} 。"
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
