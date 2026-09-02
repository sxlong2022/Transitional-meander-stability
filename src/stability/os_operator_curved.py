from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

import numpy as np

from .solve_os import (
    solve_os,
    _estimate_re_from_cf,
    _estimate_re_from_cf_open,
    make_openchannel_profile,
)
from .u1_shape import make_u1_shape_function


def _apply_curvature_only_correction(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    s: np.ndarray,
    c: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    Cf: np.ndarray,
    params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """geometry_mode="curvature_only" 。

    （ 3.1.3 ）：

    -  L^(0) ， ν（
      params["nu_curvature"]） L_c，

          L ≈ L^(0) + ν L_c,

       c ≈ c^(0) + ν c_c^(1)。
    -  L_c  s  MCMM c(s), B(s)
      ，“”：
       ΔA_c  I ，
       Im  O(ν) ，。

    ：
    -  c_eff = mean(|c(s)|) ；
    -  params  nu_curvature，
      

          λ_curved = λ_parallel + i * nu_curvature * c_eff,

      。
    -  nu_curvature=0 ，，。

    ：
    -  A,B  ΔA_c ，
      。
    """

    nu = float(params.get("nu_curvature", 0.0))
    if nu == 0.0:
        return eigvals, eigvecs

    if c.size == 0:
        return eigvals, eigvecs

    c_abs = np.abs(c.astype(float))
    c_eff = float(np.nanmean(c_abs))

    if not np.isfinite(c_eff) or c_eff == 0.0:
        return eigvals, eigvecs

    delta = 1j * nu * c_eff
    eigvals_shifted = eigvals + delta
    return eigvals_shifted, eigvecs


def assemble_delta_Ac_open_free_surface(
    aux: Dict,
    params: Dict | None = None,
) -> np.ndarray:
    L = aux["L"]
    n_v = L.shape[0]
    n_tot = n_v + 1

    delta_A = np.zeros((n_tot, n_tot), dtype=complex)

    if params is None:
        return delta_A

# ν_c、 ν_fs  ν_B
    nu = float(params.get("nu_curvature", 0.0))
    nu_width = float(params.get("nu_width", 0.0))
    nu_fs = float(params.get("nu_curvature_fs", 0.0))
# 0 ；
# (nu_fs) ， ΔA。
    if nu == 0.0 and nu_width == 0.0 and nu_fs == 0.0:
        return delta_A

# ： curvature_local(s)
    curvature_local = params.get("curvature_local", None)
    factor = 0.0
    if nu != 0.0:
        factor = nu
        if curvature_local is not None:
            try:
                curv_val = float(curvature_local)
            except (TypeError, ValueError):
                curv_val = np.nan
            if np.isfinite(curv_val) and curv_val != 0.0:
                factor = nu * curv_val

    y = aux.get("y")
    D2 = aux.get("D2")
    if y is None or D2 is None:
        return delta_A

    y_arr = np.asarray(y, dtype=float)
    if y_arr.shape[0] != n_v or D2.shape != (n_v, n_v):
        return delta_A

    alpha = float(aux.get("alpha", params.get("alpha", 1.0)))
    I_v = np.eye(n_v)

    D_mat = aux.get("D")
    D3 = aux.get("D3")
    U0 = aux.get("U")
    U0_y = aux.get("U_y")
    Re_val = float(aux.get("Re", params.get("Re", params.get("Re", 10000.0))))

# U1(y)：，
    U1_mode = str(params.get("U1_mode", "analytic"))

    if U1_mode in ("self_similar", "polynomial_zs"):
# Gemini  3-4
# params  beta, Fr, Cf
        u1_params = {
            "u1_shape_mode": U1_mode,
            "beta": float(params.get("beta", 10.0)),
            "Fr": float(aux.get("Fr", params.get("Fr", 0.7))),
            "Cf": params.get("Cf", None),
            "A_scour": float(params.get("A_scour", 4.0)),
        }
        D_mat = aux.get("D")
        U0 = aux.get("U")
        U0_y = aux.get("U_y")
        U0_yy = aux.get("U_yy")
        try:
            U1, _U1_y, U1_yy = make_u1_shape_function(
                y_arr, D_mat, D2, u1_params, U0, U0_y, U0_yy
            )
        except Exception:
# 
            U1 = y_arr * (1.0 - y_arr ** 2)
            U1_yy = D2 @ U1

    elif U1_mode == "external":
# params["U1_array"]  U1(y)， MCMM/ZS/IPS
# 。，
# ，。
        U1_external = params.get("U1_array", None)
        try:
            U1_candidate = np.asarray(U1_external, dtype=float)
        except Exception:
            U1_candidate = None

        if U1_candidate is not None and U1_candidate.shape == (n_v,):
            U1 = U1_candidate
        else:
# 
            U1 = y_arr * (1.0 - y_arr ** 2)

    elif U1_mode == "laminar_curved_diff":
# make_openchannel_profile  (y, D, D2, Re, Fr)
# laminar_ref  laminar_curved(curvature=1.0) ，
# U1(y) 。 laminar_curved  MCMM
# ， U1 。
        U_ref = aux.get("U")
        D_mat = aux.get("D")
        Re_val = float(aux.get("Re", params.get("Re", 10000.0)))
        Fr_val = float(aux.get("Fr", params.get("Fr", 1.0)))

        if (
            U_ref is not None
            and D_mat is not None
            and np.asarray(U_ref).shape[0] == n_v
            and D_mat.shape == (n_v, n_v)
        ):
            try:
                U_curv, _U_y_curv, _U_yy_curv = make_openchannel_profile(
                    y_arr,
                    D_mat,
                    D2,
                    beta_scalar=None,
                    Cf_scalar=None,
                    Fr_scalar=Fr_val,
                    Re_scalar=Re_val,
                    curvature=1.0,
                    width=None,
                    mode="laminar_curved",
                )
                U1 = np.asarray(U_curv, dtype=float) - np.asarray(U_ref, dtype=float)
            except Exception:
# ，
                U1 = y_arr * (1.0 - y_arr ** 2)
        else:
            U1 = y_arr * (1.0 - y_arr ** 2)
    else:
# U1(y) = y * (1 - y^2)
        U1 = y_arr * (1.0 - y_arr ** 2)
    U1_yy = D2 @ U1

    M_U1 = np.diag(U1)
    M_U1_yy = np.diag(U1_yy)

    bulk_op_c = M_U1_yy - M_U1 @ (D2 - (alpha ** 2) * I_v)

# L_B： W1(y)  U0, U0'
    width_local = params.get("width_local", None)
    dBds_local = params.get("dBds_local", None)
    eps_B = 0.0
    if (
        nu_width != 0.0
        and width_local is not None
        and dBds_local is not None
        and D_mat is not None
        and D3 is not None
        and U0 is not None
        and U0_y is not None
    ):
        try:
            B_val = float(width_local)
            dB_val = float(dBds_local)
        except (TypeError, ValueError):
            B_val = np.nan
            dB_val = np.nan
        if np.isfinite(B_val) and B_val != 0.0 and np.isfinite(dB_val):
            sigma_local = dB_val / B_val
            eps_B = nu_width * sigma_local

    A_B_bulk = None
    if eps_B != 0.0 and D_mat is not None and D3 is not None and U0 is not None and U0_y is not None:
        try:
            U0_arr = np.asarray(U0, dtype=float)
            U0_y_arr = np.asarray(U0_y, dtype=float)
            n_pts = U0_arr.shape[0]
            if n_pts == n_v:
                W1 = np.zeros_like(U0_arr, dtype=float)
                idx_sorted = np.argsort(y_arr)
                y_sorted = y_arr[idx_sorted]
                U_sorted = U0_arr[idx_sorted]
                acc = 0.0
                W1_sorted = np.zeros_like(U_sorted, dtype=float)
                for k in range(1, n_pts):
                    dy = float(y_sorted[k] - y_sorted[k - 1])
                    acc += 0.5 * dy * float(U_sorted[k] + U_sorted[k - 1])
                    W1_sorted[k] = -acc
                W1[idx_sorted] = W1_sorted

                M_W1 = np.diag(W1)
                M_U0 = np.diag(U0_arr)
                M_U0_y = np.diag(U0_y_arr)

# (2025-12-05, Rev.4):  (Final Rigorous Derivation)
                # 
# -:
                # (1) u' * grad(Omega): u' * (sigma*U') -> 1 * U'*D
# (2) Omega * div(u'): (-U') * (-sigma*u') -> 1 * U'*D  <-- !
# U'*D  2. ()
                #
# -sigma*U*w'  1 * U*D^2. ( 3,  1)
                #
# :
                # Re*eps * [ -W1*D3 + U0*D2 + (2*U0' + alpha^2*W1)*D - alpha^2*U0 ]
                
                coeff = Re_val * eps_B
                A_B_bulk = (
                    coeff
                    * (
                        -M_W1 @ D3
                        + M_U0 @ D2
                        + (2.0 * M_U0_y + (alpha ** 2) * M_W1) @ D_mat
                        - (alpha ** 2) * M_U0
                    )
                )
        except Exception:
            A_B_bulk = None

# bulk ：
    bulk_start = 2
    bulk_end = n_v - 2  # bottom2, bottom
    if bulk_end > bulk_start:
        rows = slice(bulk_start, bulk_end)
        v_slice = slice(0, n_v)
        if factor != 0.0:
# (2025-12-04):  Bulk term
            # Delta A = -i * alpha * Re * eps * [ U1 * (D2-k2) - U1'' ]
            #           = +i * alpha * Re * eps * [ U1'' - U1 * (D2-k2) ]
# bulk_op_c  [ U1'' - U1 * (D2-k2) ]
# coeff = 1j * alpha * Re
            coeff_bulk = 1j * alpha * Re_val
            delta_A[rows, v_slice] += coeff_bulk * factor * bulk_op_c[rows, :]
        if A_B_bulk is not None:
            delta_A[rows, v_slice] += A_B_bulk[rows, :]

# 
    if eps_B != 0.0 and U0 is not None:
        try:
            U0_surf = float(np.asarray(U0, dtype=float)[0])
        except Exception:
            U0_surf = np.nan
        if np.isfinite(U0_surf):
            row_kin = n_tot - 1
            eta_col = n_v
            delta_A[row_kin, eta_col] += 0.0

    nu_fs = float(params.get("nu_curvature_fs", 0.0))
    if nu_fs != 0.0:
        fs_factor = nu_fs
        if curvature_local is not None:
            try:
                curv_val_fs = float(curvature_local)
            except (TypeError, ValueError):
                curv_val_fs = np.nan
            if np.isfinite(curv_val_fs) and curv_val_fs != 0.0:
                fs_factor = nu_fs * curv_val_fs
        D_mat = aux.get("D")
        if (
            fs_factor != 0.0
            and D_mat is not None
            and np.asarray(D_mat).shape == (n_v, n_v)
        ):
            U1_y = D_mat @ U1
            U1_surf = float(U1[0])
            U1_y_surf = float(U1_y[0])
            D_row0 = D_mat[0, :]
            I_row0 = I_v[0, :]

            row_dyn = 1
            eta_col = n_v
            row_kin = n_tot - 1
            v_slice = slice(0, n_v)

# (2025-12-05):  Dynamic BC (Row 1)  Scaling 。
# solve_os.py  Row 1  (-i*alpha*Re) ：
            # Eq: (1/iRe) * Viscous + (U-c)*D*v - U'*v ...
# ， U  U -> U + eps*U1 ：
            # Delta LHS = eps * ( U1 * D - U1' ) * v
# -1j*alpha*Re， Re 。
            
            delta_A[row_dyn, v_slice] += fs_factor * (
                U1_surf * D_row0 - U1_y_surf * I_row0
            )
            delta_A[row_kin, eta_col] += fs_factor * (-1j * alpha * U1_surf)

    return delta_A


def solve_os_curved(
    s: np.ndarray,
    c: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    Cf: np.ndarray,
    bc: Dict,
    params: Dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    params_local: Dict = dict(params or {})
    geometry_mode = str(params_local.get("geometry_mode", "local_parallel"))

    if geometry_mode == "local_parallel":
        eigvals, eigvecs = solve_os(s, c, B, beta, Cf, bc, params_local)
        return eigvals, eigvecs

    if geometry_mode == "curvature_only":
        params_mod: Dict = dict(params_local)

# profile_mode， laminar_ref；
# profile_mode（ "zs_turbulent"）。
        params_mod.setdefault("profile_mode", "laminar_ref")
        params_mod.pop("curvature", None)

# Gemini  U1
# : "polynomial_zs", "self_similar", "laminar_curved_diff", "analytic"
        params_mod.setdefault("U1_mode", "polynomial_zs")

# beta  Cf  U1
# s
        if "beta" not in params_mod:
            try:
                beta_arr = np.asarray(beta, dtype=float)
                params_mod["beta"] = float(np.nanmean(beta_arr))
            except Exception:
                params_mod["beta"] = 10.0
        if "Cf" not in params_mod:
            try:
                Cf_arr = np.asarray(Cf, dtype=float)
                params_mod["Cf"] = float(np.nanmean(Cf_arr))
            except Exception:
                params_mod["Cf"] = 0.01

        params_mod["_delta_Ac_assembler"] = assemble_delta_Ac_open_free_surface

        eigvals, eigvecs = solve_os(s, c, B, beta, Cf, bc, params_mod)
        return eigvals, eigvecs

    raise ValueError(
        f"solve_os_curved:  geometry_mode={geometry_mode!r}， 'local_parallel'  'curvature_only'。"
    )