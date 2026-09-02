"""Planar 2D shallow-water Exner linear morphodynamic stability solver.

Solves the generalized eigenvalue problem governing free and forced bar instability in straight
and weakly curved alluvial river channels using Chebyshev spectral collocation.

Governing Equations (dimensionless, scaled by B, H, U):
    Continuity:        i*k*u + d v/dn + i*k*h = -sigma * h
    Streamwise mom:    (i*k + 2*beta_eff*Cf)*u + (i*k/Fr^2 - beta_eff*Cf)*h + (i*k/Fr^2)*zb = -sigma * u
    Transverse mom:    (i*k + beta_eff*Cf)*v + (1/Fr^2)*dh/dn + (1/Fr^2)*dzb/dn - nu*u = -sigma * v
    Exner equation:    i*k*gamma_u*u + d v/dn - f_sec*nu*du/dn - Gamma_beta*d^2 zb/dn^2 = -sigma * zb

where:
    beta = B / H                  (aspect ratio)
    Fr = U / sqrt(g*H)            (Froude number)
    Cf = friction coefficient
    nu = H / R                    (curvature ratio, R = centerline curvature radius)
    nu_beta = B / R = nu * beta   (bend parameter)
    f_sec = 5.0                   (secondary helical flow shear coefficient)
    gamma_u = 2*b*theta / (theta - theta_c)  (sediment velocity sensitivity, b = 1.5)
    Gamma_beta = (Gamma / beta_eff) * sqrt(theta_c / theta)  (transverse slope diffusion)
    beta_eff = beta / mode_m      (effective aspect ratio for transverse mode m)
"""
from __future__ import annotations

# stdlib
from typing import Tuple, List, Dict, Optional

# third-party
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar


def chebyshev_collocation(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate N-th order Chebyshev-Gauss-Lobatto collocation nodes and differentiation matrix D.

    Parameters
    ----------
    N : int
        Polynomial degree (number of nodes = N + 1).

    Returns
    -------
    D_y : np.ndarray
        (N+1, N+1) differentiation matrix scaled to physical transverse domain n in [-0.5, 0.5].
    y : np.ndarray
        (N+1,) collocation nodes in [-0.5, 0.5], ordered from right bank (+0.5) to left bank (-0.5).
    """
    if N <= 0:
        return np.array([[0.0]]), np.array([0.0])

    k = np.arange(N + 1)
    # Computational coordinate xi in [-1, 1]
    xi = np.cos(np.pi * k / N)
    
    # Coefficients c_k (2 for boundaries, 1 for interior) with alternating sign
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** k)

    # Coordinate differences dXi[i, j] = xi[i] - xi[j]
    dXi = xi[:, None] - xi[None, :]
    
    # Off-diagonal entries on computational domain [-1, 1]
    D_xi = (c[:, None] / c[None, :]) / (dXi + np.eye(N + 1))
    # Diagonal entries via negative sum trick
    D_xi = D_xi - np.diag(np.sum(D_xi, axis=1))

    # Physical domain mapping: y = 0.5 * xi, so y in [-0.5, 0.5]
    y = 0.5 * xi
    # Chain rule: d/dy = (d xi / dy) * (d/d_xi) = 2.0 * d/d_xi
    D_y = 2.0 * D_xi
    
    return D_y, y


def solve_bar_stability(
    beta: float,
    Cf: float,
    Fr: float,
    theta: float,
    theta_c: float = 0.047,
    k_wavenumber: float = 3.0,
    nu: float = 0.0,
    f_sec: float = 5.0,
    N_cheb: int = 36,
    Gamma: float = 4.0,
    sediment_exponent: float = 1.5,
    mode_m: int = 1,
    filter_physical: bool = True,
    min_zb_ratio: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the 2D planar SWE-Exner generalized eigenvalue problem for bar stability.

    Parameters
    ----------
    beta : float
        Channel aspect ratio B/H.
    Cf : float
        Dimensionless bed friction coefficient.
    Fr : float
        Froude number U / sqrt(g*H).
    theta : float
        Shields parameter.
    theta_c : float, optional
        Critical Shields parameter for sediment motion (default 0.047).
    k_wavenumber : float, optional
        Width-scaled dimensionless longitudinal wavenumber k = 2*pi*B / lambda (default 3.0).
    nu : float, optional
        Channel curvature ratio nu = H/R (default 0.0 for straight channel).
    f_sec : float, optional
        Secondary helical flow bed shear deflection coefficient (default 5.0).
    N_cheb : int, optional
        Chebyshev polynomial truncation degree (default 36).
    Gamma : float, optional
        Transverse bedload slope deflection coefficient (default 4.0).
    sediment_exponent : float, optional
        Bedload transport power-law exponent b in MPM (default 1.5).
    mode_m : int, optional
        Transverse bar mode index (m=1: alternate bars, m=2: central bars, m>=3: multi-row bars).
    filter_physical : bool, optional
        Whether to filter out non-physical spurious boundary modes (default True).
    min_zb_ratio : float, optional
        Minimum bed perturbation amplitude ratio ||zb|| / ||X|| to accept as physical mode (default 1e-4).

    Returns
    -------
    eigvals_sorted : np.ndarray
        Complex eigenvalues sigma = sigma_r + i*sigma_i sorted in descending order of growth rate Re(sigma).
    eigvecs_sorted : np.ndarray
        Corresponding discrete eigenvectors [u(n), v(n), h(n), zb(n)]^T.
    """
    if beta <= 0.0 or Cf <= 0.0 or Fr <= 0.0:
        raise ValueError(f"Parameters must be positive: beta={beta}, Cf={Cf}, Fr={Fr}")
    if mode_m < 1:
        raise ValueError(f"mode_m must be >= 1, got {mode_m}")
    if N_cheb < 12:
        raise ValueError(f"N_cheb must be at least 12 for spectral accuracy, got {N_cheb}")

    # Effective aspect ratio for higher-order transverse mode m
    beta_eff = beta / float(mode_m)

    # 1. Collocation grid and differentiation matrices
    D, y = chebyshev_collocation(N_cheb)
    D2 = D @ D
    I_val = np.eye(N_cheb + 1)
    n_pts = N_cheb + 1
    n_tot = 4 * n_pts

    # 2. Linearized sediment parameters
    if theta > theta_c:
        gamma_u = (2.0 * sediment_exponent * theta) / (theta - theta_c)
    else:
        gamma_u = 0.0
    Gamma_beta = (Gamma / beta_eff) * np.sqrt(theta_c / max(1e-6, theta))

    # Slice indices for state components
    u_sl = slice(0 * n_pts, 1 * n_pts)
    v_sl = slice(1 * n_pts, 2 * n_pts)
    h_sl = slice(2 * n_pts, 3 * n_pts)
    zb_sl = slice(3 * n_pts, 4 * n_pts)

    A_mat = np.zeros((n_tot, n_tot), dtype=complex)
    B_mat = np.zeros((n_tot, n_tot), dtype=complex)

    # 3. Assemble interior governing equations (j = 1, ..., N_cheb - 1)
    for j in range(1, N_cheb):
        # (Row 0) Streamwise momentum: (i*k + 2*beta_eff*Cf)*u + (i*k/Fr^2 - beta_eff*Cf)*h + (i*k/Fr^2)*zb = -sigma * u
        A_mat[0 * n_pts + j, u_sl] = (1j * k_wavenumber + 2.0 * beta_eff * Cf) * I_val[j, :]
        A_mat[0 * n_pts + j, h_sl] = (1j * k_wavenumber / (Fr**2) - beta_eff * Cf) * I_val[j, :]
        A_mat[0 * n_pts + j, zb_sl] = (1j * k_wavenumber / (Fr**2)) * I_val[j, :]
        B_mat[0 * n_pts + j, u_sl] = -I_val[j, :]

        # (Row 1) Transverse momentum (with centrifugal curvature forcing -nu*u):
        # (i*k + beta_eff*Cf)*v - nu*u + (1/Fr^2)*dh/dn + (1/Fr^2)*dzb/dn = -sigma * v
        A_mat[1 * n_pts + j, u_sl] = -nu * I_val[j, :]
        A_mat[1 * n_pts + j, v_sl] = (1j * k_wavenumber + beta_eff * Cf) * I_val[j, :]
        A_mat[1 * n_pts + j, h_sl] = (1.0 / (Fr**2)) * D[j, :]
        A_mat[1 * n_pts + j, zb_sl] = (1.0 / (Fr**2)) * D[j, :]
        B_mat[1 * n_pts + j, v_sl] = -I_val[j, :]

        # (Row 2) Water continuity equation: i*k*u + dv/dn + i*k*h = -sigma * h
        A_mat[2 * n_pts + j, u_sl] = 1j * k_wavenumber * I_val[j, :]
        A_mat[2 * n_pts + j, v_sl] = D[j, :]
        A_mat[2 * n_pts + j, h_sl] = 1j * k_wavenumber * I_val[j, :]
        B_mat[2 * n_pts + j, h_sl] = -I_val[j, :]

        # (Row 3) Exner bed continuity (with secondary helical transport -f_sec*nu*du/dn):
        # i*k*gamma_u*u + dv/dn - f_sec*nu*du/dn - Gamma_beta*d^2 zb/dn^2 = -sigma * zb
        A_mat[3 * n_pts + j, u_sl] = 1j * k_wavenumber * gamma_u * I_val[j, :] - f_sec * nu * D[j, :]
        A_mat[3 * n_pts + j, v_sl] = D[j, :]
        A_mat[3 * n_pts + j, zb_sl] = -Gamma_beta * D2[j, :]
        B_mat[3 * n_pts + j, zb_sl] = -I_val[j, :]

    # 4. Enforce physical lateral bank boundary conditions at j = 0 (right bank) and j = N_cheb (left bank)
    for j in [0, N_cheb]:
        # (Row 0) Streamwise velocity free-slip: du/dn = 0
        r_u = 0 * n_pts + j
        A_mat[r_u, :] = 0.0
        B_mat[r_u, :] = 0.0
        A_mat[r_u, u_sl] = D[j, :]

        # (Row 1) Impermeable bank: v = 0
        r_v = 1 * n_pts + j
        A_mat[r_v, :] = 0.0
        B_mat[r_v, :] = 0.0
        A_mat[r_v, 1 * n_pts + j] = 1.0

        # (Row 2) Continuity equation holds at lateral boundaries
        r_h = 2 * n_pts + j
        A_mat[r_h, :] = 0.0
        B_mat[r_h, :] = 0.0
        A_mat[r_h, 0 * n_pts + j] = 1j * k_wavenumber
        A_mat[r_h, v_sl] = D[j, :]
        A_mat[r_h, 2 * n_pts + j] = 1j * k_wavenumber
        B_mat[r_h, 2 * n_pts + j] = -1.0

        # (Row 3) Zero transverse sediment flux: dzb/dn = 0
        r_zb = 3 * n_pts + j
        A_mat[r_zb, :] = 0.0
        B_mat[r_zb, :] = 0.0
        A_mat[r_zb, zb_sl] = D[j, :]

    # 5. Solve the generalized eigenvalue problem A * X = sigma * B * X
    eigvals, eigvecs = la.eig(A_mat, B_mat)

    # 6. Filter infinite/algebraic boundary modes
    valid = np.isfinite(eigvals) & (np.abs(eigvals) < 1e4)
    eigvals_filt = eigvals[valid]
    eigvecs_filt = eigvecs[:, valid]

    if len(eigvals_filt) == 0:
        return np.array([], dtype=complex), np.zeros((n_tot, 0), dtype=complex)

    # 7. Physical mode filtering: bed elevation component and transverse modal parity
    if filter_physical:
        phys_indices = []
        for idx in range(len(eigvals_filt)):
            vec = eigvecs_filt[:, idx]
            zb_vec = vec[zb_sl]
            tot_norm = np.linalg.norm(vec)
            zb_norm = np.linalg.norm(zb_vec)
            
            # (a) Morphodynamic bed response threshold: a bar mode must have active bed topography
            if tot_norm > 0 and (zb_norm / tot_norm) >= min_zb_ratio:
                phys_indices.append(idx)
        
        if len(phys_indices) > 0:
            eigvals_filt = eigvals_filt[phys_indices]
            eigvecs_filt = eigvecs_filt[:, phys_indices]
        else:
            return np.array([], dtype=complex), np.zeros((n_tot, 0), dtype=complex)

    # Sort descending by real growth rate Re(sigma)
    sort_idx = np.argsort(np.real(eigvals_filt))[::-1]
    eigvals_sorted = eigvals_filt[sort_idx]
    eigvecs_sorted = eigvecs_filt[:, sort_idx]

    return eigvals_sorted, eigvecs_sorted


def find_most_amplified_mode(
    beta: float,
    Cf: float,
    Fr: float,
    theta: float,
    theta_c: float = 0.047,
    nu: float = 0.0,
    f_sec: float = 5.0,
    k_bounds: Tuple[float, float] = (0.10, 20.00),
    N_cheb: int = 36,
    Gamma: float = 4.0,
    sediment_exponent: float = 1.5,
    mode_m: int = 1,
) -> Dict[str, float]:
    """Find the most amplified longitudinal wavenumber and peak growth rate using bounded scalar optimization.

    Returns dict with keys: sigma_r_max, sigma_i, k_max, alpha_max, c_migr.
    """
    def _neg_growth_coarse(k_val: float) -> float:
        try:
            evs, _ = solve_bar_stability(
                beta=beta,
                Cf=Cf,
                Fr=Fr,
                theta=theta,
                theta_c=theta_c,
                k_wavenumber=k_val,
                nu=nu,
                f_sec=f_sec,
                N_cheb=max(12, min(18, N_cheb)),
                Gamma=Gamma,
                sediment_exponent=sediment_exponent,
                mode_m=mode_m,
            )
            if len(evs) == 0:
                return 1e6
            return -float(evs[0].real)
        except Exception:
            return 1e6

    def _neg_growth_fine(k_val: float) -> float:
        try:
            evs, _ = solve_bar_stability(
                beta=beta,
                Cf=Cf,
                Fr=Fr,
                theta=theta,
                theta_c=theta_c,
                k_wavenumber=k_val,
                nu=nu,
                f_sec=f_sec,
                N_cheb=N_cheb,
                Gamma=Gamma,
                sediment_exponent=sediment_exponent,
                mode_m=mode_m,
            )
            if len(evs) == 0:
                return 1e6
            return -float(evs[0].real)
        except Exception:
            return 1e6

    # 1. Fast coarse search (25 log-spaced points) at N_cheb=18
    k_coarse = np.logspace(np.log10(k_bounds[0]), np.log10(k_bounds[1]), 25)
    coarse_growth = [-_neg_growth_coarse(kv) for kv in k_coarse]
    if max(coarse_growth) <= -1e5:
        return {"sigma_r_max": np.nan, "sigma_i": np.nan, "k_max": np.nan, "alpha_max": np.nan, "c_migr": np.nan}
    best_coarse_idx = int(np.argmax(coarse_growth))
    
    # 2. Refined bounded optimization around the peak at full N_cheb
    lo = max(0, best_coarse_idx - 2)
    hi = min(len(k_coarse) - 1, best_coarse_idx + 2)
    if lo >= hi:
        bracket_min, bracket_max = k_bounds
    else:
        bracket_min = max(k_bounds[0], float(k_coarse[lo]))
        bracket_max = min(k_bounds[1], float(k_coarse[hi]))
        if bracket_max <= bracket_min:
            bracket_min, bracket_max = k_bounds

    opt = minimize_scalar(
        _neg_growth_fine,
        bounds=(bracket_min, bracket_max),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 25},
    )

    k_opt = float(opt.x) if (opt.success and np.isfinite(opt.fun) and opt.fun < 1e5) else float(k_coarse[best_coarse_idx])
    try:
        evs_opt, _ = solve_bar_stability(
            beta=beta,
            Cf=Cf,
            Fr=Fr,
            theta=theta,
            theta_c=theta_c,
            k_wavenumber=k_opt,
            nu=nu,
            f_sec=f_sec,
            N_cheb=N_cheb,
            Gamma=Gamma,
            sediment_exponent=sediment_exponent,
            mode_m=mode_m,
        )
        if len(evs_opt) == 0:
            sigma_r = np.nan
            sigma_i = np.nan
            c_migr = np.nan
        else:
            sigma_opt = evs_opt[0]
            sigma_r = float(sigma_opt.real)
            sigma_i = float(sigma_opt.imag)
            c_migr = float(-sigma_i / k_opt) if k_opt > 0 else 0.0
    except Exception:
        sigma_r = np.nan
        sigma_i = np.nan
        c_migr = np.nan

    return {
        "sigma_r_max": sigma_r,
        "sigma_i": sigma_i,
        "k_max": k_opt,
        "alpha_max": k_opt / beta if np.isfinite(k_opt) else np.nan,
        "c_migr": c_migr,
    }


def solve_modal_competition(
    beta: float,
    Cf: float,
    Fr: float,
    theta: float,
    theta_c: float = 0.047,
    nu: float = 0.0,
    f_sec: float = 5.0,
    m_modes: Optional[List[int]] = None,
    k_bounds: Tuple[float, float] = (0.10, 20.00),
    N_cheb: int = 36,
    Gamma: float = 4.0,
    sediment_exponent: float = 1.5,
) -> Dict[int, Dict[str, float]]:
    """Compute stability properties across transverse modes m = 1, 2, 3, 4."""
    if m_modes is None:
        m_modes = [1, 2, 3, 4]
        
    results = {}
    for m in m_modes:
        res = find_most_amplified_mode(
            beta=beta,
            Cf=Cf,
            Fr=Fr,
            theta=theta,
            theta_c=theta_c,
            nu=nu,
            f_sec=f_sec,
            mode_m=m,
            k_bounds=k_bounds,
            N_cheb=N_cheb,
            Gamma=Gamma,
            sediment_exponent=sediment_exponent,
        )
        results[m] = res
    return results


def compute_curvature_modulation_exact(
    beta: float,
    Cf: float,
    Fr: float,
    theta: float,
    nu: float,
    theta_c: float = 0.047,
    f_sec: float = 5.0,
    k_bounds: Tuple[float, float] = (0.10, 20.00),
    N_cheb: int = 36,
    Gamma: float = 4.0,
    sediment_exponent: float = 1.5,
    mode_m: int = 1,
) -> Dict[str, float]:
    """Compute exact curvature modulation E (%) by solving straight and curved eigenvalue problems directly."""
    # 1. Straight channel baseline
    res_str = find_most_amplified_mode(
        beta=beta, Cf=Cf, Fr=Fr, theta=theta, theta_c=theta_c, nu=0.0, k_bounds=k_bounds, N_cheb=N_cheb,
        Gamma=Gamma, sediment_exponent=sediment_exponent, mode_m=mode_m,
    )
    # 2. Curved channel with explicit secondary flow operator
    res_crv = find_most_amplified_mode(
        beta=beta, Cf=Cf, Fr=Fr, theta=theta, theta_c=theta_c, nu=nu, f_sec=f_sec, k_bounds=k_bounds, N_cheb=N_cheb,
        Gamma=Gamma, sediment_exponent=sediment_exponent, mode_m=mode_m,
    )
    
    sig_str = res_str["sigma_r_max"]
    sig_crv = res_crv["sigma_r_max"]
    
    if np.isfinite(sig_str) and np.isfinite(sig_crv) and abs(sig_str) > 1e-6:
        E_pct = (sig_crv - sig_str) / abs(sig_str) * 100.0
    else:
        E_pct = np.nan
        
    return {
        "sigma_r_straight": sig_str,
        "sigma_r_curved": sig_crv,
        "k_max_straight": res_str["k_max"],
        "k_max_curved": res_crv["k_max"],
        "c_migr": res_str["c_migr"],
        "E_pct": E_pct,
        "nu_beta": nu * beta,
    }
