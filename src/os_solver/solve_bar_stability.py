from __future__ import annotations

# stdlib
from typing import Tuple

# third-party
import numpy as np
import scipy.linalg as la

def _cheb(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """ N  Chebyshev collocation  D。
    
    Parameters
    ------
    N : int
        Chebyshev  ( N + 1)
        
    Returns
    ------
    D : np.ndarray
        First-order Chebyshev differentiation matrix (shape (N+1, N+1))
    x : np.ndarray
        ， [-0.5, 0.5]  ( (N+1,))
    """
    if N <= 0:
        return np.array([[0.0]]), np.array([0.0])
    k = np.arange(0, N + 1)
    # n = 0.5 (k=0) n = -0.5 (k=N)
    x = 0.5 * np.cos(np.pi * k / N)
    c = np.ones(N + 1)
    c[0], c[-1] = 2.0, 2.0
    c = c * ((-1.0) ** k)
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

def solve_bar_stability(
    beta: float,
    Cf: float,
    Fr: float,
    theta: float,
    theta_c: float,
    k_wavenumber: float,
    N_cheb: int = 40,
    Gamma: float = 4.0,
    sediment_exponent: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """-Exner。
    
     1D  Orr-Sommerfeld ，。
    Boundary conditions，。

    Parameters
    ------
    beta : float
         (B/H)。Parameters。
    Cf : float
        。。
    Fr : float
        Froude number。。
    theta : float
        Shields 。。
    theta_c : float
         Shields  ( 0.03 - 0.05 )。
    k_wavenumber : float
         (k = 2 * pi * B / lambda)。
        ： B ， OS  alpha_os  k = alpha_os * beta。
    N_cheb : int, default 40
        Chebyshev  collocation 。 4 * (N_cheb + 1)。
    Gamma : float, default 4.0
         (morphological slope coefficient)。。
    sediment_exponent : float, default 1.5
         ( Meyer-Peter & Müller  Engelund-Hansen )。

    Returns
    ------
    eigvals_sorted : np.ndarray
        /、 Re(sigma) 。
         eigvals_sorted[0] Dominant bar mode。
    eigvecs_sorted : np.ndarray
        ，。
    """
    # 
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if Cf <= 0.0:
        raise ValueError(f"Cf must be positive, got {Cf}")
    if Fr <= 0.0:
        raise ValueError(f"Fr must be positive, got {Fr}")
    if N_cheb < 10:
        raise ValueError(f"N_cheb is too small ({N_cheb}), needs >= 20 to ensure numerical accuracy")

    # 1.
    D, y = _cheb(N_cheb)
    D2 = D @ D
    I_val = np.eye(N_cheb + 1)
    
    n_pts = N_cheb + 1
    n_var = 4 # : u_prime, v_prime, h_prime, z_b_prime
    n_tot = n_var * n_pts
    
    A_ext = np.zeros((n_tot, n_tot), dtype=complex)
    B_ext = np.zeros((n_tot, n_tot), dtype=complex)
    
    # 2. Parameterscalculate
    # A_theta
    if theta > theta_c:
        A_theta = (2.0 * sediment_exponent * theta) / (theta - theta_c)
    else:
        A_theta = 0.0
        
    # Gamma_beta ， beta
    Gamma_beta = (Gamma / beta) * np.sqrt(theta_c / max(1e-6, theta))
    
    # 
    u_slice = slice(0 * n_pts, 1 * n_pts)
    v_slice = slice(1 * n_pts, 2 * n_pts)
    h_slice = slice(2 * n_pts, 3 * n_pts)
    zb_slice = slice(3 * n_pts, 4 * n_pts)
    
    # 3. (j = 1, ..., N_cheb - 1)
    for j in range(1, N_cheb):
        # --- (Equation 1) Flow continuity equation ---
        A_ext[0*n_pts + j, u_slice] = 1j * k_wavenumber * I_val[j, :]
        A_ext[0*n_pts + j, v_slice] = D[j, :]
        A_ext[0*n_pts + j, h_slice] = 1j * k_wavenumber * I_val[j, :]
        B_ext[0*n_pts + j, h_slice] = -I_val[j, :]
        
        # --- (Equation 2) Longitudinal (s) momentum equation ---
        A_ext[1*n_pts + j, u_slice] = (1j * k_wavenumber + 2.0 * beta * Cf) * I_val[j, :]
        A_ext[1*n_pts + j, h_slice] = (1j * k_wavenumber / (Fr**2) - beta * Cf) * I_val[j, :]
        A_ext[1*n_pts + j, zb_slice] = (1j * k_wavenumber / (Fr**2)) * I_val[j, :]
        B_ext[1*n_pts + j, u_slice] = -I_val[j, :]
        
        # --- (Equation 3) Transverse (n) momentum equation ---
        A_ext[2*n_pts + j, v_slice] = (1j * k_wavenumber + beta * Cf) * I_val[j, :]
        A_ext[2*n_pts + j, h_slice] = (1.0 / (Fr**2)) * D[j, :]
        A_ext[2*n_pts + j, zb_slice] = (1.0 / (Fr**2)) * D[j, :]
        B_ext[2*n_pts + j, v_slice] = -I_val[j, :]
        
        # --- (Equation 4) Exner ---
        A_ext[3*n_pts + j, u_slice] = 1j * k_wavenumber * A_theta * I_val[j, :]
        A_ext[3*n_pts + j, v_slice] = D[j, :]
        A_ext[3*n_pts + j, zb_slice] = -Gamma_beta * D2[j, :]
        B_ext[3*n_pts + j, zb_slice] = -I_val[j, :]
        
    # 4.
    # (a)
    for j in [0, N_cheb]:
        A_ext[0*n_pts + j, u_slice] = 1j * k_wavenumber * I_val[j, :]
        A_ext[0*n_pts + j, v_slice] = D[j, :]
        A_ext[0*n_pts + j, h_slice] = 1j * k_wavenumber * I_val[j, :]
        B_ext[0*n_pts + j, h_slice] = -I_val[j, :]
        
    # (b) Boundary conditions: du/dn = 0 (，)
    for j in [0, N_cheb]:
        row = 1 * n_pts + j
        A_ext[row, :] = 0.0
        B_ext[row, :] = 0.0
        A_ext[row, u_slice] = D[j, :]
        
    # (c) Boundary conditions: v = 0 ()
    for j in [0, N_cheb]:
        row = 2 * n_pts + j
        A_ext[row, :] = 0.0
        B_ext[row, :] = 0.0
        A_ext[row, 1*n_pts + j] = 1.0
        
    # (d) Exner Boundary conditions: dzb/dn = 0 ()
    for j in [0, N_cheb]:
        row = 3 * n_pts + j
        A_ext[row, :] = 0.0
        B_ext[row, :] = 0.0
        A_ext[row, zb_slice] = D[j, :]
        
    # 5. A * X = sigma * B * X
    eigvals, eigvecs = la.eig(A_ext, B_ext)
    
    # 6.
    # （B ），B 。
    # （|sigma| -> inf）。。
    finite_mask = np.isfinite(eigvals) & (np.abs(eigvals) < 1e5)
    eigvals_filtered = eigvals[finite_mask]
    eigvecs_filtered = eigvecs[:, finite_mask]
    
    if len(eigvals_filtered) == 0:
        return np.array([], dtype=complex), np.array([[]], dtype=complex)
        
    # （ Re(sigma)）
    idx_sort = np.argsort(np.real(eigvals_filtered))[::-1]
    eigvals_sorted = eigvals_filtered[idx_sort]
    eigvecs_sorted = eigvecs_filtered[:, idx_sort]
    
    return eigvals_sorted, eigvecs_sorted
