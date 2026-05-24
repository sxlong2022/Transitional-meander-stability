from __future__ import annotations

# stdlib
from typing import Tuple

# third-party
import numpy as np
import scipy.linalg as la

def _cheb(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成 N 阶 Chebyshev collocation 点和一阶微分矩阵 D。
    
    参数
    ------
    N : int
        Chebyshev 阶数 (节点数为 N + 1)
        
    返回
    ------
    D : np.ndarray
        一阶 Chebyshev 微分矩阵 (形状为 (N+1, N+1))
    x : np.ndarray
        横向坐标节点数组，已归一化到 [-0.5, 0.5] 区间 (形状为 (N+1,))
    """
    if N <= 0:
        return np.array([[0.0]]), np.array([0.0])
    k = np.arange(0, N + 1)
    # 横向坐标从右岸 n = 0.5 (k=0) 映射到左岸 n = -0.5 (k=N)
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
    """求解二维平面深度平均浅水方程-Exner耦合系统特征值的广义本征值求解器。
    
    该求解器用于替代原有的 1D 垂向 Orr-Sommerfeld 求解器，建立符合物理一致性的河流沙洲稳定性理论模型。
    通过横向边界条件置换和数值奇异值过滤，稳健求解二维沙洲不稳定性增长率谱。

    参数
    ------
    beta : float
        通道宽深比 (B/H)。作为核心控制参数决定横向微分项的缩放。
    Cf : float
        阻力系数。调节水流与床面响应之间的相位滞后。
    Fr : float
        Froude 数。调节重力势能与水流惯性项。
    theta : float
        Shields 数。表征无量纲底面剪切力强度。
    theta_c : float
        泥沙起动的临界 Shields 数 (通常在 0.03 - 0.05 之间)。
    k_wavenumber : float
        流向无量纲波数 (k = 2 * pi * B / lambda)。
        注意：该波数以河宽 B 归一化，与垂直 OS 波数 alpha_os 的转换关系为 k = alpha_os * beta。
    N_cheb : int, 默认 40
        Chebyshev  collocation 阶数。总离散矩阵维度为 4 * (N_cheb + 1)。
    Gamma : float, 默认 4.0
        侧坡重力输沙系数 (morphological slope coefficient)。控制沙洲的高频短波截止。
    sediment_exponent : float, 默认 1.5
        输沙公式指数 (如 Meyer-Peter & Müller 或 Engelund-Hansen 公式的幂指数)。

    返回
    ------
    eigvals_sorted : np.ndarray
        已过滤数值无穷/病态奇异值、并按增长率实部 Re(sigma) 降序排列的一维本征值数组。
        首个元素 eigvals_sorted[0] 即对应系统的最不稳定主导模态。
    eigvecs_sorted : np.ndarray
        排序后对应的特征向量矩阵，每一列对应一个本征值。
    """
    # 物理与数值边界校验
    if beta <= 0.0:
        raise ValueError(f"beta 必须为正数，得到 {beta}")
    if Cf <= 0.0:
        raise ValueError(f"Cf 必须为正数，得到 {Cf}")
    if Fr <= 0.0:
        raise ValueError(f"Fr 必须为正数，得到 {Fr}")
    if N_cheb < 10:
        raise ValueError(f"N_cheb 离散点数过小 ({N_cheb})，建议 >= 20 以确保 Chebyshev 谱精度")

    # 1. 离散格点与微分算子准备
    D, y = _cheb(N_cheb)
    D2 = D @ D
    I_val = np.eye(N_cheb + 1)
    
    n_pts = N_cheb + 1
    n_var = 4 # 四个物理分量: u_prime, v_prime, h_prime, z_b_prime
    n_tot = n_var * n_pts
    
    A_ext = np.zeros((n_tot, n_tot), dtype=complex)
    B_ext = np.zeros((n_tot, n_tot), dtype=complex)
    
    # 2. 泥沙动力学线性化闭合参数计算
    # A_theta 表示流速扰动对流向输沙扰动的敏感反馈因子
    if theta > theta_c:
        A_theta = (2.0 * sediment_exponent * theta) / (theta - theta_c)
    else:
        A_theta = 0.0
        
    # Gamma_beta 为无量纲化的侧向重力输沙扩散系数，包含宽深比 beta 的调节作用
    Gamma_beta = (Gamma / beta) * np.sqrt(theta_c / max(1e-6, theta))
    
    # 状态向量中各物理量区块的切片索引
    u_slice = slice(0 * n_pts, 1 * n_pts)
    v_slice = slice(1 * n_pts, 2 * n_pts)
    h_slice = slice(2 * n_pts, 3 * n_pts)
    zb_slice = slice(3 * n_pts, 4 * n_pts)
    
    # 3. 填充控制方程内点 (j = 1, ..., N_cheb - 1)
    for j in range(1, N_cheb):
        # --- (Equation 1) 水流连续方程 ---
        A_ext[0*n_pts + j, u_slice] = 1j * k_wavenumber * I_val[j, :]
        A_ext[0*n_pts + j, v_slice] = D[j, :]
        A_ext[0*n_pts + j, h_slice] = 1j * k_wavenumber * I_val[j, :]
        B_ext[0*n_pts + j, h_slice] = -I_val[j, :]
        
        # --- (Equation 2) 纵向 (s) 动量方程 ---
        A_ext[1*n_pts + j, u_slice] = (1j * k_wavenumber + 2.0 * beta * Cf) * I_val[j, :]
        A_ext[1*n_pts + j, h_slice] = (1j * k_wavenumber / (Fr**2) - beta * Cf) * I_val[j, :]
        A_ext[1*n_pts + j, zb_slice] = (1j * k_wavenumber / (Fr**2)) * I_val[j, :]
        B_ext[1*n_pts + j, u_slice] = -I_val[j, :]
        
        # --- (Equation 3) 横向 (n) 动量方程 ---
        A_ext[2*n_pts + j, v_slice] = (1j * k_wavenumber + beta * Cf) * I_val[j, :]
        A_ext[2*n_pts + j, h_slice] = (1.0 / (Fr**2)) * D[j, :]
        A_ext[2*n_pts + j, zb_slice] = (1.0 / (Fr**2)) * D[j, :]
        B_ext[2*n_pts + j, v_slice] = -I_val[j, :]
        
        # --- (Equation 4) Exner 床沙连续方程 ---
        A_ext[3*n_pts + j, u_slice] = 1j * k_wavenumber * A_theta * I_val[j, :]
        A_ext[3*n_pts + j, v_slice] = D[j, :]
        A_ext[3*n_pts + j, zb_slice] = -Gamma_beta * D2[j, :]
        B_ext[3*n_pts + j, zb_slice] = -I_val[j, :]
        
    # 4. 填充两岸边界处的守恒与代数约束行
    # (a) 水流连续方程在河岸依然完全成立
    for j in [0, N_cheb]:
        A_ext[0*n_pts + j, u_slice] = 1j * k_wavenumber * I_val[j, :]
        A_ext[0*n_pts + j, v_slice] = D[j, :]
        A_ext[0*n_pts + j, h_slice] = 1j * k_wavenumber * I_val[j, :]
        B_ext[0*n_pts + j, h_slice] = -I_val[j, :]
        
    # (b) 纵向动量方程河岸边界条件: du/dn = 0 (侧向自由滑动边界，零摩阻传导)
    for j in [0, N_cheb]:
        row = 1 * n_pts + j
        A_ext[row, :] = 0.0
        B_ext[row, :] = 0.0
        A_ext[row, u_slice] = D[j, :]
        
    # (c) 横向动量方程河岸边界条件: v = 0 (固体壁无穿透条件)
    for j in [0, N_cheb]:
        row = 2 * n_pts + j
        A_ext[row, :] = 0.0
        B_ext[row, :] = 0.0
        A_ext[row, 1*n_pts + j] = 1.0
        
    # (d) Exner 床沙方程河岸边界条件: dzb/dn = 0 (零侧向泥沙通量在无穿透条件下的化简)
    for j in [0, N_cheb]:
        row = 3 * n_pts + j
        A_ext[row, :] = 0.0
        B_ext[row, :] = 0.0
        A_ext[row, zb_slice] = D[j, :]
        
    # 5. 调用特征值求解器求解 A * X = sigma * B * X
    eigvals, eigvecs = la.eig(A_ext, B_ext)
    
    # 6. 数值奇异值过滤与物理主模筛选
    # 由于边界行无时间导数项（B 的对应行为零），B 矩阵退化为奇异矩阵。
    # 这会产生大量由代数约束引起的数值无穷奇异本征值（|sigma| -> inf）。我们在此将其剔除。
    finite_mask = np.isfinite(eigvals) & (np.abs(eigvals) < 1e5)
    eigvals_filtered = eigvals[finite_mask]
    eigvecs_filtered = eigvecs[:, finite_mask]
    
    if len(eigvals_filtered) == 0:
        return np.array([], dtype=complex), np.array([[]], dtype=complex)
        
    # 对过滤后的物理本征值按实部（扰动增长率 Re(sigma)）降序排列
    idx_sort = np.argsort(np.real(eigvals_filtered))[::-1]
    eigvals_sorted = eigvals_filtered[idx_sort]
    eigvecs_sorted = eigvecs_filtered[:, idx_sort]
    
    return eigvals_sorted, eigvecs_sorted
