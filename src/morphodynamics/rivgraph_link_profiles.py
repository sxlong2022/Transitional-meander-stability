"""RivGraph channel link profiles: along-link computation of s, x, y, B(s), and C(s).

Core geometric routines:
- Reads binary water mask GeoTIFF and RivGraph link shapefiles.
- Densifies sampling along each link centerline by arc length s to produce (s, x, y).
- Measures local channel width B(s) via normal ray intersections with water boundaries.
- Calculates centerline curvature C(s) from smoothed coordinate derivatives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import fiona
import numpy as np
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import LineString, MultiLineString, shape


def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    lat_rad = np.deg2rad(float(lat_deg))
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad) - 0.0023 * np.cos(6 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad)
    return float(m_per_deg_lon), float(m_per_deg_lat)


def _iter_lines_from_vector(path: Path) -> Iterable[Tuple[str, LineString]]:
    """Iterate over LineString geometries and identifiers from a vector file.

    Prioritizes "id" or "link_id" fields; falls back to sequential indices.
    """

    with fiona.open(path) as src:
        for idx, feat in enumerate(src):
            geom = feat["geometry"]
            if geom is None:
                continue
            shp = shape(geom)
            lines: list[LineString] = []
            if isinstance(shp, LineString):
                lines = [shp]
            elif isinstance(shp, MultiLineString):
                lines = list(shp.geoms)
            if not lines:
                continue

            props = feat.get("properties", {}) or feat
            link_id = (
                str(props.get("id"))
                if props.get("id") is not None
                else (
                    str(props.get("link_id"))
                    if props.get("link_id") is not None
                    else str(idx)
                )
            )

            for li, line in enumerate(lines):
                # 若一个要素中包含多条 LineString，则在 id 后附加子索引
                yield (f"{link_id}_{li}" if li > 0 else link_id, line)


def _densify_line(line: LineString, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按给定弧长间距 step (m) 对 LineString 进行加密采样。

    返回：
    - s : 沿程距离数组，形状 (n,)
    - xs, ys : 采样点坐标，形状 (n,)
    """

    if step <= 0:
        raise ValueError("step 必须为正数")

    length = line.length
    if length <= 0:
        raise RuntimeError("LineString 长度为 0，无法加密采样")

    # 如果原始几何本身已经包含足够顶点，优先保留这些顶点
    # （避免在短 link 上因 step 过大导致只有 2 个采样点，从而曲率恒为 0）
    coords = np.asarray(line.coords, dtype=float)
    if coords.ndim == 2 and coords.shape[0] >= 3:
        xs0 = coords[:, 0]
        ys0 = coords[:, 1]
        ds0 = np.hypot(np.diff(xs0), np.diff(ys0))
        keep = np.concatenate([[True], ds0 > 0])
        xs0 = xs0[keep]
        ys0 = ys0[keep]
        if xs0.size >= 3:
            ds = np.hypot(np.diff(xs0), np.diff(ys0))
            s = np.concatenate([[0.0], np.cumsum(ds)])
            return s, xs0, ys0

    n_step = int(np.ceil(length / step))
    # 至少 3 个点（n_step>=2），否则曲率计算会退化为全 0
    n_step = max(n_step, 2)
    s = np.linspace(0.0, length, n_step + 1)
    xs = np.empty_like(s)
    ys = np.empty_like(s)
    for i, si in enumerate(s):
        pt = line.interpolate(si)
        xs[i] = pt.x
        ys[i] = pt.y
    return s, xs, ys


def _densify_line_always(line: LineString, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if step <= 0:
        raise ValueError("step 必须为正数")

    length = line.length
    if length <= 0:
        raise RuntimeError("LineString 长度为 0，无法加密采样")

    n_step = int(np.ceil(length / step))
    n_step = max(n_step, 2)
    s = np.linspace(0.0, length, n_step + 1)
    xs = np.empty_like(s)
    ys = np.empty_like(s)
    for i, si in enumerate(s):
        pt = line.interpolate(si)
        xs[i] = pt.x
        ys[i] = pt.y
    return s, xs, ys


def _compute_tangent_normal(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据采样点坐标计算切向 (tx, ty) 和法向 (nx, ny) 单位向量。"""

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    t_norm = np.hypot(dx, dy)
    t_norm[t_norm == 0] = 1.0
    tx = dx / t_norm
    ty = dy / t_norm

    # 法向：切向顺时针旋转 90 度
    nx = ty
    ny = -tx

    return tx, ty, nx, ny


def _compute_curvature(xs: np.ndarray, ys: np.ndarray, s: np.ndarray) -> np.ndarray:
    """根据 (xs, ys) 与弧长 s 计算离散曲率 C(s)。"""

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    t_norm = np.hypot(dx, dy)
    t_norm[t_norm == 0] = 1.0
    tx = dx / t_norm
    ty = dy / t_norm

    theta = np.unwrap(np.arctan2(ty, tx))
    # 使用弧长作为自变量对方位角求导
    dtheta_ds = np.gradient(theta, s)
    return dtheta_ds


def _sample_width_along_normal(
    mask: np.ndarray,
    transform,
    xs: np.ndarray,
    ys: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    search_halfwidth: float,
    sample_spacing: float,
    min_valid_fraction: float = 0.1,
) -> np.ndarray:
    """沿每个采样点的法线方向，通过水-陆交界点计算局地宽度 B(s)。

    改进版：
    1. 不再强制要求中心点恰好在水体内（RivGraph 骨架线常有像元边界偏移）；
    2. 寻找法线上最长连续水体区段，以该区段长度作为 B；
    3. 仅当连续水体区段足够长时才返回有效值。

    参数：
    - mask: 二值水体掩膜数组 (1=水, 0=非水)
    - transform: 栅格仿射变换 (rasterio.Affine)
    - xs, ys: 采样点坐标
    - nx, ny: 法向单位向量
    - search_halfwidth: 法线方向搜索半宽 (m)
    - sample_spacing: 法线方向采样间距 (m)
    - min_valid_fraction: 法线上水体像元占比低于该值时认为 B 无效
    """

    h, w = mask.shape
    widths = np.full_like(xs, np.nan, dtype=float)

    # 预先构建相对坐标
    u_vals = np.arange(-search_halfwidth, search_halfwidth + sample_spacing, sample_spacing, dtype=float)
    if u_vals.size < 3:
        return widths

    for i, (x0, y0, nx0, ny0) in enumerate(zip(xs, ys, nx, ny)):
        # 构造法线上采样点
        xu = x0 + u_vals * nx0
        yu = y0 + u_vals * ny0

        # 转换到行列索引
        rr, cc = rowcol(transform, xu, yu)
        rr = np.asarray(rr)
        cc = np.asarray(cc)
        # 越界处理：标记为陆地
        in_bounds = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        water = np.zeros_like(u_vals, dtype=np.uint8)
        water[in_bounds] = mask[rr[in_bounds], cc[in_bounds]]

        # 质量控制：整条法线上水体比例过低则跳过
        water_frac = water.sum() / float(water.size)
        if water_frac < min_valid_fraction:
            continue

        # 寻找最长连续水体区段（run-length 方法）
        # 在 water 数组首尾各添加一个 0，方便检测边界
        padded = np.concatenate([[0], water, [0]])
        diff = np.diff(padded.astype(np.int8))
        # 上升沿 (0→1) 位置
        starts = np.where(diff == 1)[0]
        # 下降沿 (1→0) 位置
        ends = np.where(diff == -1)[0]

        if starts.size == 0 or ends.size == 0:
            continue

        # 每段长度（以采样点数计）
        lengths = ends - starts
        max_idx = np.argmax(lengths)
        run_start = starts[max_idx]
        run_end = ends[max_idx]

        # 计算该区段对应的 u 范围
        left_u = u_vals[run_start]
        right_u = u_vals[run_end - 1]  # run_end 指向区段结束后第一个 0

        width_candidate = right_u - left_u
        # 基本合理性检查：宽度应为正
        if width_candidate > 0:
            widths[i] = width_candidate

    return widths


def compute_link_profiles(
    mask_raster_path: str,
    links_vector_path: str,
    step_m: float = 100.0,
    normal_search_halfwidth_m: float | None = None,
    sample_spacing_factor: float = 0.5,
    min_valid_fraction: float = 0.05,
) -> Dict[str, Dict[str, np.ndarray]]:
    """为 RivGraph link 计算几何剖面：s, x, y, B(s), C(s)。

    返回字典：{link_id: {"s": ..., "x": ..., "y": ..., "B": ..., "C": ...}}

    参数说明：
    - normal_search_halfwidth_m: 法线搜索半宽（米），默认 30 个像元（约 900m@30m 分辨率），
      足以覆盖 Jurua-A 主河道 500–600m 宽度。
    - min_valid_fraction: 法线上水体比例阈值，默认 0.05（5%），放宽以适应宽搜索范围。
    """

    mask_path = Path(mask_raster_path)
    vec_path = Path(links_vector_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"找不到水体掩膜栅格: {mask_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"找不到 RivGraph link 矢量文件: {vec_path}")

    with rasterio.open(mask_path) as src:
        mask_raw = src.read(1)
        # 将任意非零值归一为 1，确保后续按二值掩膜处理 (1=水, 0=非水)
        mask = (mask_raw > 0).astype(np.uint8)
        transform = src.transform
        crs = src.crs
        is_geographic = bool(crs is not None and getattr(crs, "is_geographic", False))
        # 像元大小，用于设置默认搜索半宽与采样间距
        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)
        if is_geographic:
            mid_lat = float((src.bounds.top + src.bounds.bottom) / 2.0)
            m_per_deg_lon, m_per_deg_lat = _meters_per_degree(mid_lat)
            base_cell = max(cell_size_x * m_per_deg_lon, cell_size_y * m_per_deg_lat)
        else:
            base_cell = max(cell_size_x, cell_size_y)

    if normal_search_halfwidth_m is None:
        # 默认搜索半宽取 30 个像元宽度，确保覆盖整个河道宽度
        normal_search_halfwidth_m = 30.0 * base_cell

    sample_spacing = max(base_cell * sample_spacing_factor, base_cell * 0.25)

    profiles: Dict[str, Dict[str, np.ndarray]] = {}

    for link_id, line in _iter_lines_from_vector(vec_path):
        if is_geographic:
            coords = np.asarray(line.coords, dtype=float)
            xs0 = coords[:, 0]
            ys0 = coords[:, 1]
            lat0 = float(np.nanmean(ys0)) if np.isfinite(ys0).any() else 0.0
            m_per_deg_lon0, m_per_deg_lat0 = _meters_per_degree(lat0)

            line_m = LineString(list(zip(xs0 * m_per_deg_lon0, ys0 * m_per_deg_lat0)))
            s, xm, ym = _densify_line_always(line_m, step=step_m)

            xs = xm / m_per_deg_lon0
            ys = ym / m_per_deg_lat0

            _, _, nx_m, ny_m = _compute_tangent_normal(xm, ym)
            nx = nx_m / m_per_deg_lon0
            ny = ny_m / m_per_deg_lat0
            C = _compute_curvature(xm, ym, s)
        else:
            s, xs, ys = _densify_line_always(line, step=step_m)
            _, _, nx, ny = _compute_tangent_normal(xs, ys)
            C = _compute_curvature(xs, ys, s)

        B = _sample_width_along_normal(
            mask=mask,
            transform=transform,
            xs=xs,
            ys=ys,
            nx=nx,
            ny=ny,
            search_halfwidth=normal_search_halfwidth_m,
            sample_spacing=sample_spacing,
            min_valid_fraction=min_valid_fraction,
        )

        profiles[link_id] = {
            "s": s,
            "x": xs,
            "y": ys,
            "B": B,
            "C": C,
        }

    return profiles
