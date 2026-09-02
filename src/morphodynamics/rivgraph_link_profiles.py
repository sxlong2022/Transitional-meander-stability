"""RivGraph ： link  s, x, y, B(s), C(s)。

：
-  RivGraph  link ；
-  link ， (s, x, y)；
-  B(s)；
-  C(s)。

 PIV  Mn(s) 。
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
    """ LineString  id。

     "id"  "link_id" ；，。
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
# LineString， id
                yield (f"{link_id}_{li}" if li > 0 else link_id, line)


def _densify_line(line: LineString, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ step (m)  LineString 。

    ：
    - s : ， (n,)
    - xs, ys : ， (n,)
    """

    if step <= 0:
        raise ValueError("step ")

    length = line.length
    if length <= 0:
        raise RuntimeError("LineString  0，")

# ，
# （ link  step  2 ， 0）
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
# 3 （n_step>=2）， 0
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
        raise ValueError("step ")

    length = line.length
    if length <= 0:
        raise RuntimeError("LineString  0，")

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
    """ (tx, ty)  (nx, ny) 。"""

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    t_norm = np.hypot(dx, dy)
    t_norm[t_norm == 0] = 1.0
    tx = dx / t_norm
    ty = dy / t_norm

# ： 90
    nx = ty
    ny = -tx

    return tx, ty, nx, ny


def _compute_curvature(xs: np.ndarray, ys: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ (xs, ys)  s  C(s)。"""

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    t_norm = np.hypot(dx, dy)
    t_norm[t_norm == 0] = 1.0
    tx = dx / t_norm
    ty = dy / t_norm

    theta = np.unwrap(np.arctan2(ty, tx))
# 
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
    """，- B(s)。

    ：
    1. （RivGraph ）；
    2. ， B；
    3. 。

    ：
    - mask:  (1=, 0=)
    - transform:  (rasterio.Affine)
    - xs, ys: 
    - nx, ny: 
    - search_halfwidth:  (m)
    - sample_spacing:  (m)
    - min_valid_fraction:  B 
    """

    h, w = mask.shape
    widths = np.full_like(xs, np.nan, dtype=float)

# 
    u_vals = np.arange(-search_halfwidth, search_halfwidth + sample_spacing, sample_spacing, dtype=float)
    if u_vals.size < 3:
        return widths

    for i, (x0, y0, nx0, ny0) in enumerate(zip(xs, ys, nx, ny)):
# 
        xu = x0 + u_vals * nx0
        yu = y0 + u_vals * ny0

# 
        rr, cc = rowcol(transform, xu, yu)
        rr = np.asarray(rr)
        cc = np.asarray(cc)
# ：
        in_bounds = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        water = np.zeros_like(u_vals, dtype=np.uint8)
        water[in_bounds] = mask[rr[in_bounds], cc[in_bounds]]

# ：
        water_frac = water.sum() / float(water.size)
        if water_frac < min_valid_fraction:
            continue

# （run-length ）
# water  0，
        padded = np.concatenate([[0], water, [0]])
        diff = np.diff(padded.astype(np.int8))
# (0→1)
        starts = np.where(diff == 1)[0]
# (1→0)
        ends = np.where(diff == -1)[0]

        if starts.size == 0 or ends.size == 0:
            continue

# （）
        lengths = ends - starts
        max_idx = np.argmax(lengths)
        run_start = starts[max_idx]
        run_end = ends[max_idx]

# u
        left_u = u_vals[run_start]
        right_u = u_vals[run_end - 1]  # run_end  0

        width_candidate = right_u - left_u
# ：
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
    """ RivGraph link ：s, x, y, B(s), C(s)。

    ：{link_id: {"s": ..., "x": ..., "y": ..., "B": ..., "C": ...}}

    ：
    - normal_search_halfwidth_m: （）， 30 （ 900m@30m ），
       Jurua-A  500–600m 。
    - min_valid_fraction: ， 0.05（5%），。
    """

    mask_path = Path(mask_raster_path)
    vec_path = Path(links_vector_path)

    if not mask_path.exists():
        raise FileNotFoundError(f": {mask_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f" RivGraph link : {vec_path}")

    with rasterio.open(mask_path) as src:
        mask_raw = src.read(1)
# 1， (1=, 0=)
        mask = (mask_raw > 0).astype(np.uint8)
        transform = src.transform
        crs = src.crs
        is_geographic = bool(crs is not None and getattr(crs, "is_geographic", False))
# ，
        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)
        if is_geographic:
            mid_lat = float((src.bounds.top + src.bounds.bottom) / 2.0)
            m_per_deg_lon, m_per_deg_lat = _meters_per_degree(mid_lat)
            base_cell = max(cell_size_x * m_per_deg_lon, cell_size_y * m_per_deg_lat)
        else:
            base_cell = max(cell_size_x, cell_size_y)

    if normal_search_halfwidth_m is None:
# 30 ，
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