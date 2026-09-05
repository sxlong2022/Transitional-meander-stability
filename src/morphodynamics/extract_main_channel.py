# -*- coding: utf-8 -*-
"""Main channel trunk extraction: reconstruct continuous thalweg trunks from RivGraph link profiles.

Strategy:
  Because natural meandering rivers exhibit local multidirectional flow (bends may curve westward),
  a strict Directed Acyclic Graph (DAG) approach drops reversed links. Instead, this module searches
  for the longest simple path on an **undirected graph**.

  Algorithm:
  1. Read link endpoint coordinates and arc lengths from RivGraph profile CSVs.
  2. Perform spatial clustering on link endpoints to construct undirected graph nodes.
  3. Identify the westernmost and easternmost degree=1 nodes as source and target.
  4. Find the longest path using a modified Dijkstra algorithm on the undirected graph.
  5. Concatenate along-stream s, B, and C profiles into a continuous single-thread trunk.

  For k > 1 trunks, iteratively remove used links and repeat the graph search.

Usage:
  python -m src.morphodynamics.extract_main_channel --years 2016
  python -m src.morphodynamics.extract_main_channel
  python -m src.morphodynamics.extract_main_channel --endpoint-tol 150 --min-trunk-km 20
"""
from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PROFILE_DIR = _PROJECT_ROOT / "results" / "profiles"
TRUNK_DIR = _PROJECT_ROOT / "results" / "trunks"


# ── Geometry helpers ──────────────────────────────────────

def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """Return (m_per_deg_lon, m_per_deg_lat) for a given latitude."""
    lat_rad = np.deg2rad(float(lat_deg))
    m_per_deg_lat = (
        111132.92
        - 559.82 * np.cos(2 * lat_rad)
        + 1.175 * np.cos(4 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * np.cos(lat_rad)
        - 93.5 * np.cos(3 * lat_rad)
    )
    return float(m_per_deg_lon), float(m_per_deg_lat)


# ── Union-Find for endpoint clustering ───────────────────

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _cluster_endpoints(
    x_m: np.ndarray,
    y_m: np.ndarray,
    tol_m: float,
) -> np.ndarray:
    """Cluster endpoints spatially using grid-accelerated Union-Find.

    Returns an array of 0-based contiguous cluster IDs.
    """
    n = len(x_m)
    if n == 0:
        return np.array([], dtype=int)

    cell = tol_m * 1.01
    uf = _UnionFind(n)
    grid: Dict[Tuple[int, int], list[int]] = {}

    xmin = float(np.nanmin(x_m)) - cell
    ymin = float(np.nanmin(y_m)) - cell

    for i in range(n):
        xi, yi = float(x_m[i]), float(y_m[i])
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        gx = int((xi - xmin) / cell)
        gy = int((yi - ymin) / cell)

        tol2 = tol_m * tol_m
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (gx + dx, gy + dy)
                for j in grid.get(nb, []):
                    dd = (x_m[i] - x_m[j]) ** 2 + (y_m[i] - y_m[j]) ** 2
                    if dd <= tol2:
                        uf.union(i, j)
        grid.setdefault((gx, gy), []).append(i)

    roots = np.array([uf.find(i) for i in range(n)], dtype=int)
    _, inv = np.unique(roots, return_inverse=True)
    return inv


# ── Longest path on undirected graph ─────────────────────

def _longest_path_undirected(
    n_nodes: int,
    edges: List[Tuple[int, int, float, str]],
    source: int,
    target: int,
) -> List[str]:
    """Find the longest simple path from source to target on an undirected weighted graph.

    Uses a modified Dijkstra traversal with max-heap on a tree-like river network.
    Returns the ordered list of link_ids along the primary trunk.
    """
    if n_nodes <= 0 or len(edges) == 0:
        return []

    # Build undirected adjacency list
    adj: List[List[Tuple[int, float, str]]] = [[] for _ in range(n_nodes)]
    for u, v, w, lid in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            adj[u].append((v, w, lid))
            adj[v].append((u, w, lid))

    # Modified Dijkstra for longest path on tree-like graph
    # Maintain dist[v] = max cumulative distance using a max-heap
    INF = float("inf")
    dist = [-INF] * n_nodes
    pred_node = [-1] * n_nodes
    pred_edge = [""] * n_nodes
    dist[source] = 0.0

    # Max-heap (storing negative distances in min-heap)
    heap = [(-0.0, source)]
    visited = [False] * n_nodes

    while heap:
        neg_d, u = heapq.heappop(heap)
        d_u = -neg_d
        if visited[u]:
            continue
        visited[u] = True

        if u == target:
            break

        for v, w, lid in adj[u]:
            if visited[v]:
                continue
            new_dist = d_u + w
            if new_dist > dist[v]:
                dist[v] = new_dist
                pred_node[v] = u
                pred_edge[v] = lid
                heapq.heappush(heap, (-new_dist, v))
    # Backtrack path
    if dist[target] <= -INF / 2:
        return []

    path_edges: List[str] = []
    cur = target
    while cur != source:
        e = pred_edge[cur]
        pn = pred_node[cur]
        if pn < 0 or e == "":
            return []  # Unreachable
        path_edges.append(e)
        cur = pn

    path_edges.reverse()
    return path_edges


# ── Main trunk extraction ────────────────────────────────

def extract_trunks_from_csv(
    csv_path: str | Path,
    k_trunks: int = 1,
    endpoint_tol_m: float = 150.0,
    weight_by: str = "length",
    min_trunk_length_m: float = 5000.0,
) -> Dict[str, pd.DataFrame]:
    """Extract k longest continuous primary channel trunks from link_sBC CSV.

    Parameters
    ----------
    csv_path : str or Path
        Input link_sBC CSV file.
    k_trunks : int, default 1
        Number of primary trunks to extract.
    endpoint_tol_m : float, default 150.0
        Endpoint spatial clustering tolerance in meters.
    weight_by : str, default "length"
        Edge weighting strategy ("length" or "length_B").
    min_trunk_length_m : float, default 5000.0
        Minimum valid trunk length in meters.

    Returns
    -------
    trunks : dict of str -> pd.DataFrame
        Dictionary mapping trunk_id to DataFrame with columns [s_m, lon, lat, B_m, C_1m].
    """
    df = pd.read_csv(csv_path)
    link_ids = df["link_id"].unique()
    n_links = len(link_ids)

    if n_links == 0:
        return {}

    # Extract endpoint coordinates and summary statistics for each link
    x0 = np.full(n_links, np.nan)
    y0 = np.full(n_links, np.nan)
    x1 = np.full(n_links, np.nan)
    y1 = np.full(n_links, np.nan)
    link_len = np.full(n_links, np.nan)
    link_meanB = np.full(n_links, np.nan)
    lid_list: List[str] = []

    for i, lid in enumerate(link_ids):
        sub = df[df["link_id"] == lid].sort_values("sample_idx")
        lid_list.append(str(lid))

        lons = sub["lon"].values
        lats = sub["lat"].values
        s_vals = sub["s_m"].values
        b_vals = pd.to_numeric(sub["B_m"], errors="coerce").values

        if len(lons) >= 1:
            x0[i], y0[i] = float(lons[0]), float(lats[0])
            x1[i], y1[i] = float(lons[-1]), float(lats[-1])
        if np.isfinite(s_vals).any():
            link_len[i] = float(np.nanmax(s_vals) - np.nanmin(s_vals))
        if np.isfinite(b_vals).any():
            link_meanB[i] = float(np.nanmean(b_vals[np.isfinite(b_vals)]))

    # Convert geographic coordinates to local Cartesian meters
    lat_ref = float(np.nanmean(np.concatenate([y0[np.isfinite(y0)], y1[np.isfinite(y1)]])))
    m_lon, m_lat = _meters_per_degree(lat_ref)

    pts_x_m = np.concatenate([x0 * m_lon, x1 * m_lon])
    pts_y_m = np.concatenate([y0 * m_lat, y1 * m_lat])

    # Spatial clustering of link endpoints
    valid = np.isfinite(pts_x_m) & np.isfinite(pts_y_m)
    pts_x_safe = np.where(valid, pts_x_m, 1e30)
    pts_y_safe = np.where(valid, pts_y_m, 1e30)

    cluster_ids = _cluster_endpoints(pts_x_safe, pts_y_safe, endpoint_tol_m)
    start_node = cluster_ids[:n_links]
    end_node = cluster_ids[n_links:]
    n_nodes = int(np.max(cluster_ids)) + 1 if cluster_ids.size else 0

    # Compute node longitude coordinates for source/target selection
    node_x_deg = np.full(n_nodes, np.nan)
    for nid in range(n_nodes):
        mm = cluster_ids == nid
        if not np.any(mm):
            continue
        x_deg = np.concatenate([x0, x1])[mm]
        node_x_deg[nid] = float(np.nanmean(x_deg[np.isfinite(x_deg)])) if np.isfinite(x_deg).any() else np.nan

    # Build undirected weighted edges
    def _edge_weight(i: int) -> float:
        L = float(link_len[i]) if np.isfinite(link_len[i]) else 0.0
        if weight_by == "length_B":
            Bm = float(link_meanB[i]) if np.isfinite(link_meanB[i]) else 0.0
            return L * Bm
        return L

    edges: List[Tuple[int, int, float, str]] = []
    node_degree = np.zeros(n_nodes, dtype=int)

    for i, lid in enumerate(lid_list):
        u = int(start_node[i])
        v = int(end_node[i])
        if u == v:
            continue  # Self-loop
        w = _edge_weight(i)
        if w <= 0:
            w = 1.0  # Minimum edge weight
        edges.append((u, v, w, lid))
        node_degree[u] += 1
        node_degree[v] += 1

    # Select source and target: degree=1 nodes with extreme longitudes
    endpoint_nodes = np.where(node_degree == 1)[0]
    if endpoint_nodes.size == 0:
        # Degenerate case: fallback to extreme nodes across all valid longitudes
        finite_x = np.isfinite(node_x_deg)
        endpoint_nodes = np.where(finite_x)[0]

    if endpoint_nodes.size == 0:
        return {}

    ep_x = node_x_deg[endpoint_nodes]
    # Source: westernmost endpoint (minimum longitude)
    source_idx = endpoint_nodes[int(np.nanargmin(ep_x))]
    # Target: easternmost endpoint (maximum longitude)
    target_idx = endpoint_nodes[int(np.nanargmax(ep_x))]

    if source_idx == target_idx:
        return {}

    # Iteratively extract k trunks
    remaining = list(edges)
    trunks: Dict[str, pd.DataFrame] = {}
    lid_to_idx = {lid: i for i, lid in enumerate(lid_list)}

    for k in range(k_trunks):
        path = _longest_path_undirected(
            n_nodes=n_nodes,
            edges=remaining,
            source=source_idx,
            target=target_idx,
        )
        if not path:
            break

        # Remove already traversed links
        used = set(path)
        remaining = [(u, v, w, lid) for u, v, w, lid in remaining if lid not in used]

        # Concatenate along-stream profiles ensuring consistent link orientation
        # Build link -> (u, v) mapping to determine traversal direction
        # 构建 link→(u,v) 映射以确定遍历方向
        link_edge_map: Dict[str, Tuple[int, int]] = {}
        for u, v, w, lid in edges:
            link_edge_map[lid] = (u, v)

        s_cat, lon_cat, lat_cat, B_cat, C_cat = [], [], [], [], []
        s_offset = 0.0

        # Reconstruct node path sequence from link path
        node_path = [source_idx]
        for lid in path:
            u, v = link_edge_map[lid]
            prev = node_path[-1]
            if u == prev:
                node_path.append(v)
            elif v == prev:
                node_path.append(u)
            else:
                # Discontinuous path; break
                break

        for step, lid in enumerate(path):
            idx = lid_to_idx.get(lid)
            if idx is None:
                continue
            sub = df[df["link_id"] == int(lid)].sort_values("sample_idx")
            if sub.empty:
                continue

            s_vals = sub["s_m"].values.astype(float)
            lons = sub["lon"].values.astype(float)
            lats = sub["lat"].values.astype(float)
            b_vals = pd.to_numeric(sub["B_m"], errors="coerce").values
            c_vals = pd.to_numeric(sub["C_1m"], errors="coerce").values

            # Determine traversal direction for this link
            # Reverse link if entered from the end_node side
            u_edge, v_edge = link_edge_map.get(lid, (start_node[idx], end_node[idx]))
            if step < len(node_path) - 1:
                path_from = node_path[step]
                path_to = node_path[step + 1]
                # start_node corresponds to CSV row 0
                link_start = start_node[idx]
                need_reverse = (path_from != link_start) if (link_start in (u_edge, v_edge)) else False
            else:
                need_reverse = False

            if need_reverse:
                s_vals = s_vals[::-1]
                max_s = float(np.nanmax(s_vals)) if np.isfinite(s_vals).any() else 0.0
                s_vals = max_s - s_vals
                lons = lons[::-1]
                lats = lats[::-1]
                b_vals = b_vals[::-1]
                c_vals = c_vals[::-1]

            s_cat.append(s_vals + s_offset)
            lon_cat.append(lons)
            lat_cat.append(lats)
            B_cat.append(b_vals)
            C_cat.append(c_vals)

            if np.isfinite(s_vals).any():
                s_offset += float(np.nanmax(s_vals))

        if not s_cat:
            continue

        trunk_s = np.concatenate(s_cat)
        trunk_len = float(np.nanmax(trunk_s)) if trunk_s.size > 0 else 0.0

        if trunk_len < min_trunk_length_m:
            continue

        trunk_id = f"trunk_{k}"
        trunks[trunk_id] = pd.DataFrame({
            "s_m": np.concatenate(s_cat),
            "lon": np.concatenate(lon_cat),
            "lat": np.concatenate(lat_cat),
            "B_m": np.concatenate(B_cat),
            "C_1m": np.concatenate(C_cat),
        })

    return trunks


# ── CLI ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract main channel trunk(s) from RivGraph link profiles.",
    )
    parser.add_argument(
        "--years", nargs="*", type=int, default=None,
        help="Year(s) to process (default: all available)",
    )
    parser.add_argument(
        "--k-trunks", type=int, default=1,
        help="Number of trunks to extract per year (default: 1)",
    )
    parser.add_argument(
        "--endpoint-tol", type=float, default=150.0,
        help="Endpoint clustering tolerance in meters (default: 150)",
    )
    parser.add_argument(
        "--min-trunk-km", type=float, default=20.0,
        help="Minimum trunk length in km (default: 20)",
    )
    parser.add_argument(
        "--weight-by", default="length",
        choices=["length", "length_B"],
        help="Edge weight strategy (default: length)",
    )
    args = parser.parse_args()

    TRUNK_DIR.mkdir(parents=True, exist_ok=True)

    if args.years is None:
        csvs = sorted(PROFILE_DIR.glob("Gaocun-Sunkou_*_link_sBC.csv"))
        years = []
        for csv in csvs:
            try:
                y = int(csv.stem.split("_")[1])
                years.append(y)
            except (IndexError, ValueError):
                pass
    else:
        years = args.years

    min_trunk_m = args.min_trunk_km * 1000.0

    print(f"Processing {len(years)} year(s)")
    print(f"k_trunks={args.k_trunks}, endpoint_tol={args.endpoint_tol}m, "
          f"min_trunk={args.min_trunk_km}km, weight_by={args.weight_by}")
    print()

    for year in years:
        csv_path = PROFILE_DIR / f"Gaocun-Sunkou_{year}_link_sBC.csv"
        if not csv_path.exists():
            print(f"[SKIP] {year}: no profile CSV")
            continue

        trunks = extract_trunks_from_csv(
            csv_path=str(csv_path),
            k_trunks=args.k_trunks,
            endpoint_tol_m=args.endpoint_tol,
            weight_by=args.weight_by,
            min_trunk_length_m=min_trunk_m,
        )

        if not trunks:
            print(f"[WARN] {year}: no trunks found (min_length={args.min_trunk_km}km)")
            continue

        for trunk_id, trunk_df in trunks.items():
            out_path = TRUNK_DIR / f"Gaocun-Sunkou_{year}_{trunk_id}.csv"
            trunk_df.to_csv(out_path, index=False, float_format="%.6f")

            trunk_len_km = float(trunk_df["s_m"].max()) / 1000.0
            b_mean = float(pd.to_numeric(trunk_df["B_m"], errors="coerce").mean())
            n_pts = len(trunk_df)
            lon_range = f"[{trunk_df['lon'].min():.4f}, {trunk_df['lon'].max():.4f}]"

            print(f"[{year}] {trunk_id}: {trunk_len_km:.1f} km, {n_pts} pts, "
                  f"B_mean={b_mean:.0f}m, lon={lon_range}")

    print("\nDone.")


if __name__ == "__main__":
    main()
