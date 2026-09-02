""" s–B–C ：。

 C&G  link_sBCMn_pipeline.py，
 Subproject-3 ：
  -  PIV （Mn）
  -  RivGraph link （s, B, C）
  -  s–B–C 

（）：

    python -m src.morphodynamics.link_sBC_pipeline \
        --site Gaocun-Sunkou \
        --mask-level 2 \
        --mask-raster data/GEOTIFFS/Gaocun-Sunkou/Gaocun-Sunkou_2016_mask2.tif \
        --links-vector results/RivGraph/Gaocun-Sunkou/mask2/links.shp \
        --step-m 100 \
        --export-csv results/profiles/Gaocun-Sunkou_2016_link_sBC.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np


# ── sys.path hack for standalone execution ────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def compute_link_sBC_for_site(
    site: str,
    mask_level: int,
    mask_raster_path: str,
    links_vector_path: str,
    step_m: float = 100.0,
    export_csv_path: str | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """/ RivGraph link  s-B-C 。

    
    ------
    site : str
        ， "Gaocun-Sunkou"。
    mask_level : int
        ， 2  Mask2。
    mask_raster_path : str
         (.tif)。
    links_vector_path : str
        RivGraph link  (.shp / .gpkg)。
    step_m : float
         link （）。
    export_csv_path : str, optional
        ， link  s-B-C  CSV。

    
    ------
    link_sBC : {link_id: {"s", "x", "y", "B", "C"}}
    """
    try:
        from src.morphodynamics.rivgraph_link_profiles import compute_link_profiles
    except ImportError:
        raise ImportError(
            "rivgraph_link_profiles not found. This module requires the "
            "RivGraph geometry profiling code. Please copy "
            "rivgraph_link_profiles.py from the C&G project."
        )

    # Check input files
    if not Path(mask_raster_path).exists():
        raise FileNotFoundError(f"Mask raster not found: {mask_raster_path}")
    if not Path(links_vector_path).exists():
        raise FileNotFoundError(f"Links vector not found: {links_vector_path}")

    # Compute RivGraph link geometric profiles (s, x, y, B, C)
    link_geom = compute_link_profiles(
        mask_raster_path=mask_raster_path,
        links_vector_path=links_vector_path,
        step_m=step_m,
    )

    if not link_geom:
        raise RuntimeError(
            "No link profiles obtained from RivGraph vector. "
            "Check input data."
        )

    print(f"Computed {len(link_geom)} link profiles for {site} mask{mask_level}.")

    # Optional CSV export
    if export_csv_path is not None:
        _export_link_sBC_csv(link_geom, site, mask_level, step_m, export_csv_path)

    return link_geom


def _export_link_sBC_csv(
    link_profiles: Dict[str, Dict[str, np.ndarray]],
    site: str,
    mask_level: int,
    step_m: float,
    out_path: str | Path,
) -> None:
    """ per-link  s-B-C  CSV。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["link_id", "sample_idx", "s", "x", "y", "B", "C"]
    link_ids = sorted(link_profiles.keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for link_id in link_ids:
            prof = link_profiles[link_id]
            s = np.asarray(prof["s"], dtype=float)
            x = np.asarray(prof["x"], dtype=float)
            y = np.asarray(prof["y"], dtype=float)
            B = np.asarray(prof["B"], dtype=float)
            C = np.asarray(prof["C"], dtype=float)

            for i in range(s.size):
                writer.writerow({
                    "link_id": str(link_id),
                    "sample_idx": i,
                    "s": f"{s[i]:.2f}",
                    "x": f"{x[i]:.6f}",
                    "y": f"{y[i]:.6f}",
                    "B": f"{B[i]:.2f}" if np.isfinite(B[i]) else "",
                    "C": f"{C[i]:.8f}" if np.isfinite(C[i]) else "",
                })

    print(f"[SAVED] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute s-B-C profiles along RivGraph links for Gaocun-Sunkou.",
    )
    parser.add_argument("--site", default="Gaocun-Sunkou",
                        help="Site name (default: Gaocun-Sunkou)")
    parser.add_argument("--mask-level", type=int, default=2,
                        help="Mask level (default: 2)")
    parser.add_argument("--mask-raster", required=True,
                        help="Binary water mask raster path (.tif)")
    parser.add_argument("--links-vector", required=True,
                        help="RivGraph link vector file path (.shp / .gpkg)")
    parser.add_argument("--step-m", type=float, default=100.0,
                        help="Sampling interval along link (meters)")
    parser.add_argument("--export-csv", default=None,
                        help="Output CSV path for flat s-B-C profiles")

    args = parser.parse_args()

    compute_link_sBC_for_site(
        site=args.site,
        mask_level=args.mask_level,
        mask_raster_path=args.mask_raster,
        links_vector_path=args.links_vector,
        step_m=args.step_m,
        export_csv_path=args.export_csv,
    )


if __name__ == "__main__":
    main()