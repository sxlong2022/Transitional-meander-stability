# -*- coding: utf-8 -*-
"""Step 2.3:  DSWE  RivGraph 。

 DSWE mask:
1. RivGraph river() -> compute_network() -> to_geovectors("network", "shp")
    links.shp / nodes.shp
2.  rivgraph_link_profiles.compute_link_profiles()  s, B(s), C(s)
3.  link_sBC  CSV

:
# 
  python -m src.morphodynamics.run_rivgraph_batch --years 2016

# 
  python -m src.morphodynamics.run_rivgraph_batch

# exit_sides
  python -m src.morphodynamics.run_rivgraph_batch --exit-sides WN
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

# ── paths ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MASK_DIR = _PROJECT_ROOT / "data" / "GEOTIFFS" / "Gaocun-Sunkou"
RIVGRAPH_DIR = _PROJECT_ROOT / "results" / "RivGraph" / "Gaocun-Sunkou"
PROFILE_DIR = _PROJECT_ROOT / "results" / "profiles"


def run_rivgraph_for_year(
    year: int,
    exit_sides: str = "WE",
    step_m: float = 100.0,
    normal_search_halfwidth_m: float | None = None,
    skip_if_exists: bool = True,
) -> Path | None:
    """Run RivGraph + profile extraction for a single year.

    Returns path to exported CSV, or None on failure.
    """
    # Find mask file
    pattern = f"Gaocun-Sunkou_{year}_*_mask2.tif"
    masks = sorted(MASK_DIR.glob(pattern))
    if not masks:
        print(f"[SKIP] {year}: no mask file matching {pattern}")
        return None
    mask_path = masks[0]

    # Output dirs
    year_dir = RIVGRAPH_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    links_shp = year_dir / f"Gaocun-Sunkou_{year}_links.shp"
    csv_path = PROFILE_DIR / f"Gaocun-Sunkou_{year}_link_sBC.csv"

    # Step 1: RivGraph network extraction
    if skip_if_exists and links_shp.exists():
        print(f"[SKIP] {year}: links.shp already exists")
    else:
        print(f"[{year}] Running RivGraph on {mask_path.name} ...")
        t0 = time.time()
        try:
            from rivgraph.classes import river

            net = river(
                name=f"Gaocun-Sunkou_{year}",
                path_to_mask=str(mask_path),
                results_folder=str(year_dir),
                exit_sides=exit_sides,
                verbose=False,
            )
            net.compute_network()
            net.to_geovectors(export="network", ftype="shp")
            dt = time.time() - t0
            print(f"[{year}] RivGraph done in {dt:.1f}s")

            # Find actual links shapefile (RivGraph names it <name>_links.shp)
            found_links = list(year_dir.glob("*links*.shp"))
            if found_links:
                # Rename to our standard name if different
                actual = found_links[0]
                if actual != links_shp:
                    actual.rename(links_shp)
                    # Also rename companion files
                    for ext in [".shx", ".dbf", ".prj", ".cpg"]:
                        companion = actual.with_suffix(ext)
                        if companion.exists():
                            companion.rename(links_shp.with_suffix(ext))
                print(f"[{year}] Links: {links_shp}")
            else:
                print(f"[WARN] {year}: no links shapefile found in {year_dir}")
                return None
        except Exception as exc:
            print(f"[ERROR] {year}: RivGraph failed: {exc}")
            import traceback
            traceback.print_exc()
            return None

    # Step 2: Profile extraction (s, B, C)
    if not links_shp.exists():
        # Try to find any links shp in the directory
        found_links = list(year_dir.glob("*links*.shp"))
        if found_links:
            links_shp = found_links[0]
        else:
            print(f"[SKIP] {year}: no links shapefile for profile extraction")
            return None

    if skip_if_exists and csv_path.exists():
        print(f"[SKIP] {year}: profile CSV already exists")
        return csv_path

    print(f"[{year}] Computing B(s), C(s) profiles ...")
    t0 = time.time()
    try:
        from src.morphodynamics.rivgraph_link_profiles import compute_link_profiles

        profiles = compute_link_profiles(
            mask_raster_path=str(mask_path),
            links_vector_path=str(links_shp),
            step_m=step_m,
            normal_search_halfwidth_m=normal_search_halfwidth_m,
        )

        if not profiles:
            print(f"[WARN] {year}: no profiles computed")
            return None

        # Export CSV
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        _export_profiles_csv(profiles, year, csv_path)
        dt = time.time() - t0
        print(f"[{year}] Profiles done in {dt:.1f}s -> {csv_path}")
        return csv_path
    except Exception as exc:
        print(f"[ERROR] {year}: profile extraction failed: {exc}")
        import traceback
        traceback.print_exc()
        return None


def _export_profiles_csv(
    profiles: dict,
    year: int,
    out_path: Path,
) -> None:
    """Export link profiles to flat CSV."""
    import csv

    fieldnames = ["year", "link_id", "sample_idx", "s_m", "lon", "lat", "B_m", "C_1m"]
    link_ids = sorted(profiles.keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for link_id in link_ids:
            prof = profiles[link_id]
            s = np.asarray(prof["s"])
            x = np.asarray(prof["x"])
            y = np.asarray(prof["y"])
            B = np.asarray(prof["B"])
            C = np.asarray(prof["C"])
            for i in range(s.size):
                writer.writerow({
                    "year": year,
                    "link_id": link_id,
                    "sample_idx": i,
                    "s_m": f"{s[i]:.2f}",
                    "lon": f"{x[i]:.6f}",
                    "lat": f"{y[i]:.6f}",
                    "B_m": f"{B[i]:.2f}" if np.isfinite(B[i]) else "",
                    "C_1m": f"{C[i]:.8f}" if np.isfinite(C[i]) else "",
                })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch RivGraph + B(s)/C(s) profile extraction for Gaocun-Sunkou.",
    )
    parser.add_argument(
        "--years", nargs="*", type=int, default=None,
        help="Year(s) to process (default: all 2000-2023)",
    )
    parser.add_argument(
        "--exit-sides", default="WE",
        help="RivGraph exit_sides: upstream side first (default: WE)",
    )
    parser.add_argument(
        "--step-m", type=float, default=100.0,
        help="Profile sampling interval in meters (default: 100)",
    )
    parser.add_argument(
        "--search-halfwidth", type=float, default=None,
        help="Normal search halfwidth in meters (default: auto ~30 pixels)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if outputs exist",
    )
    args = parser.parse_args()

    if args.years is None:
        years = list(range(2000, 2024))
    else:
        years = args.years

    print(f"Processing {len(years)} year(s): {years[0]}-{years[-1]}")
    print(f"Exit sides: {args.exit_sides}")
    print(f"Step: {args.step_m}m")
    print(f"Force: {args.force}")
    print()

    results = {}
    for year in years:
        csv_path = run_rivgraph_for_year(
            year=year,
            exit_sides=args.exit_sides,
            step_m=args.step_m,
            normal_search_halfwidth_m=args.search_halfwidth,
            skip_if_exists=not args.force,
        )
        results[year] = csv_path
        print()

    # Summary
    ok = sum(1 for v in results.values() if v is not None)
    print("=" * 60)
    print(f"Done: {ok}/{len(results)} years processed successfully.")
    failed = [y for y, v in results.items() if v is None]
    if failed:
        print(f"Failed/skipped: {failed}")


if __name__ == "__main__":
    main()