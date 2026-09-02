""" JRC Global Surface Water occurrence — tight boundary。

 JRC/GSW1_4/GlobalSurfaceWater occurrence  (1984-2021, 0-100%),
 bbox ,  buffer  GeoJSON,
 DSWE  tight ROI。

:
    - Google Drive GeoJSON: GaocunSunkou_boundary.geojson
    - Google Drive Shapefile: GaocunSunkou_boundary.shp
    - Google Drive GeoTIFF: GaocunSunkou_JRC_occurrence.tif ()

:
    Step 1: python -m src.gee_data.export_jrc_boundary
    Step 2:  Google Drive  data/GIS/Gaocun-Sunkou/
    Step 3: python -m src.gee_data.filter_boundary  ()
    Step 4: python -m src.gee_data.export_gaocun_sunkou_masks ( filtered boundary)

:
    - earthengine-api

 GEE:
    earthengine authenticate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ee


# =====================================================================================
# === GEE  ======================================================================
# =====================================================================================

GEE_PROJECT = "hip-apricot-453400-m1"


def initialize_gee() -> bool:
    """ Google Earth Engine。"""
    try:
        ee.Initialize(project=GEE_PROJECT)
        print("Google Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Google Earth Engine: {e}")
        print("Please run 'earthengine authenticate' first.")
        return False


# =====================================================================================
# ===  bbox  ==================================================================
# =====================================================================================

# — ROI（WGS84 ）
# ： PyGEE-SWToolbox  TIF  [115.076, 35.395, 115.914, 35.935]
# （）:  (115.0759, 35.3641),  (115.9052, 35.9340)
# 35.34 (35.3641N)
_ROUGH_BBOX_COORDS = [115.07, 35.34, 115.92, 35.94]


# =====================================================================================
# === JRC occurrence boundary  ====================================================
# =====================================================================================

def build_occurrence_boundary(
    occurrence_threshold: int = 5,
    buffer_m: int = 1000,
) -> tuple[ee.FeatureCollection, ee.Image]:
    """ JRC occurrence  tight boundary 。

    
    ------
    occurrence_threshold : int
        occurrence （%），。 5%。
    buffer_m : int
        （）。 1000m。

    
    ------
    boundary_fc : ee.FeatureCollection
        （ Feature）。
    occurrence : ee.Image
         JRC occurrence （0-100），。
    """
    rough_bbox = ee.Geometry.Rectangle(_ROUGH_BBOX_COORDS)

# JRC Global Surface Water occurrence
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    occurrence = gsw.select("occurrence").clip(rough_bbox)
# ：occurrence >= threshold →
    water_ever = occurrence.gte(occurrence_threshold).selfMask()

# ： 10 （~9000 m²）
# connectedPixelCount ，
    connected = water_ever.connectedPixelCount(maxSize=50)
    water_clean = water_ever.updateMask(connected.gte(10))

# 
    vectors = water_clean.reduceToVectors(
        geometry=rough_bbox,
        scale=30,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="water",
        maxPixels=1e8,
        bestEffort=True,
    )

# geometry， buffer
    merged = vectors.union(maxError=30)
    buffered = merged.geometry().buffer(buffer_m, maxError=30)

# FeatureCollection
    boundary_fc = ee.FeatureCollection([ee.Feature(buffered, {
        "site": "Gaocun-Sunkou",
        "source": "JRC/GSW1_4/GlobalSurfaceWater",
        "occurrence_threshold_pct": occurrence_threshold,
        "buffer_m": buffer_m,
        "description": "Tight boundary from JRC water occurrence (1984-2021)",
    })])

    return boundary_fc, occurrence


# =====================================================================================
# ===  ========================================================================
# =====================================================================================

def export_boundary_and_occurrence(
    occurrence_threshold: int = 5,
    buffer_m: int = 1000,
    drive_folder: str = "GaocunSunkou_WaterMasks",
) -> list:
    """ boundary GeoJSON  occurrence GeoTIFF  Google Drive。

    
    ------
    occurrence_threshold : int
        occurrence （%）。
    buffer_m : int
        （）。
    drive_folder : str
        Google Drive 。

    
    ------
    tasks : list[tuple[str, ee.batch.Task]]
        。
    """
    print(f"\n{'=' * 60}")
    print("Export JRC occurrence boundary for Gaocun-Sunkou")
    print(f"  Occurrence threshold: >= {occurrence_threshold}%")
    print(f"  Buffer: {buffer_m} m")
    print(f"  Drive folder: {drive_folder}")
    print(f"{'=' * 60}\n")

    boundary_fc, occurrence = build_occurrence_boundary(
        occurrence_threshold=occurrence_threshold,
        buffer_m=buffer_m,
    )

    tasks = []

# ---  1: boundary GeoJSON ---
    task_boundary = ee.batch.Export.table.toDrive(
        collection=boundary_fc,
        description="GaocunSunkou_JRC_boundary",
        folder=drive_folder,
        fileNamePrefix="GaocunSunkou_boundary",
        fileFormat="GeoJSON",
    )
    task_boundary.start()
    tasks.append(("boundary_geojson", task_boundary))
    print("Submitted: GaocunSunkou_boundary.geojson")

    time.sleep(1)

    rough_bbox = ee.Geometry.Rectangle(_ROUGH_BBOX_COORDS)

# ---  2: occurrence GeoTIFF（）---
    task_occurrence = ee.batch.Export.image.toDrive(
        image=occurrence.toFloat(),
        description="GaocunSunkou_JRC_occurrence",
        folder=drive_folder,
        fileNamePrefix="GaocunSunkou_JRC_occurrence",
        scale=30,
        region=rough_bbox,
        maxPixels=1e9,
        crs="EPSG:4326",
    )
    task_occurrence.start()
    tasks.append(("occurrence_tif", task_occurrence))
    print("Submitted: GaocunSunkou_JRC_occurrence.tif")

    time.sleep(1)

# ---  3: boundary  SHP（ GIS ）---
    task_shp = ee.batch.Export.table.toDrive(
        collection=boundary_fc,
        description="GaocunSunkou_JRC_boundary_shp",
        folder=drive_folder,
        fileNamePrefix="GaocunSunkou_boundary",
        fileFormat="SHP",
    )
    task_shp.start()
    tasks.append(("boundary_shp", task_shp))
    print("Submitted: GaocunSunkou_boundary.shp")

    print(f"\n{'=' * 60}")
    print(f"Submitted {len(tasks)} export tasks to Google Drive")
    print(f"Check status: https://code.earthengine.google.com/tasks")
    print(f"Download from Drive folder: '{drive_folder}'")
    print(f"{'=' * 60}")

    print("\n--- Next steps after download ---")
    print("1. Download GaocunSunkou_boundary.shp from Google Drive")
    print("2. Place in: data/GIS/Gaocun-Sunkou/Gaocun-Sunkou.shp")
    print("   (include .shx, .dbf, .prj sidecar files)")
    print("3. Run DSWE export: python -m src.gee_data.export_gaocun_sunkou_masks")
    print("   (will auto-detect Shapefile and use tight ROI)")

    return tasks


# =====================================================================================
# === （， boundary ）=========================================
# =====================================================================================

def print_area_stats(
    occurrence_threshold: int = 5,
    buffer_m: int = 1000,
):
    """ boundary ， bbox 。"""
    boundary_fc, _ = build_occurrence_boundary(
        occurrence_threshold=occurrence_threshold,
        buffer_m=buffer_m,
    )

    rough_bbox = ee.Geometry.Rectangle(_ROUGH_BBOX_COORDS)
    boundary_area = boundary_fc.geometry().area(maxError=100).getInfo()
    bbox_area = rough_bbox.area(maxError=100).getInfo()
    boundary_km2 = boundary_area / 1e6
    bbox_km2 = bbox_area / 1e6
    ratio = boundary_km2 / bbox_km2 * 100

    print(f"\nArea statistics:")
    print(f"  Rough bbox:       {bbox_km2:8.1f} km^2")
    print(f"  Tight boundary:   {boundary_km2:8.1f} km^2")
    print(f"  Ratio:            {ratio:8.1f}%")
    print(f"  Reduction:        {100 - ratio:8.1f}%")


# =====================================================================================
# === CLI  ========================================================================
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export JRC occurrence-based tight boundary for Gaocun-Sunkou",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: occurrence >= 5%, buffer 1000m
    python -m src.gee_data.export_jrc_boundary

    # More conservative: only where water appeared > 10% of time
    python -m src.gee_data.export_jrc_boundary --occurrence-threshold 10

    # Tighter buffer
    python -m src.gee_data.export_jrc_boundary --buffer-m 500

    # Just check area stats (no export)
    python -m src.gee_data.export_jrc_boundary --stats-only
        """,
    )
    parser.add_argument(
        "--occurrence-threshold", type=int, default=5,
        help="Minimum water occurrence (%%) to include (default: 5)",
    )
    parser.add_argument(
        "--buffer-m", type=int, default=1000,
        help="Buffer distance in meters (default: 1000)",
    )
    parser.add_argument(
        "--drive-folder", default="GaocunSunkou_WaterMasks",
        help="Google Drive target folder name",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only print area statistics, do not export",
    )

    args = parser.parse_args()

    if not initialize_gee():
        return

    if args.stats_only:
        print_area_stats(
            occurrence_threshold=args.occurrence_threshold,
            buffer_m=args.buffer_m,
        )
    else:
        export_boundary_and_occurrence(
            occurrence_threshold=args.occurrence_threshold,
            buffer_m=args.buffer_m,
            drive_folder=args.drive_folder,
        )


if __name__ == "__main__":
    main()