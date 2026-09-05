"""Export tight study reach boundary from JRC Global Surface Water occurrence.

Uses JRC/GSW1_4/GlobalSurfaceWater occurrence band (1984-2021, 0-100%) to extract
historical water presence within the rough bounding box, buffers the polygon,
and exports GeoJSON and Shapefile to Google Drive as the tight ROI for DSWE annual masks.

Workflow:
    Step 1: python -m src.gee_data.export_jrc_boundary
    Step 2: Download from Google Drive to data/GIS/Gaocun-Sunkou/
    Step 3: python -m src.gee_data.filter_boundary (remove small detached fragments)
    Step 4: python -m src.gee_data.export_gaocun_sunkou_masks (using filtered boundary)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ee


# =====================================================================================
# === GEE 初始化 ======================================================================
# =====================================================================================

GEE_PROJECT = "hip-apricot-453400-m1"


def initialize_gee() -> bool:
    """初始化 Google Earth Engine。"""
    try:
        ee.Initialize(project=GEE_PROJECT)
        print("Google Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Google Earth Engine: {e}")
        print("Please run 'earthengine authenticate' first.")
        return False


# =====================================================================================
# === 粗框 bbox 定义 ==================================================================
# =====================================================================================

# 高村—孙口河段粗框 ROI（WGS84 经纬度）
# 来源：用户已有研究 PyGEE-SWToolbox 导出 TIF 边界 [115.076, 35.395, 115.914, 35.935]
# 站点坐标（十进制）: 高村 (115.0759, 35.3641), 孙口 (115.9052, 35.9340)
# 南边界取到 35.34 以将高村站(35.3641N)包括在内并预留缓冲
_ROUGH_BBOX_COORDS = [115.07, 35.34, 115.92, 35.94]


# =====================================================================================
# === JRC occurrence boundary 提取 ====================================================
# =====================================================================================

def build_occurrence_boundary(
    occurrence_threshold: int = 5,
    buffer_m: int = 1000,
) -> tuple[ee.FeatureCollection, ee.Image]:
    """从 JRC occurrence 构建 tight boundary 多边形。

    参数
    ------
    occurrence_threshold : int
        occurrence 最低阈值（%），低于此值视为噪声。默认 5%。
    buffer_m : int
        向外缓冲距离（米）。默认 1000m。

    返回
    ------
    boundary_fc : ee.FeatureCollection
        缓冲后的边界多边形（单个 Feature）。
    occurrence : ee.Image
        原始 JRC occurrence 影像（0-100），用于可视化导出。
    """
    rough_bbox = ee.Geometry.Rectangle(_ROUGH_BBOX_COORDS)

    # 加载 JRC Global Surface Water occurrence 波段
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    occurrence = gsw.select("occurrence").clip(rough_bbox)
    # 阈值化：occurrence >= threshold → 二值掩膜
    water_ever = occurrence.gte(occurrence_threshold).selfMask()

    # 连通域降噪：移除小于 10 像元（~9000 m²）的孤立斑块
    # connectedPixelCount 计算连通像元数，过滤小碎片
    connected = water_ever.connectedPixelCount(maxSize=50)
    water_clean = water_ever.updateMask(connected.gte(10))

    # 矢量化
    vectors = water_clean.reduceToVectors(
        geometry=rough_bbox,
        scale=30,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="water",
        maxPixels=1e8,
        bestEffort=True,
    )

    # 合并所有多边形为单个 geometry，然后 buffer
    merged = vectors.union(maxError=30)
    buffered = merged.geometry().buffer(buffer_m, maxError=30)

    # 封装为 FeatureCollection
    boundary_fc = ee.FeatureCollection([ee.Feature(buffered, {
        "site": "Gaocun-Sunkou",
        "source": "JRC/GSW1_4/GlobalSurfaceWater",
        "occurrence_threshold_pct": occurrence_threshold,
        "buffer_m": buffer_m,
        "description": "Tight boundary from JRC water occurrence (1984-2021)",
    })])

    return boundary_fc, occurrence


# =====================================================================================
# === 导出函数 ========================================================================
# =====================================================================================

def export_boundary_and_occurrence(
    occurrence_threshold: int = 5,
    buffer_m: int = 1000,
    drive_folder: str = "GaocunSunkou_WaterMasks",
) -> list:
    """导出 boundary GeoJSON 和 occurrence GeoTIFF 到 Google Drive。

    参数
    ------
    occurrence_threshold : int
        occurrence 最低阈值（%）。
    buffer_m : int
        向外缓冲距离（米）。
    drive_folder : str
        Google Drive 目标文件夹名。

    返回
    ------
    tasks : list[tuple[str, ee.batch.Task]]
        已提交的导出任务列表。
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

    # --- 导出 1: boundary GeoJSON ---
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

    # --- 导出 2: occurrence GeoTIFF（可视化参考）---
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

    # --- 导出 3: boundary 也导出为 SHP（本地 GIS 使用）---
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
# === 面积统计（可选，快速检查 boundary 合理性）=========================================
# =====================================================================================

def print_area_stats(
    occurrence_threshold: int = 5,
    buffer_m: int = 1000,
):
    """打印 boundary 面积统计，与原始 bbox 面积对比。"""
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
# === CLI 入口 ========================================================================
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
