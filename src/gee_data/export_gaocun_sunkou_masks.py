"""
— Google Drive

 Jones DSWE  Landsat Collection 2 ， Google Drive。
 Google Drive  data/GEOTIFFS/Gaocun-Sunkou/ 。

 C&G  export_huanghe_masks_to_drive.py，
 Subproject-3 —：
  -  Gaocun-Sunkou
  -  2000-2023（）
  - ROI  Shapefile / GeoJSON / 
  - Drive  GaocunSunkou_WaterMasks

ROI （ filtered ）：
  1. data/GIS/Gaocun-Sunkou/GaocunSunkou_boundary_filtered.shp
  2. data/GIS/Gaocun-Sunkou/GaocunSunkou_boundary.shp
  3. data/GIS/Gaocun-Sunkou/Gaocun-Sunkou.shp
  4. data/GIS/Gaocun-Sunkou.shp
  5. data/GIS/Gaocun-Sunkou/GaocunSunkou_boundary_filtered.geojson
  6. data/GIS/Gaocun-Sunkou/GaocunSunkou_boundary.geojson
  7.  (115.55-115.95E, 35.70-36.12N)

：
  1.  export_jrc_boundary.py  JRC occurrence  Google Drive
  2.  Shapefile ( shx, dbf )  data/GIS/Gaocun-Sunkou/ 
  3.  DSWE （ tight ROI）

：
    python -m src.gee_data.export_gaocun_sunkou_masks --start-year 2000 --end-year 2023

：
    - earthengine-api
    - fiona ( Shapefile)  json ( GeoJSON)

 GEE：
    earthengine authenticate
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ee


# =====================================================================================
# === GEE  ======================================================================
# =====================================================================================

def initialize_gee():
    """ Google Earth Engine。"""
    try:
        ee.Initialize(project="hip-apricot-453400-m1")
        print("Google Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Google Earth Engine: {e}")
        print("Please run 'earthengine authenticate' first.")
        return False


# =====================================================================================
# ===  ==================================================================
# =====================================================================================

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

# — ROI （WGS84 ）
# ： PyGEE-SWToolbox  TIF  [115.076, 35.395, 115.914, 35.935]
# （）:  (115.0759, 35.3641),  (115.9052, 35.9340)
# 35.34 (35.3641N)
_GAOCUN_SUNKOU_FALLBACK_BBOX = [
    [115.07, 35.34],
    [115.92, 35.34],
    [115.92, 35.94],
    [115.07, 35.94],
    [115.07, 35.34],
]


def get_roi(site: str = "Gaocun-Sunkou") -> ee.Geometry:
    """ Shapefile  GeoJSON  EE Geometry。

    （ filtered ）：
        1. data/GIS/{site}/GaocunSunkou_boundary_filtered.shp
        2. data/GIS/{site}/GaocunSunkou_boundary.shp
        3. data/GIS/{site}/{site}.shp
        4. data/GIS/{site}.shp
        5. data/GIS/{site}/GaocunSunkou_boundary_filtered.geojson
        6. data/GIS/{site}/GaocunSunkou_boundary.geojson
        7. 
    """
    gis_dir = DATA_ROOT / "GIS"

# ---  Shapefile ---
    site_compact = site.replace("-", "")
    shp_candidates = [
        gis_dir / site / f"{site_compact}_boundary_filtered.shp",
        gis_dir / site / f"{site_compact}_boundary.shp",
        gis_dir / site / f"{site}.shp",
        gis_dir / f"{site}.shp",
    ]
    for shp_path in shp_candidates:
        if shp_path.exists():
            return _load_shp_roi(shp_path)

# ---  GeoJSON ---
    geojson_candidates = [
        gis_dir / site / f"{site_compact}_boundary_filtered.geojson",
        gis_dir / site / f"{site_compact}_boundary.geojson",
        gis_dir / site / f"{site}.geojson",
        gis_dir / f"{site}.geojson",
    ]
    for geojson_path in geojson_candidates:
        if geojson_path.exists():
            return _load_geojson_roi(geojson_path)

# ---  ---
    print(f"WARNING: No boundary file found in {gis_dir}")
    print(f"  Searched: {', '.join(str(p.name) for p in shp_candidates + geojson_candidates)}")
    print("  Using fallback bounding box (~1645 km^2, mostly land).")
    print("  TIP: Run 'python -m src.gee_data.export_jrc_boundary' first to generate tight ROI.")
    return ee.Geometry.Polygon([_GAOCUN_SUNKOU_FALLBACK_BBOX])


def _load_shp_roi(shp_path: Path) -> ee.Geometry:
    """ Shapefile  ROI 。"""
    try:
        import fiona
    except ImportError:
        print("Warning: fiona not available, falling back to hardcoded bbox.")
        return ee.Geometry.Polygon([_GAOCUN_SUNKOU_FALLBACK_BBOX])

    print(f"Using Shapefile ROI: {shp_path}")

    with fiona.open(shp_path) as src:
        if len(src) == 0:
            raise ValueError(f"Shapefile has no features: {shp_path}")

        feat = next(iter(src))
        geom = feat["geometry"]
        if geom is None:
            raise ValueError(f"Feature geometry is null: {shp_path}")

        return _geom_dict_to_ee(geom, str(shp_path))


def _load_geojson_roi(geojson_path: Path) -> ee.Geometry:
    """ GeoJSON  ROI 。"""
    import json

    print(f"Using GeoJSON ROI: {geojson_path}")

    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)

# GeoJSON  FeatureCollection  Feature  Geometry
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError(f"GeoJSON has no features: {geojson_path}")
        geom = features[0].get("geometry")
    elif data.get("type") == "Feature":
        geom = data.get("geometry")
    else:
        geom = data  # Geometry

    if geom is None:
        raise ValueError(f"GeoJSON geometry is null: {geojson_path}")

    return _geom_dict_to_ee(geom, str(geojson_path))


def _geom_dict_to_ee(geom: dict, source: str) -> ee.Geometry:
    """ GeoJSON-style geometry dict  ee.Geometry.Polygon。"""
    gtype = geom.get("type")
    coords = geom.get("coordinates")

    if gtype == "Polygon":
        outer = coords[0]
    elif gtype == "MultiPolygon":
# （）
        outer = max(coords, key=lambda poly: len(poly[0]))[0]
    else:
        raise ValueError(f"Unsupported geometry type '{gtype}' in {source}")

    coords_2d = [[x, y] for (x, y, *_) in outer]
    return ee.Geometry.Polygon([coords_2d])


# =====================================================================================
# === Jones DSWE  =========================================================
# =====================================================================================

def compute_dswe_mask(image, water_level: int = 2):
    """
     Jones et al. (2019) DSWE 。

     GEE_watermasks (evan-greenbrg) 。
    ：https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/
          s3fs-public/media/files/LSDS-2084_LandsatC2_L3_DSWE_ADD-v1.pdf

    water_level:
        1 = High confidence water
        2 = Moderate confidence water
        3 = Potential wetland
        4 = Low confidence water / partial surface water
    """
    image = ee.Image(image)

    blue = image.select("SR_B2")
    green = image.select("SR_B3")
    red = image.select("SR_B4")
    nir = image.select("SR_B5")
    swir1 = image.select("SR_B6")
    swir2 = image.select("SR_B7")

    # Water indices
    mndwi = green.subtract(swir1).divide(green.add(swir1))
    mbsrv = green.add(red)
    mbsrn = nir.add(swir1)
    ndvi = nir.subtract(red).divide(nir.add(red))
    awesh = (blue.add(green.multiply(2.5))
             .subtract(mbsrn.multiply(1.5))
             .subtract(swir2.multiply(0.25)))

    # DSWE 5 tests (Jones et al. 2019)
    t1 = mndwi.gt(0.124).toInt()
    t2 = mbsrv.gt(mbsrn).toInt()
    t3 = awesh.gt(0).toInt()
    t4 = (mndwi.gt(-0.44).And(swir1.lt(0.09))
          .And(nir.lt(0.15)).And(ndvi.lt(0.7)).toInt())
    t5 = (mndwi.gt(-0.5).And(blue.lt(0.1)).And(swir1.lt(0.3))
          .And(swir2.lt(0.1)).And(nir.lt(0.25)).toInt())

    # 5-digit decimal encoding
    t_code = (t1.add(t2.multiply(10)).add(t3.multiply(100))
              .add(t4.multiply(1000)).add(t5.multiply(10000)))

    dswe = ee.Image(0).toInt()

    # High confidence water (class 1)
    high_conf = (t_code.eq(1111).Or(t_code.eq(10111))
                 .Or(t_code.eq(11101)).Or(t_code.eq(11110))
                 .Or(t_code.eq(11111)))
    dswe = dswe.where(high_conf, 1)

    # Moderate confidence water (class 2)
    mod_conf = (
        t_code.eq(111).Or(t_code.eq(1011)).Or(t_code.eq(1101)).Or(t_code.eq(1110))
        .Or(t_code.eq(10011)).Or(t_code.eq(10101)).Or(t_code.eq(10110))
        .Or(t_code.eq(11001)).Or(t_code.eq(11010)).Or(t_code.eq(11100))
    )
    dswe = dswe.where(mod_conf, 2)

    # Potential wetland (class 3)
    dswe = dswe.where(t_code.eq(11000), 3)

    # Low confidence water (class 4)
    low_conf = (
        t_code.eq(11).Or(t_code.eq(101)).Or(t_code.eq(110))
        .Or(t_code.eq(1001)).Or(t_code.eq(1010)).Or(t_code.eq(1100))
        .Or(t_code.eq(10000)).Or(t_code.eq(10001))
        .Or(t_code.eq(10010)).Or(t_code.eq(10100))
    )
    dswe = dswe.where(low_conf, 4)

    water_mask = dswe.gte(1).And(dswe.lte(water_level))
    return water_mask.rename("water_mask")


def get_landsat_collection(year: int, roi: ee.Geometry,
                           start_date: str, end_date: str):
    """ Landsat Collection 2 Level-2 。"""

    if year <= 1984:
        collection_id = "LANDSAT/LT04/C02/T1_L2"
        sensor = "L4"
    elif year <= 2012:
        if year <= 1999:
            collection_id = "LANDSAT/LT05/C02/T1_L2"
            sensor = "L5"
        else:
            collection_id = "LANDSAT/LE07/C02/T1_L2"
            sensor = "L7"
    elif year <= 2021:
        collection_id = "LANDSAT/LC08/C02/T1_L2"
        sensor = "L8"
    else:
        collection_id = "LANDSAT/LC09/C02/T1_L2"
        sensor = "L9"

    if sensor in ["L4", "L5", "L7"]:
        def rename_bands(img):
            img = ee.Image(img)
            spec = img.select(
                ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
            )
            band_names = img.bandNames()
            qa_exists = band_names.contains("QA_PIXEL")

            def _with_qa():
                return spec.addBands(img.select("QA_PIXEL"))

            def _no_qa():
                return spec

            out = ee.Image(ee.Algorithms.If(qa_exists, _with_qa(), _no_qa()))
            return out.copyProperties(img, ["system:time_start"])

        collection = (ee.ImageCollection(collection_id)
                      .filterBounds(roi)
                      .filterDate(start_date, end_date)
                      .map(rename_bands))
    else:
        collection = (ee.ImageCollection(collection_id)
                      .filterBounds(roi)
                      .filterDate(start_date, end_date))

    return collection, sensor


def scale_landsat(image):
    """ Landsat C2 L2 。"""
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return optical.copyProperties(image, ["system:time_start"])


def mask_clouds(image):
    """ QA_PIXEL ； QA_PIXEL 。"""
    image = ee.Image(image)
    band_names = image.bandNames()
    qa_exists = band_names.contains("QA_PIXEL")

    def _apply_mask(img):
        qa = img.select("QA_PIXEL")
        cloud_mask = (qa.bitwiseAnd(1 << 3).eq(0)
                      .And(qa.bitwiseAnd(1 << 4).eq(0)))
        return img.updateMask(cloud_mask)

    def _no_mask(img):
        return img

    return ee.Image(
        ee.Algorithms.If(qa_exists, _apply_mask(image), _no_mask(image))
    )


# =====================================================================================
# ===  ======================================================================
# =====================================================================================

def export_annual_masks(
    site: str = "Gaocun-Sunkou",
    start_year: int = 2000,
    end_year: int = 2023,
    water_level: int = 2,
    drive_folder: str = "GaocunSunkou_WaterMasks",
):
    """
     Google Drive。

    
    ------
    site : str
        ， "Gaocun-Sunkou"。
    start_year : int
        ， 2000（）。
    end_year : int
        ， 2023。
    water_level : int
        DSWE  (1-4)。
    drive_folder : str
        Google Drive 。
    """
    roi = get_roi(site)

    print(f"\n{'=' * 60}")
    print(f"Export {site} annual water masks to Google Drive")
    print(f"Time range: {start_year} - {end_year}")
    print(f"Water Level: {water_level}")
    print(f"Drive folder: {drive_folder}")
    print(f"{'=' * 60}\n")

    tasks = []

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        print(f"Processing {year}...")

        try:
            collection, sensor = get_landsat_collection(
                year, roi, start_date, end_date
            )

            count = collection.size().getInfo()
            if count == 0:
                print(f"  WARNING: No images available for {year}, skipping.")
                continue

            print(f"  Found {count} {sensor} images")

            collection = collection.map(mask_clouds).map(scale_landsat)
            composite = collection.median()
            water_mask = compute_dswe_mask(composite, water_level=water_level)
            water_mask = water_mask.clip(roi).toUint8()

            task_name = f"{site}_{year}_01-01_12-31_mask{water_level}"

            task = ee.batch.Export.image.toDrive(
                image=water_mask,
                description=task_name,
                folder=drive_folder,
                fileNamePrefix=task_name,
                scale=30,
                region=roi,
                maxPixels=1e13,
                crs="EPSG:4326",
            )
            task.start()
            tasks.append((year, task_name, task))

            print(f"  Submitted export task: {task_name}")

            # Avoid API rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"  ERROR processing {year}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(f"Submitted {len(tasks)} export tasks to Google Drive")
    print(f"Check status: https://code.earthengine.google.com/tasks")
    print(f"Download from Drive folder: '{drive_folder}'")
    print(f"{'=' * 60}")

    return tasks


# =====================================================================================
# === CLI  ========================================================================
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export Gaocun-Sunkou annual water masks to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export default 2000-2023 at water_level=2
    python -m src.gee_data.export_gaocun_sunkou_masks

    # Export only 2010-2020
    python -m src.gee_data.export_gaocun_sunkou_masks --start-year 2010 --end-year 2020

    # Export water_level=1 (high confidence only)
    python -m src.gee_data.export_gaocun_sunkou_masks --water-level 1
        """,
    )
    parser.add_argument("--site", default="Gaocun-Sunkou",
                        help="Site name (default: Gaocun-Sunkou)")
    parser.add_argument("--start-year", type=int, default=2000,
                        help="Start year (default: 2000)")
    parser.add_argument("--end-year", type=int, default=2023,
                        help="End year (default: 2023)")
    parser.add_argument("--water-level", type=int, default=2,
                        choices=[1, 2, 3, 4],
                        help="DSWE confidence level (1=high, 2=moderate, 3=wetland, 4=low)")
    parser.add_argument("--drive-folder", default="GaocunSunkou_WaterMasks",
                        help="Google Drive target folder name")

    args = parser.parse_args()

    if not initialize_gee():
        return

    export_annual_masks(
        site=args.site,
        start_year=args.start_year,
        end_year=args.end_year,
        water_level=args.water_level,
        drive_folder=args.drive_folder,
    )


if __name__ == "__main__":
    main()