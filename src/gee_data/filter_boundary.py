"""Filter boundary MultiPolygon: keep only sub-polygons > min_area_km2."""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from pathlib import Path

import fiona
from shapely.geometry import shape, mapping, MultiPolygon
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

BASE = Path(__file__).resolve().parent.parent.parent / 'data' / 'GIS' / 'Gaocun-Sunkou'
MIN_AREA_KM2 = 5.0


def filter_boundary():
    geojson_path = BASE / 'GaocunSunkou_boundary.geojson'
    shp_path = BASE / 'GaocunSunkou_boundary.shp'

    # Prefer GeoJSON over Shapefile (Shapefile may lack .dbf from GEE export)
    if geojson_path.exists():
        with open(geojson_path, 'r', encoding='utf-8') as f:
            gj = json.load(f)
        feat = gj['features'][0]
        geom = shape(feat['geometry'])
        props = feat.get('properties', {})
    elif shp_path.exists():
        with fiona.open(shp_path) as src:
            feat = next(iter(src))
        geom = shape(feat['geometry'])
        props = feat['properties']
    else:
        raise FileNotFoundError(f'No boundary file found in {BASE}')

    print(f'Original: {geom.geom_type}, {len(list(geom.geoms))} sub-polygons')

    # Transform to UTM for area calc
    to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32650', always_xy=True)

    kept = []
    dropped = []
    for i, poly in enumerate(geom.geoms):
        poly_utm = shapely_transform(to_utm.transform, poly)
        area_km2 = poly_utm.area / 1e6
        if area_km2 >= MIN_AREA_KM2:
            kept.append((i, area_km2, poly))
        else:
            dropped.append((i, area_km2))

    print(f'\nKept {len(kept)} polygons (>= {MIN_AREA_KM2} km2):')
    total_kept = 0
    for idx, area, _ in kept:
        print(f'  #{idx}: {area:.2f} km2')
        total_kept += area

    print(f'\nDropped {len(dropped)} polygons (< {MIN_AREA_KM2} km2):')
    total_dropped = sum(a for _, a in dropped)
    print(f'  Total dropped area: {total_dropped:.2f} km2')

    print(f'\nFiltered boundary area: {total_kept:.2f} km2')

    # Build filtered geometry
    kept_polys = [poly for _, _, poly in kept]
    if len(kept_polys) == 1:
        filtered_geom = kept_polys[0]
    else:
        filtered_geom = MultiPolygon(kept_polys)

    # Save as GeoJSON
    filtered_feat = {
        'type': 'Feature',
        'geometry': mapping(filtered_geom),
        'properties': {
            **props,
            'filter_min_area_km2': MIN_AREA_KM2,
            'original_polygon_count': len(list(geom.geoms)),
            'filtered_polygon_count': len(kept_polys),
        }
    }
    geojson = {
        'type': 'FeatureCollection',
        'features': [filtered_feat]
    }

    out_geojson = BASE / 'GaocunSunkou_boundary_filtered.geojson'
    with open(out_geojson, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)
    print(f'\nSaved: {out_geojson}')

    # Also save as Shapefile via fiona
    schema = {
        'geometry': filtered_geom.geom_type,
        'properties': {k: type(v).__name__ for k, v in filtered_feat['properties'].items()}
    }
    # Fix types for fiona
    fiona_props = {}
    for k, v in filtered_feat['properties'].items():
        if isinstance(v, int):
            fiona_props[k] = 'int'
        elif isinstance(v, float):
            fiona_props[k] = 'float'
        else:
            fiona_props[k] = 'str'
    schema['properties'] = fiona_props

    out_shp = BASE / 'GaocunSunkou_boundary_filtered.shp'
    with fiona.open(out_shp, 'w', driver='ESRI Shapefile',
                    crs='EPSG:4326', schema=schema) as dst:
        dst.write({
            'geometry': mapping(filtered_geom),
            'properties': filtered_feat['properties'],
        })
    print(f'Saved: {out_shp}')

    return filtered_geom, total_kept


if __name__ == '__main__':
    filter_boundary()
