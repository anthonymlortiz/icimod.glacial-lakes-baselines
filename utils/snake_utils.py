import os
import fiona
from shapely.geometry import Polygon
from functools import partial
from shapely.ops import transform

import pyproj
import warnings

import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from utils import data



warnings.simplefilter(action='ignore', category=FutureWarning)


def buffer_polygon_in_meters(polygon, buffer, percentage=0.8):
    proj_meters = pyproj.Proj('epsg:3857')
    proj_latlng = pyproj.Proj('epsg:4326')

    project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
    project_to_latlng = partial(pyproj.transform, proj_meters, proj_latlng)

    pt_meters = transform(project_to_meters, polygon)
    buffer_meters = pt_meters

    while buffer_meters.area > percentage * pt_meters.area:
        buffer_meters = buffer_meters.buffer(buffer)
    buffer_polygon = transform(project_to_latlng, buffer_meters.buffer(0))

    return buffer_polygon


def get_bing_glid_from_fn(bing_fn):
    base = os.path.basename(bing_fn)
    gl_id = os.path.splitext(base)[0]
    return gl_id


def get_sentinel_glid_from_fn(bing_fn):
    base = os.path.basename(bing_fn)
    gl_id, _ = base.split("_")
    return gl_id


def get_glacial_lake_2015_outline_from_glid(lakes_shapefile_2015, glid):
    with fiona.open(lakes_shapefile_2015, "r") as shape_file:
        for feature in shape_file:
            if glid == feature['properties']["GL_ID"]:
                area = feature['properties']["Area"]
                polygon = Polygon(feature["geometry"]["coordinates"][0])
                return area, polygon


def polygonize_raster_and_buffer(raster_fn, border_pixels=20):
    """Open and polygonize raster file."""
    mask = None
    with rasterio.Env():
        with rasterio.open(raster_fn) as src:
            image = src.read(1)[border_pixels:-border_pixels, border_pixels:-border_pixels] 
            results = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for i, (s, v) 
                        in enumerate(
                        shapes(image, mask=mask, transform=src.transform)))
    geometries = list(results)
    geoms = [geom for geom in geometries if geom['properties']['raster_val'] == 1]
    xy_polygons = []
    count = 0
    is_multi = False
    for geom in geoms:
        poly = shape(geom['geometry'])
        proj_meters = pyproj.Proj('epsg:3857')
        proj_latlng = pyproj.Proj('epsg:4326')

        project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
        pt_meters = transform(project_to_meters, poly)
        area = pt_meters.area/1000000
        if area < 0.05:
            continue
        buffer_size = data.get_buffer_from_area(area)
        buffered_polygon = buffer_polygon_in_meters(poly, buffer_size, percentage=.5)

        if buffered_polygon.geom_type == 'MultiPolygon':
            polygons = list(buffered_polygon)
            for poly in polygons:
                xy_buffered_polygon = []
                for i, (lon, lat) in enumerate(poly.exterior.coords):
                    py, px = src.index(lon, lat)
                    xy_buffered_polygon.append((px, py))
                xy_polygons.append(xy_buffered_polygon)
                count+=1
        else:
            for i, (lon, lat) in enumerate(buffered_polygon.exterior.coords):
                py, px = src.index(lon, lat)
                xy_polygons.append((px, py))
            count+=1
        if count > 1:
            is_multi = True

    return xy_polygons, is_multi