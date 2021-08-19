from math import sqrt
from pathlib import Path
from rasterio.enums import Resampling
from shapely.geometry import Point, box, Polygon
from utils.data import save_raster
import geopandas as gpd
import numpy as np
import planetary_computer as pc
import rasterio
import rasterio.warp as rw
import shutil
import subprocess


# Some tricks to make rasterio faster when using vsicurl -- see https://github.com/pangeo-data/cog-best-practices
RASTERIO_BEST_PRACTICES = dict(
    CURL_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt',
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    AWS_NO_SIGN_REQUEST='YES',
    GDAL_MAX_RAW_BLOCK_CACHE_SIZE='200000000',
    GDAL_SWATH_SIZE='200000000',
    VSI_CURL_CACHE_SIZE='200000000'
)

def to_square(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    centroid = [(maxx+minx)/2, (maxy+miny)/2]
    diagonal = sqrt((maxx-minx)**2+(maxy-miny)**2)
    return Point(centroid).buffer(diagonal/2, cap_style=3)


def fetch_hrefs(catalog, aoi, time_range, max_nodata=20, n_scenes=15):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=aoi.bounds,
        datetime=time_range,
        query={"s2:nodata_pixel_percentage": {"lt": max_nodata}}
    )

    items = list(search.get_items())
    if len(items) == 0:
        raise ValueError("No items satisfy query!")
    items = sorted(items, key=lambda z: z.properties["eo:cloud_cover"])[:n_scenes]

    result = []
    for item in items:
        links = (
            pc.sign(item.assets["visual"].href),
            item.properties,
            pc.sign(item.assets["SCL"].href)
        )
        result.append(links)
    return result

def pass_checks(f, geom, scl, max_cloud, max_snow):
    cloud_frac = (scl == 3) + (scl == 7) + (scl == 8) + (scl == 9) + (scl == 10)

    if not box(*f.bounds).contains(Polygon(geom["coordinates"][0])):
        return False
    elif np.mean(cloud_frac) > max_cloud:
        return False
    elif np.mean(scl == 11) > max_snow:
        return False
    return True


def download_(vis_ref, cloud_ref, props, geom, buffer=0.001, max_cloud=0.05,
              max_snow=0.7):
    geom = to_square(geom.buffer(buffer))
    geom = rw.transform_geom("epsg:4326", f"epsg:{props['proj:epsg']}", geom)

    with rasterio.Env(**RASTERIO_BEST_PRACTICES):
        with rasterio.open(cloud_ref) as f:
            scl, _ = rasterio.mask.mask(f, [geom], crop=True, invert=False, pad=False, all_touched=True)
            if not pass_checks(f, geom, scl, max_cloud, max_snow):
                return None, None, None

        with rasterio.open(vis_ref) as f:
            image, transform = rasterio.mask.mask(f, [geom], crop=True, invert=False, pad=False, all_touched=True)
            meta = f.meta
            if np.all(np.isin(image, [0, 255])):
                return None, None, None

            return image, meta, transform


def download(catalog, geom, time_range, buffer=0.001, max_nodata=0.2,
             max_cloud=0.05, max_snow=0.7, n_scenes=15):
    items = fetch_hrefs(catalog, geom, time_range, max_nodata, n_scenes)
    results = []
    for (vis_ref, props, scl_ref) in items:
        image, meta, transform = download_(vis_ref, scl_ref, props, geom,
                                           buffer, max_cloud, max_snow)
        if image is not None:
            results.append((image, meta, transform, props))

    return results


def upscale(input_path, out_path, scale=1):
    if scale <= 1:
        shutil.move(input_path, out_path)
        return

    with rasterio.open(input_path) as f:
        data = f.read(
            out_shape=(f.count, int(f.height * scale), int(f.width * scale)),
            resampling=Resampling.bilinear
        )

        transform = f.transform * f.transform.scale(
            (f.width / data.shape[-1]),
            (f.height / data.shape[-2])
        )
        tmp_path = Path(f"tmp-{out_path}")
        save_raster(data, f.meta, transform, out_path)
