"""
This script downloads lake imagery from Sentinel, using the Planetary Computer

Some characteristics of our pipeline,
  * We read in only the areas around lakes of interest. Their boundaries are
   defined by the "Glacial Lakes" dataset in the ICIMOD Regional Database
   Service.
  * We only download the lakes within the top 75% of areas. The other lakes are
   so small that it's hard to make out their boundary (they also pose less
   threat for GLOF).
  * For lakes that would have returned a masked image of size less than 500 x
   500 pixels, we resample up to 500 in the x-dimension. The resize dimension is
   an option that can be modified through the --out_size argument.
  * We download one image each month from 2015
"""
from options.download_options import DownloadOptions
from osgeo import gdal
from pathlib import Path
from pystac_client import Client
from rasterio.enums import Resampling
from tempfile import NamedTemporaryFile
from tqdm import tqdm
import csv
from datetime import datetime
import geopandas as gpd
import shutil
import utils.download as udl
import utils.data as udt
opts = DownloadOptions().parse()


catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
lakes = gpd.read_file(opts.vector_labels)
lakes = lakes.set_index("GL_ID")
lakes = lakes[lakes.Area > lakes.Area.quantile(opts.area_filter)]

# prepare directory to save the results
opts.save_dir = Path(opts.save_dir)
if (opts.save_dir).exists():
    shutil.rmtree(opts.save_dir)
(opts.save_dir).mkdir(parents=True)

# What will become our metadata file
fields = ["filename", "year", "month", "s2:granule_id", "s2:product_uri",
          "datetime", "s2:high_proba_clouds_percentage", "s2:mean_solar_zenith",
          "s2:mean_solar_azimuth", "s2:water_percentage",
          "s2:cloud_shadow_percentage", "s2:high_proba_clouds_percentage"]
f = open(opts.save_dir / "metadata.csv", "a")
writer = csv.writer(f)
writer.writerow(fields)

for lake_id in tqdm(lakes.index):
    for year in range(2015, 2022):
        geom = lakes.loc[lake_id]["geometry"]
        time_range = f"{year}-01-01/{year}-12-31"
        out_path = f"{lake_id}_{year}"

        try:
            results = udl.download(
                catalog,
                geom,
                time_range,
                opts.buffer,
                opts.max_nodata,
                opts.max_cloud,
                opts.max_snow,
                opts.n_scenes
            )

            for i, result in enumerate(results):
                image, metas, transforms, props = result
                dt = datetime.strptime(props["datetime"], "%Y-%m-%dT%H:%M:%S.%fZ")
                out_path_i = opts.save_dir / f"{out_path}-{dt.month:02}-{dt.day:02}.tif"

                # resample the image if it is very small
                tmp = [Path(NamedTemporaryFile().name) for _ in range(2)]
                udt.save_raster(image, metas, transforms, tmp[0])
                scale = max(opts.resize / image.shape[1], opts.resize / image.shape[2])
                udl.upscale(tmp[0], tmp[1], scale)

                # warp the image and save as a raster
                udt.save_raster(image, metas, transforms, out_path_i)
                gdal.Warp(str(out_path_i), gdal.Open(str(tmp[1])), dstSRS="EPSG:4326")
                [s.unlink() for s in tmp]

                # save the metadata
                writer.writerow([str(out_path_i.stem)] + [props[s] for s in fields[3:]])
        except:
            writer.writerow([out_path])

f.close()
