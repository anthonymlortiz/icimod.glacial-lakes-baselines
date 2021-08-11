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
from calendar import monthrange
import csv
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
    geom = lakes.loc[lake_id]["geometry"]

    for year in range(2015, 2022):
        for month in range(1, 13):
            out_path = opts.save_dir / f"{lake_id}-{year}{month:02}.tif"
            if year == 2022 and month > 8:
                continue

            try:
                last_day = monthrange(year, month)[1]
                time_range = f"{year}-{month:02}-01/{year}-{month:02}-{last_day}"
                image, meta, transform, props = udl.download(
                    catalog,
                    geom,
                    time_range,
                    opts.buffer,
                    max_cloud=opts.max_cloud,
                    max_nodata=opts.max_nodata
                )

                # resample the image if it is very small
                tmp = [Path(NamedTemporaryFile().name) for _ in range(2)]
                udt.save_raster(image, meta, transform, out_path)
                udt.save_raster(image, meta, transform, tmp[0])
                udl.upscale(tmp[0], tmp[1], opts.resize / image.shape[1])

                # warp the image and save as a raster
                gdal.Warp(str(out_path), gdal.Open(str(tmp[1])), dstSRS="EPSG:4326")
                [s.unlink() for s in tmp]

                # save the metadata
                writer.writerow([str(out_path.stem)] + [props[s] for s in fields[3:]])

            except ValueError as e:
                writer.writerow([str(out_path.stem)])

f.close()
