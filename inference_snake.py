import os
import datetime
import argparse
import glob


import rasterio
import numpy as np

from models import snake
from utils import data, snake_utils
from utils import utils
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Smithsonian model inference script')
parser.add_argument('--input_dir', type=str, required=True, help='The path to the raster to run the model on')
parser.add_argument('--output_dir', type=str, required=True,  help='The path to output the model predictions as a GeoTIFF. Will fail if this file already exists.')
parser.add_argument('--overwrite', action="store_true",  help='Flag for overwriting `output_dir` if that file already exists.')
parser.add_argument('--image_source', default='bing', const='sentinel',
                    nargs='?', choices=['maxar', 'bing', 'sentinel'], help='Type of imagery to do inference on')
parser.add_argument('--verbose', action="store_true",  help='Print details of inference')
parser.add_argument('--gl_filename', default='/datadrive/snake/lakes/GL_3basins_2015.shp', type=str, help='The path to the glacial lakes vector data filename')
parser.add_argument('--n_jobs', type=int, default=100, help='How many processes to run in parallel?')
parser.add_argument('--evolution_store_frequency', type=int, default=10, help='After how many snake iterations should the evolution be saved?')

args = parser.parse_args()


def process_input(input_fn):
    output_fn = args.output_dir + str(os.path.basename(input_fn)).replace(".tif", "_pred.tif")
    with utils.Timer("loading input", args.verbose):

        with rasterio.open(input_fn) as f:
            input_profile = f.profile.copy()
            img_data = np.moveaxis(f.read(), 0, -1)

        #-------------------
        # Run model
        #-------------------
        r, c, _ = img_data.shape
        if args.image_source == "sentinel":
            gl_id = snake_utils.get_sentinel_glid_from_fn(input_fn)
        elif args.image_source == "bing":
            gl_id = snake_utils.get_bing_glid_from_fn(input_fn)
        else:
            raise NotImplementedError("Image source not supported")

        area, polygon = snake_utils.get_glacial_lake_2015_outline_from_glid(args.gl_filename, gl_id)
        buffer_size = data.get_buffer_from_area(area)
        buffered_polygon = snake_utils.buffer_polygon_in_meters(polygon, buffer_size, 0.5)
        if buffered_polygon.geom_type == 'MultiPolygon':
            polygons = list(buffered_polygon)
            xy_polygons = []
            for poly in polygons:
                xy_buffered_polygon = []
                for i, (lon, lat) in enumerate(poly.exterior.coords):
                    py, px = f.index(lon, lat)
                    xy_buffered_polygon.append((px, py))
                xy_polygons.append(xy_buffered_polygon)
            if args.image_source == "sentinel":
                snake_results, evolution = snake.snake_lakes_GAC_from_multipolygon(img_data, xy_polygons, iterations=50)
            else:
                snake_results, evolution = snake.snake_lakes_GAC_from_multipolygon(img_data, xy_polygons, iterations=150)
        else:
            xy_buffered_polygon = []
            for i, (lon, lat) in enumerate(buffered_polygon.exterior.coords):
                py, px = f.index(lon, lat)
                xy_buffered_polygon.append((px, py))
            if args.image_source == "sentinel":
                snake_results, evolution = snake.snake_lakes_GAC_from_polygon(img_data, xy_buffered_polygon, iterations=50)
            else:
                snake_results, evolution = snake.snake_lakes_GAC_from_polygon(img_data, xy_buffered_polygon, iterations=150)

        #-------------------
        # Save output
        #-------------------
        with utils.Timer("writing output", args.verbose):
            output_profile = input_profile.copy()
            output_profile["dtype"] = "uint8"
            output_profile["count"] = 1
            output_profile["nodata"] = 0

            with rasterio.open(output_fn, "w", **output_profile) as f:
                f.write(snake_results, 1)
                f.write_colormap(1, {
                    0: (0, 0, 0, 0),
                    1: (255, 0, 0, 255)
                })

            output_fn_npz = output_fn.replace('.tif', '.npy')
            np.save(output_fn_npz, np.asarray(evolution)[::args.evolution_store_frequency])


def main():
    if args.verbose:
        print("Starting Glacial Lakes project Snake inference script at %s" % (str(datetime.datetime.now())))


    #-------------------
    # Load files
    #-------------------
    ## Ensure input dir exists
    assert os.path.exists(args.input_dir)


    ## Ensure output directory exists
    if os.path.exists(args.output_dir):
        if args.overwrite:
            if args.verbose: print("WARNING! The output file, %s, already exists, and we are overwriting it!" % (args.output_dir))
        else:
            print("The output file, %s, already exists, and we don't want to overwrite it, exiting..." % (args.output_dir))
            return

    filenames = glob.glob(args.input_dir +"*.tif")
    Parallel(n_jobs=args.n_jobs)(delayed(process_input)(fn) for fn in tqdm(filenames))


if __name__ == "__main__":
    main()
