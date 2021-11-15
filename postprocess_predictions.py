import os
import datetime
import argparse
import glob


import rasterio
import numpy as np

from models import snake
from utils import snake_utils
from utils import utils
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Smithsonian model inference script')
parser.add_argument('--input_dir', type=str, required=True, help='The path to the raster to run the model on')
parser.add_argument('--predictions_dir', type=str, required=True, help='The path to the predictions directory')
parser.add_argument('--output_dir', type=str, required=True,  help='The path to output the model predictions as a GeoTIFF. Will fail if this file already exists.')
parser.add_argument('--overwrite', action="store_true",  help='Flag for overwriting `output_dir` if that file already exists.')
parser.add_argument('--image_source', default='bing', const='sentinel',
                    nargs='?', choices=['maxar', 'bing', 'sentinel'], help='Type of imagery to do inference on')
parser.add_argument('--verbose', action="store_true",  help='Print details of inference')
parser.add_argument('--n_jobs', type=int, default=100, help='How many processes to run in parallel?')
parser.add_argument('--evolution_store_frequency', type=int, default=10, help='After how many snake iterations should the evolution be saved?')

args = parser.parse_args()


def process_input(input_fn, prediction_fn):
    output_fn = args.output_dir + str(os.path.basename(input_fn)).replace(".tif", "_postprocessed_pred.tif")
    with utils.Timer("loading input", args.verbose):
        with rasterio.open(input_fn) as f:
            input_profile = f.profile.copy()
            img_data = np.moveaxis(f.read(), 0, -1)

        if args.image_source == "bing":
            xy_polygons, is_multipolygon = snake_utils.polygonize_raster_and_buffer(prediction_fn, border_pixels=80)
        else:
            xy_polygons, is_multipolygon = snake_utils.polygonize_raster_and_buffer(prediction_fn)

        if is_multipolygon:
            snake_results, evolution = snake.snake_lakes_GAC_from_multipolygon(img_data, xy_polygons, iterations=120)
        else:
            snake_results, evolution = snake.snake_lakes_GAC_from_polygon(img_data, xy_polygons, iterations=120)


        #-------------------
        # Save output
        #-------------------
        with utils.Timer("writing output", args.verbose):
            output_profile = input_profile.copy()
            output_profile["dtype"] = "uint8"
            output_profile["count"] = 1
            output_profile["nodata"] = 0

            with rasterio.open(output_fn, "w", **output_profile) as f:
                f.write(evolution[-1], 1)
                f.write_colormap(1, {
                    0: (0, 0, 0, 0),
                    1: (255, 0, 0, 255)
                })

def main():
    if args.verbose:
        print("Starting Glacial Lakes project Predictions postprocessing script at %s" % (str(datetime.datetime.now())))


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
    #for fn in tqdm(filenames):
    #    process_input(fn, args.predictions_dir + str(os.path.basename(fn)).replace(".tif", "_pred.tif") )
    Parallel(n_jobs=args.n_jobs)(delayed(process_input)(fn, args.predictions_dir + str(os.path.basename(fn)).replace(".tif", "_pred.tif")) for fn in tqdm(filenames))


if __name__ == "__main__":
    main()
