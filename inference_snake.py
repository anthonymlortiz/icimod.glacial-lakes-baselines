import os
import datetime
import argparse
import glob


import rasterio
import numpy as np

from models import snake
from utils import utils

parser = argparse.ArgumentParser(description='Smithsonian model inference script')
parser.add_argument('--input_dir', type=str, required=True, help='The path to the raster to run the model on')
parser.add_argument('--output_dir', type=str, required=True,  help='The path to output the model predictions as a GeoTIFF. Will fail if this file already exists.')
parser.add_argument('--overwrite', action="store_true",  help='Flag for overwriting `output_dir` if that file already exists.')
parser.add_argument('--image_source', default='bing', const='sentinel',
                    nargs='?', choices=['maxar', 'bing', 'sentinel'], help='Type of imagery to do inference on')
parser.add_argument('--verbose', action="store_true",  help='Print details of inference')

args = parser.parse_args()


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


    #-------------------
    # Load input
    #-------------------
    filenames = glob.glob(args.input_dir +"*.tif")
    for input_fn in filenames:
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
                snake_results, _ = snake.snake_lakes_GAC(img_data, (int(r/2), int(c/2)))
            else:
                snake_results, _ = snake.snake_lakes_GAC(img_data, (int(r/2), int(c/2)), ls_radious=50, igs_alpha=1000)

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

if __name__ == "__main__":
    main()
