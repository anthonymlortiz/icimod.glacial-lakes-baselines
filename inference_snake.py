import os
import datetime
import argparse
import glob


import rasterio
import numpy as np

from models import snake
from utils import utils
import fiona
from shapely.geometry import Polygon, mapping, Point
from functools import partial
from shapely.ops import transform
import pyproj
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Smithsonian model inference script')
parser.add_argument('--input_dir', type=str, required=True, help='The path to the raster to run the model on')
parser.add_argument('--output_dir', type=str, required=True,  help='The path to output the model predictions as a GeoTIFF. Will fail if this file already exists.')
parser.add_argument('--overwrite', action="store_true",  help='Flag for overwriting `output_dir` if that file already exists.')
parser.add_argument('--image_source', default='bing', const='sentinel',
                    nargs='?', choices=['maxar', 'bing', 'sentinel'], help='Type of imagery to do inference on')
parser.add_argument('--verbose', action="store_true",  help='Print details of inference')
parser.add_argument('--gl_filename', default='/datadrive/snake/lakes/GL_3basins_2015.shp', type=str, help='The path to the glacial lakes vector data filename')

args = parser.parse_args()


def get_bing_glid_from_fn(bing_fn):
    base = os.path.basename(bing_fn)
    gl_id = os.path.splitext(base)[0]
    return gl_id

def get_sentinel_glid_from_fn(bing_fn):
    base = os.path.basename(bing_fn)
    gl_id = os.path.splitext(base)[0].splitext('_')
    return gl_id


def buffer_polygon_in_meters(polygon, buffer):
    proj_meters = pyproj.Proj(init='epsg:3857')
    proj_latlng = pyproj.Proj(init='epsg:4326')
    
    project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
    project_to_latlng = partial(pyproj.transform, proj_meters, proj_latlng)
    
    
    pt_meters = transform(project_to_meters, polygon)
    
    buffer_meters = pt_meters.buffer(buffer)
    buffer_polygon = transform(project_to_latlng, buffer_meters)
    return buffer_polygon


def get_glacial_lake_2015_outline_from_glid(lakes_shapefile_2015, glid):
    with fiona.open(lakes_shapefile_2015, "r") as shape_file:
        for feature in shape_file:
            if glid == feature['properties']["GL_ID"]:
                area = feature['properties']["Area"]
                polygon = Polygon(feature["geometry"]["coordinates"][0])
                return area, polygon 


def get_buffer_from_area(area, percentage):
    #Area in sq km
    return np.sqrt(1000000 * area)/percentage




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
                gl_id = get_sentinel_glid_from_fn(input_fn)
            elif args.image_source == "bing":
                gl_id = get_bing_glid_from_fn(input_fn)   
            else:
                raise NotImplementedError("Image source not supported")

            area, polygon = get_glacial_lake_2015_outline_from_glid(args.gl_filename, gl_id)
            buffer_size = get_buffer_from_area(area, -25)
            buffered_polygon = buffer_polygon_in_meters(polygon, buffer_size)
            xy_buffered_polygon = []
            for i, (lon, lat) in enumerate(buffered_polygon.exterior.coords):
                py, px = f.index(lon, lat)
                xy_buffered_polygon.append((px, py))
            snake_results, evolution = snake.snake_lakes_GAC_from_polygon(img_data, xy_buffered_polygon, iterations=250)

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
                np.save(output_fn_npz, np.asarray(evolution))

if __name__ == "__main__":
    main()
