import sys
sys.path.append("..")
import utils.model_utils as mu
import models.snake as snake
import utils.data as dt
from options.infer_options import InferOptions
import torch
import pandas as pd
import rasterio
from tqdm import tqdm
from models.unet import UnetModel
from models.delse import DelseModel
from pathlib import Path
from warnings import warn, filterwarnings
filterwarnings("ignore", category=UserWarning)


# load the model according to opts
opts = InferOptions().parse()
if opts.model == "snake":
    pred_fun = mu.inference_gen(
    snake.infer,
    mu.processor_snake,
    mu.postprocessor_snake,
    device=opts.device
    )

else:
    if opts.model == "unet":
        model = UnetModel(opts)
    elif opts.model == "delse":
        model = DelseModel(opts)

    model.load_state_dict(torch.load(opts.model_pth))
    model.eval()
    model.to(opts.device)

    # function that will do inference
    base = Path(opts.data_dir)
    pred_fun = mu.inference_gen(

    model.infer,
    mu.processor_raster,
    mu.postprocessor_raster,
    device=opts.device
    )


base = Path(opts.data_dir)
stats_fn = base / "statistics.csv" 

# get paths and run inference
infer_paths = dt.inference_paths(
    base / opts.x_dir,
    base / opts.meta_dir,
    Path(opts.inference_dir),
    opts.subset_size
)

for _, (_, fn, meta_fn, out_fn_y, out_fn_prob) in tqdm(infer_paths.iterrows(), total=len(infer_paths)):
    y_hat, probs = pred_fun(fn, meta_fn, base / opts.stats_fn)
    x_meta = rasterio.open(fn).meta
    dt.save_raster(y_hat, x_meta, x_meta["transform"], out_fn_y)
    dt.save_raster(probs, x_meta, x_meta["transform"], out_fn_prob)
