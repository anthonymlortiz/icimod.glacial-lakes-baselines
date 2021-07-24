import sys
sys.path.append("..")
import utils.model_utils as mu
import utils.data as dt
from options.infer_options import InferOptions
import torch
import pandas as pd
import rasterio
from models.unet import UnetModel
from models.networks import DelseModel
from pathlib import Path
from warnings import warn, filterwarnings
filterwarnings("ignore", category=UserWarning)

# load the model according to opts
opts = InferOptions().parse()
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
    mu.processor_test,
    mu.postprocessor_test,
    device=opts.device
)

# get paths and run inference
infer_paths = dt.inference_paths(
    base / opts.x_dir,
    base / opts.meta_dir,
    base / opts.inference_dir
)
fns = [base / s for s in infer_paths.fn.values]
meta_fns = [base / s for s in infer_paths.meta_fn.values]

for i, (_, fn, meta_fn, out_fn_y, out_fn_prob) in infer_paths.iterrows():
    y_hat, probs = pred_fun(base / fn, base / meta_fn, base / opts.stats_fn)
    x_meta = rasterio.open(base / fn).meta
    dt.save_raster(y_hat, x_meta, x_meta["transform"], base / out_fn_y)
    dt.save_raster(probs, x_meta, x_meta["transform"], base / out_fn_prob)
