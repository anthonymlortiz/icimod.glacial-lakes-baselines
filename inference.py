import sys
sys.path.append("..")
import utils.model_utils as mu
import utils.data as dt
from options.train_options import TrainOptions
import torch
import pandas as pd
import rasterio
from models.unet import UnetModel
from models.networks import DelseModel
from pathlib import Path
from warnings import warn, filterwarnings
filterwarnings("ignore", category=UserWarning)

# load the model according to opts
opts = TrainOptions().parse()
opts.checkpoint_file = "/datadrive/results/save/bing_test_best.pth"  # needs to be changed....
opts.infer_paths = "/datadrive/snake/lakes/infer_test.csv"
if opts.model == "unet":
    model = UnetModel(opts)
elif opts.model == "delse":
    model = DelseModel(opts)

model.load_state_dict(torch.load(opts.checkpoint_file))
model.eval()
model.to(opts.device)

# function that will do inference
base = Path(opts.data_dir)
stats_fn = base / "statistics.csv"
pred_fun = mu.inference_gen(
    model.infer,
    mu.processor_test,
    mu.postprocessor_test,
    device=opts.device
)

# get paths and run inference
infer_paths = pd.read_csv(opts.infer_paths)
fns = [base / s for s in infer_paths.fn.values]
meta_fns = [base / s for s in infer_paths.meta_fn.values]

for i, (fn, meta_fn, out_fn) in infer_paths.iterrows():
    _, probs = pred_fun(base / fn, base / meta_fn)
    x_meta = rasterio.open(base / fn).meta
    dt.save_raster(probs, x_meta, x_meta["transform"], base / out_fn)
