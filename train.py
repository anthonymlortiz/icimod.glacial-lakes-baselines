import sys, shutil, os
from trainers import trainer_segmentation
from trainers.train_framework import Algorithm
from models.losses import MulticlassCrossEntropy, WeightedBCELoss
from options.train_options import TrainOptions
from data.dataloader import load_dataset
from models.unet import UnetModel
from utils.metrics import mean_IOU
from tensorboardX import SummaryWriter


# parse options
opts = TrainOptions().parse()
print(' '.join(sys.argv))

# Define model according to opts
if opts.model == "unet":
    model = UnetModel(opts)
else:
    assert NotImplementedError, f"Option {opts.model} not supported. Available options: unet,fcn, hrnet"

# Define loss function or criterion based on opts
if opts.loss == "ce":
    loss = MulticlassCrossEntropy
elif opts.loss == "wbce":
    loss = WeightedBCELoss
else:
    assert NotImplementedError, f"Option {opts.loss} not supported. Available options: ce, wbce"

if opts.overwrite:
    print("Warning: You have chosen to overwrite previous training directory for this experiment")
    shutil.rmtree(opts.save_dir + "/" + opts.experiment_name + "/training")
    os.makedirs(opts.save_dir + "/" + opts.experiment_name + "/training")

metrics = {"mIOU": mean_IOU}
frame = Algorithm(model, loss, opts)
datasets = load_dataset(opts)
writer = SummaryWriter(opts.save_dir)
trainer_segmentation.train_(frame, datasets, writer, opts)
