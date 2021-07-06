import sys, shutil, os
from trainers import trainer_segmentation
from trainers.train_framework import TrainFramework
from models.losses import MulticlassCrossEntropy, WeightedBCELoss
from options.train_options import TrainOptions
from data.dataloader import load_dataset
from models.unet import UnetModel


# parse options
opts = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# Define model according to opts
if opts.model == "unet":
    model = UnetModel(opts)
#elif opts.model == "hrnet":
#    import yaml
#    with open(opts.cfg) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
#        config = yaml.load(file, Loader=yaml.FullLoader)
#        print(config)
#        model = get_seg_model(config, opts)
else:
    print("Option {} not supported. Available options: unet,fcn, hrnet".format(
        opts.model))
    raise NotImplementedError

# Define loss function or criterion based on opts
if opts.loss == "ce":
    loss = MulticlassCrossEntropy
elif opts.loss == "wbce":
    loss = WeightedBCELoss

else:
    print("Option {} not supported. Available options: ce, wbce".format(opts.loss))
    raise NotImplementedError

frame = TrainFramework(
    model,
    loss,
    opts
)

if opts.overwrite:
    print("Warning: You have chosen to overwrite previous training directory for this experiment")
    shutil.rmtree(opts.save_dir + "/" + opts.experiment_name + "/training")
    os.makedirs(opts.save_dir + "/" + opts.experiment_name + "/training")

dataloaders = load_dataset(opts)

if opts.model == "unet" or opts.model == "fcn" or opts.model == "hrnet" :
    _, train_history, val_history = trainer_segmentation.train(frame, dataloaders, opts)
else:
    print("Model {} not supported. Available options: unet".format(opts.model))
    raise NotImplementedError
