from options.base_options import BaseOptions


class InferOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.set_defaults(data_dir="/datadrive/snake/lakes/sentinel-2015/processed/")
        parser.add_argument('--x_dir', type=str, default='sentinel/splits/train/images', help='Relative paths to imagery on which to perform inference')
        parser.add_argument('--meta_dir', type=str, default='sentinel/splits/train/meta', help='Relative paths to metadata on which to perform inference')
        parser.add_argument('--inference_dir', type=str, default='/datadrive/results/inference/sentinel_train-unet/', help='Absolute path to directory where results should be saved')
        parser.add_argument('--stats_fn', type=str, default='sentinel/splits/train/statistics.csv', help='Relative path to stats.csv')
        parser.add_argument('--divergence', default=False, action='store_true', help='Add divergence feature?')
        parser.add_argument('--model_pth', type=str, default='/datadrive/results/backup/sentinel-unet_best.pth', help='Absolute path to model')
        parser.add_argument('--chip_size', type=int, default=1024, help='Model input / output chip dimension')
        parser.add_argument('--historical', default=False, action='store_true', help='Add shrunken historical label?')
        return parser


class EvalOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.set_defaults(save_dir="/datadrive/results/inference/bing_val-unet")
        parser.add_argument("--vector_label", type=str, default="/datadrive/snake/lakes/GL_3basins_2015.shp")
        parser.add_argument("--buffer", type=float, default=1e-2)
        parser.add_argument("--tol", type=float, default=1e-6)
        parser.add_argument("--grid", type=int, default=20)
        parser.add_argument("--geo_prob", type=float, default=0.6)
        parser.add_argument("--n_jobs", type=int, default=50)
        parser.add_argument("--eval_paths", type=str, default="/datadrive/results/inference/bing_val_paths.csv")
        parser.add_argument("--fname", type=str, default="metrics.csv")
        return parser


class PathOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument("--image_dir", type = str, default="/datadrive/glaciers/bing_glaciers/processed/bing/", help="What directory contains full set of images to give predictions for?")
        parser.add_argument("--inference_dir", type = str, default="/datadrive/results/inference/compressed/results/bing-unet/", help="Where are the predictions stored?")
        parser.add_argument("--output_dir", type = str, default="/datadrive/results/inference/eval_paths/", help="Where to save the validation paths?")
        parser.add_argument("--labeling_dir", type = str, default="/datadrive/glaciers/bing_glaciers/labeling/", help="What directory contains subset of images with recent imagery?")
        parser.add_argument("--mode", type=str, default="pred")
        return parser
