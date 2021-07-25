from options.base_options import BaseOptions


class InferOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--x_dir', type=str, default='le7-2015/splits/val/images', help='Relative paths to imagery on which to perform inference')
        parser.add_argument('--meta_dir', type=str, default='le7-2015/splits/val/meta', help='Relative paths to metadata on which to perform inference')
        parser.add_argument('--inference_dir', type=str, default='/datadrive/results/inference/landsat_val-unet/', help='Absolute path to directory where results should be saved')
        parser.add_argument('--stats_fn', type=str, default='le7-2015/splits/val/statistics.csv', help='Relative path to stats.csv')
        parser.add_argument('--model_pth', type=str, default='/datadrive/results/save/unet-landsat7_best.pth', help='Absolte path to model')
        return parser


class EvalOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.set_defaults(save_dir="/datadrive/results/inference/landsat_val-unet")
        parser.add_argument("--inference_dir", type=str, default="/datadrive/results/inference/landsat_val-unet")
        parser.add_argument("--vector_label", type=str, default="/datadrive/snake/lakes/GL_3basins_2015.shp")
        parser.add_argument("--buffer", type=float, default=0.01)
        parser.add_argument("--tol", type=float, default=1e-4)
        return parser
