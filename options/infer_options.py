from options.base_options import BaseOptions


class InferOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--chip_size', type=int, default=256, help='Size for each training patch')
        parser.add_argument('--device', type=str, default='cuda', help='Which device to use? [cuda | cpu]')
        parser.add_argument('--infer_paths', type=str, default='infer_test.csv', help='Relative paths to inference files')
        parser.add_argument('--stats_fn', type=str, default='le7-2015/splits/train/statistics.csv', help='Relative path to stats.csv?')
        return parser


class EvalOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.set_defaults(save_dir="le7-2015/evaluate_test")
        parser.add_argument("--eval_paths", type=str, default="eval.csv")
        parser.add_argument("--vector_label", type=str, default="GL_3basins_2015.shp")
        parser.add_argument("--buffer", type=float, default=0.01)
        return parser
