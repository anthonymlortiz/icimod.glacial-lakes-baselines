import argparse

class DownloadOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--save_dir', type=str, default='/datadrive/snake/lakes/sentinel-tests/', help='Absolute of the directory to create and save results')
        parser.add_argument('--area_filter', type=float, default=0.25, help='Lakes with 2015 area below this quantile will not be downloaded')
        parser.add_argument('--vector_labels', type=str, default='/datadrive/snake/lakes/GL_3basins_2015.shp', help='Absolute path to shapefiles with the area of interest geoms to query')
        parser.add_argument('--buffer', type=float, default=1e-4, help='How much should we buffer the area of interests before downloading?')
        parser.add_argument('--resize', type=float, default=500, help='If the downloaded mask for a lake is below this size, it will be resized up to this.')
        parser.add_argument('--max_cloud', type=float, default=35, help='What is the maximum amount of cloud coverage that we will tolerate?')
        parser.add_argument('--max_nodata', type=float, default=20, help='What is the maximum acceptable nodata in the overall image from which we build the mask?')
        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser
        return opt

    def parse(self, save=False):
        opt = self.gather_options()
        self.print_options(opt)
        return opt
