import argparse

class ProcessOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--in_dir', type=str, default='/datadrive/snake/lakes/sentinel-2015/', help='Absolute path to imagery to preprocess')
        parser.add_argument('--label_path', type=str, default='/datadrive/snake/lakes/GL_3basins_2015.shp', help='Absolute path to shapefiles to use for labeling')
        parser.add_argument('--out_dir', type=str, default='/datadrive/snake/lakes/sentinel-2015/processed/sentinel/splits', help='Absolute path to directory to save train / val / test splits')
        parser.add_argument('--split', type=bool, default=False, help='Split into train / test / val or just process without shuffling?')
        parser.add_argument('--subset_size', type=int, default=None, help='For test runs, can preprocess just a subset of size subset_size')
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
