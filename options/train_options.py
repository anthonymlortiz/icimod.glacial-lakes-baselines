from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        # Options for training
        parser.add_argument('--n_epochs', type=int, default=30, help='# of training epochs')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training model [adam | sgd]')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')

        parser.add_argument('--chip_size', type=int, default=256, help='Size for each training patch')
        parser.add_argument('--divergence', default=False, action='store_true', help='Add divergence feature?')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--loss', type=str, default='wce', help='loss for training model [ce | wbce | dicebce | delse]')
        parser.add_argument('--scheduler_patience', type=int, default=5, help='lr scheduler patience')

        parser.add_argument('--max_grad_norm', type=float, default=10, help='Gradient clipping')
        parser.add_argument('--val_metric', type=str, default='IoU', help='Which metric to use for model selection? [IoU | precision | recall]')
        parser.add_argument('--save_epoch', type=int, default=10, help='How frequently to save checkpoints?')
        self.isTrain = True
        return parser
