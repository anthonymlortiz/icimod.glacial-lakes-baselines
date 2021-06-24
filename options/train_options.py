from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
            
        #Options for training
        parser.add_argument('--n_epochs', type=int, default=10, help='# of training epochs')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training model [adam | sgd]')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
     
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--chip_size', type=int, default=512, help='Size for each training patch')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--loss', type=str, default='ce', help='loss for training model [ce | dice | focal]')
        parser.add_argument('--scheduler_patience', type=int, default=5, help='lr scheduler patience')

        parser.add_argument('--finetune_model', action='store_true', default=False, help='if set framework will finetune model. Also provide model to finetune')
        parser.add_argument('--model_to_finetune', type=str, default='none', help='if set framework will finetune model. Also provide model to finetune')


        #HRNET
        parser.add_argument('--cfg',
                        help='experiment configure file for HRNet',
                        default='options/hrnet.yaml',
                        type=str)

        self.isTrain = True
        return parser