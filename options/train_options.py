from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        #dataset root path
        self.parser.add_argument('--dataroot', type=str, default='/home/saiteja/experimentation_classification/datasets/Train_val/refined', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        #model name
        self.parser.add_argument('--modelname', type=str, default='resnet18', help='name of the experiment. It decides where to store samples and models')

        #no of classes
        self.parser.add_argument('--num_classes', type=int, default=3, help='number of classes')

        #intermediate layers and their neurons
        self.parser.add_argument('--intermediate_layers', type=list, default=[128, 64], help='intermediate layers and their neurons')

        #dropout
        self.parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

        #activation function
        self.parser.add_argument('--activation_fn', type=str, default='ReLU', help='activation function to use')


        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')

        #epochs
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')

        #batch size
        self.parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')


        #augmentation_toml
        self.parser.add_argument('--augmentations_toml', type=str, default='metadata/augmentation.toml', help='augmentation toml file')


        # ==============================> Optimizer <================================



        #optimizer
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')

        #learning rate
        self.parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')

        #auto learning rate finder
        # self.parser.add_argument('--auto_lr_finder', type=bool, default=False, help='auto learning rate finder')

        #weight decay
        self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

        #momentum
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

        #step size
        self.parser.add_argument('--step_size', type=int, default=7, help='step size')

        #gamma
        self.parser.add_argument('--gamma', type=float, default=0.1, help='gamma')

        # learning rate scheduler
        self.parser.add_argument('--lr_schedule', type=str, default='step', help='learning rate scheduler to use')
        self.parser.add_argument('--lr_step_size', type=int, default=7, help='step size for StepLR scheduler')
        self.parser.add_argument('--lr_milestones', type=int, nargs='+', default=[30, 60, 90], help='milestones for MultiStepLR scheduler')
        self.parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma for ExponentialLR, ReduceLROnPlateau, and CyclicLR schedulers')
        self.parser.add_argument('--lr_patience', type=int, default=5, help='patience for ReduceLROnPlateau scheduler')
        self.parser.add_argument('--lr_min', type=float, default=0.0001, help='minimum learning rate for CosineAnnealingLR and CyclicLR schedulers')

        # early_stop_epochs
        self.parser.add_argument('--early_stop_epochs', type=int, default=10, help='early stop epochs')

        #early stop patience
        self.parser.add_argument('--early_stop_delta', type=int, default=5, help='early stop delta')



        # ==============================> Loss <================================

        #loss function
        self.parser.add_argument('--loss_fn', type=str, default='CrossEntropyLoss', help='loss function to use')

        #device
        self.parser.add_argument('--device', type=str, default='cuda', help='device to use')


        #wandb
        self.parser.add_argument('--wandb', type=bool, default=True, help='log the parameters')

        self.isTrain = True