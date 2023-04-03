from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)


        #model name
        self.parser.add_argument('--modelname', type=str, default='resnet18', help='name of the experiment. It decides where to store samples and models')

        #no of classes
        self.parser.add_argument('--num_classes', type=int, default=2, help='number of classes')

        #intermediate layers and their neurons
        self.parser.add_argument('--intermediate_layers', type=list, default=[512,256,128], help='intermediate layers and their neurons')

        #dropout
        self.parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

        #activation function
        self.parser.add_argument('--activation_fn', type=str, default='ReLU', help='activation function to use')

        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        
        #epochs
        self.parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')

        #batch size
        self.parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')

        #optimizer
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')

        #learning rate
        self.parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')

        #scheduler
        self.parser.add_argument('--scheduler', type=str, default='StepLR', help='scheduler to use')

        #weight decay
        self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

        #momentum
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

        #step size
        self.parser.add_argument('--step_size', type=int, default=7, help='step size')

        #gamma
        self.parser.add_argument('--gamma', type=float, default=0.1, help='gamma')

        #loss function
        self.parser.add_argument('--loss_fn', type=str, default=['CrossEntropyLoss','precision','recall'], help='loss function to use')

        self.isTrain = True