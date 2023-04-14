from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        #dataset root path
        self.parser.add_argument('--dataroot', type=str, default='/home/saiteja/detectwork/helmetdetection/completedataset/helmetclassification_helmetai', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        #model name
        self.parser.add_argument('--modelpath', type=str, default='resnet18', help='model path')

        #no of classes
        self.parser.add_argument('--num_classes', type=int, default=3, help='number of classes')

        #json file for labels
        self.parser.add_argument('--class2idx', type=str, default='model_output/labels.json', help='json file for labels')

        self.isTrain = False

