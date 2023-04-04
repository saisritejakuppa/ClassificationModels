import torch
from models.model import get_model, ModelTrain

from options.train_options import TrainOptions

def main():
    opt = TrainOptions().parse()
    opt.modelname = 'resnet34'
    model = get_model(opt)
    print(model)

    datasetpath = '/home/saiteja/ades_intense_week_gonna_deal_things_myself/JJ_complete_helmet/Helmet_2'
    model_training = ModelTrain(opt)


if __name__ == '__main__':
    main()