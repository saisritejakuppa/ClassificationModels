import torch
import timm
from options.train_options import TrainOptions

def get_model(opt):

    base_model = timm.create_model(opt.modelname, pretrained=True)

    # add intermediate layers and their neurons
    num_ftrs = base_model.fc.in_features
    layers = [num_ftrs] + opt.intermediate_layers + [opt.num_classes]
    modules = []
    for i in range(len(layers) - 2):
        modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
        modules.append(torch.nn.Dropout(opt.dropout))
        modules.append(torch.nn.__dict__[opt.activation_fn]())
    
    modules.append(torch.nn.Linear(layers[-2], layers[-1]))
    base_model.fc = torch.nn.Sequential(*modules)

    return base_model


if __name__ == '__main__':

    opt = TrainOptions().parse()

    opt.modelname = 'resnet34'

    model = get_model(opt)

    print(model)



    