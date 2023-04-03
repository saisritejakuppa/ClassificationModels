import torch
import timm

import sys
sys.path.append('/home/saiteja/extra/ClassificationModels')

from options.train_options import TrainOptions
# from data.dataloading import get_dataloader, save_images_dataloader

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



class ModelTrain:
    def __init__(self, opt):
        self.opt = opt

    
    def get_optimizer(self):

        if self.get_optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.lr)

        
        elif self.get_optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)
        
        elif self.get_optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)

        elif self.get_optimizer == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)

        
        return optimizer
    

    def get_loss(self):

        if self.opt.loss == 'MSE':
            loss_fn = torch.nn.MSELoss()

        elif self.opt.loss == 'CrossEntropy':
            loss_fn = torch.nn.CrossEntropyLoss()
            
    
    def train(self, dataloader):
        model = get_model(self.opt)

        val_losses = []
        train_losses = []

        optimizer = self.get_optimizer()
        loss_fn = self.get_loss()

        for epoch in range(self.opt.n_epochs):

            train_loss = 0.0
            val_loss = 0.0

            model.train()
            for data, target in dataloader['train']:
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)

            model.eval()
            for data, target in dataloader['val']:
                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()*data.size(0)

            train_loss = train_loss/len(dataloader['train'].sampler)
            val_loss = val_loss/len(dataloader['val'].sampler)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch+1, 
                train_loss,
                val_loss
                ))




if __name__ == '__main__':

    opt = TrainOptions().parse()
    opt.modelname = 'resnet34'
    model = get_model(opt)
    print(model)

