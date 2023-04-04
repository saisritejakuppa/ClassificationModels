import torch
import timm
from logzero import logger

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

        if self.opt.optimizer== 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)

        elif self.opt.optimizer== 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)
        
        elif self.opt.optimizer== 'RMSprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)

        elif self.opt.optimizer== 'Adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)
        
        else:
            logger.info(f'choosen optimmizer is { self.opt.optimizer}')
            logger.info('Optimizer not supported')
            raise ValueError('Optimizer not supported')

        return optimizer
    

    def get_loss(self):

        if self.opt.loss_fn == 'MSE':
            loss_fn = torch.nn.MSELoss()

        elif self.opt.loss_fn == 'CrossEntropyLoss':
            loss_fn = torch.nn.CrossEntropyLoss()

        elif self.opt.loss_fn == 'BCE':
            loss_fn = torch.nn.BCELoss()
        
        else:
            logger.info(f'choosen loss function is { self.opt.loss_fn}')
            logger.info('Loss function not supported')
            raise ValueError('Loss function not supported')
    
        return loss_fn
    
    
    def train(self,model,dataloader):

        self.model = model

        #model to gpu based on device
        self.model = self.model.to(self.opt.device)

        val_losses = []
        train_losses = []

        optimizer = self.get_optimizer()
        loss_fn = self.get_loss()


        for epoch in range(self.opt.n_epochs):

            train_loss = 0.0
            val_loss = 0.0

            train_acc = 0.0
            val_acc = 0.0

            # ------------------------------------------------------training the module
            self.model.train()
            for current_step, (data, target) in enumerate(dataloader['train'], 0):

                #convert to device
                data = data.to(self.opt.device)
                target = target.to(self.opt.device)
                
                optimizer.zero_grad()
                output = self.model(data)

                #get the loss
                loss = loss_fn(output, target)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)
                predictions = torch.argmax(output, dim=1)
                train_acc += torch.sum(predictions == target).item()
                #display the step loss
                if current_step % self.opt.display_freq == 0:
                    print('Epoch: {} \tStep: {} \tTraining Loss: {:.6f}'.format(
                        epoch+1, 
                        current_step, 
                        loss.item()
                        ))

            # --------------------------------------------------------validation the module
            self.model.eval()
            for data, target in dataloader['val']:
                
                #convert to device
                data = data.to(self.opt.device)
                target = target.to(self.opt.device)

                output = self.model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()*data.size(0)
                predictions = torch.argmax(output, dim=1)
                val_acc += torch.sum(predictions == target).item()

            #--------------------------------------------------------saving the losses
            train_loss = train_loss/len(dataloader['train'].sampler)
            val_loss = val_loss/len(dataloader['val'].sampler)
            train_acc = train_acc/len(dataloader['train'].sampler)
            val_acc = val_acc/len(dataloader['val'].sampler)

            train_losses.append(train_loss)
            val_losses.append(val_loss)


            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Acc: {:.6} \tValidation Acc: {:.6}'.format(
                epoch+1, 
                train_loss,
                val_loss,
                train_acc,
                val_acc
                ))



if __name__ == '__main__':

    opt = TrainOptions().parse()
    opt.modelname = 'resnet34'
    model = get_model(opt)
    print(model)