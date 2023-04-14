import torch
import timm
from logzero import logger
import os

import sys
sys.path.append('/home/saiteja/extra/ClassificationModels')

from options.train_options import TrainOptions
# from data.dataloading import get_dataloader, save_images_dataloader
from tqdm import tqdm



def get_model(opt):

    base_model = timm.create_model(opt.modelname, pretrained=True)

    # add intermediate layers and their neurons
    num_ftrs = base_model.fc.in_features
    layers = [num_ftrs] + opt.intermediate_layers + [opt.num_classes]

    modules = []
    for i in range(len(layers) - 2):
        modules.append(torch.nn.Linear(layers[i], layers[i + 1]))

        if opt.dropout > 0:
            modules.append(torch.nn.Dropout(opt.dropout))

        if opt.activation_fn is not None:
            modules.append(torch.nn.__dict__[opt.activation_fn]())
    
    modules.append(torch.nn.Linear(layers[-2], layers[-1]))
    base_model.fc = torch.nn.Sequential(*modules)

    # Add sigmoid activation if using BCEWithLogitsLoss
    if opt.loss_fn == 'BCEWithLogitsLoss':
        base_model.add_module('sigmoid', torch.nn.Sigmoid())

    return base_model



class ModelTrain:
    def __init__(self, opt):
        self.opt = opt
        self.best_val_loss = float('inf')
        self.best_val_epoch = -1

        

    
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



        if self.opt.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_step_size, gamma=self.opt.lr_gamma)
        elif self.opt.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.opt.lr_milestones, gamma=self.opt.lr_gamma)
        elif self.opt.lr_schedule == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.opt.lr_gamma)
        elif self.opt.lr_schedule == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.opt.lr_gamma, patience=self.opt.lr_patience, verbose=True)
        elif self.opt.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opt.n_epochs, eta_min=self.opt.lr_min)
        elif self.opt.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.opt.lr_min, max_lr=self.opt.lr, step_size_up=self.opt.lr_step_size, mode='triangular2')
        else:
            scheduler = None

        return optimizer, scheduler    

    def get_loss(self):

        if self.opt.loss_fn == 'MSE':
            loss_fn = torch.nn.MSELoss()

        elif self.opt.loss_fn == 'CrossEntropyLoss':
            loss_fn = torch.nn.CrossEntropyLoss()

        elif self.opt.loss_fn == 'BCE':
            loss_fn = torch.nn.BCELoss()

        #sigmoid
        elif self.opt.loss_fn == 'BCEWithLogitsLoss':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        else:
            logger.info(f'choosen loss function is { self.opt.loss_fn}')
            logger.info('Loss function not supported')
            raise ValueError('Loss function not supported')
    
        return loss_fn
    

    def save_model(self, model, epoch, val_loss):
        # Save model at current epoch
        current_path = os.path.join(self.opt.model_save_path, f"model_{epoch}.pt")
        torch.save(model.state_dict(), current_path)

        # Save model with best validation loss
        if val_loss < self.best_val_loss:
            best_path = os.path.join(self.opt.model_save_path, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            self.best_val_loss = val_loss
            self.best_val_epoch = epoch

    def train(self, model, dataloader):
        self.model = model

        # Move model to GPU based on device
        self.model = self.model.to(self.opt.device)

        optimizer, scheduler = self.get_optimizer()
        loss_fn = self.get_loss()

        val_losses = []
        train_losses = []



        # Initialize early stopping variables
        early_stop = False
        early_stop_counter = 0
        best_val_loss = float('inf')

        for epoch in range(self.opt.n_epochs):
            train_loss = 0.0
            val_loss = 0.0

            train_acc = 0.0
            val_acc = 0.0

            # # Update learning rate
            # if scheduler is not None:
            #     scheduler.step()

            # Training
            self.model.train()
            progress_bar = tqdm(dataloader['train'], desc=f"Epoch {epoch+1}")
            for data, target in progress_bar:
                data = data.to(self.opt.device)
                target = target.to(self.opt.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                if self.opt.loss_fn == 'BCEWithLogitsLoss':output = model.sigmoid(output.squeeze())
                else: pass

                print(output.shape)

                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                predictions = torch.argmax(output, dim=1)
                train_acc += torch.sum(predictions == target).item()

                # Update progress bar
                progress_bar.set_postfix({'training_loss': f"{loss.item():.6f}"})

            # Validation
            self.model.eval()
            with torch.no_grad():
                for data, target in dataloader['val']:
                    data = data.to(self.opt.device)
                    target = target.to(self.opt.device)
                    output = self.model(data)

                    if opt.loss_fn == 'BCEWithLogitsLoss':output = model.sigmoid(output.squeeze())
                    else: pass


                                        
                    loss = loss_fn(output, target)
                    val_loss += loss.item() * data.size(0)
                    predictions = torch.argmax(output, dim=1)
                    val_acc += torch.sum(predictions == target).item()

            # Saving losses
            train_loss = train_loss / len(dataloader['train'].sampler)
            val_loss = val_loss / len(dataloader['val'].sampler)
            train_acc = train_acc / len(dataloader['train'].sampler)
            val_acc = val_acc / len(dataloader['val'].sampler)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Print results
            print(f"Epoch {epoch+1}: training_loss = {train_loss:.6f}, validation_loss = {val_loss:.6f}, training_acc = {train_acc:.6f}, validation_acc = {val_acc:.6f}")


            #save the results based on the epoch
            self.opt.best_model_path = os.path.join(self.opt.model_save_path, f"{self.opt.modelname}_{epoch}_best.pth")
            self.opt.current_model_path = os.path.join(self.opt.model_save_path, f"{self.opt.modelname}_{epoch}_current.pth")


            # Early stopping based on validation loss
            if val_loss < best_val_loss - self.opt.early_stop_delta:
                early_stop_counter = 0
                best_val_loss = val_loss
                # Save the current model checkpoint
                torch.save(self.model.state_dict(), 'current_model.pt')
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.opt.early_stop_epochs:
                    early_stop = True
                    break
                

            # Check if current epoch model is the best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.opt.best_model_path)

            # Save current epoch model
            torch.save(self.model.state_dict(), self.opt.current_model_path)

            # Learning rate scheduling
            if self.opt.lr_schedule == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            elif self.opt.lr_schedule != 'None':
                scheduler.step()


        return train_losses, val_losses


if __name__ == '__main__':

    opt = TrainOptions().parse()
    opt.modelname = 'resnet34'
    model = get_model(opt)
    print(model)