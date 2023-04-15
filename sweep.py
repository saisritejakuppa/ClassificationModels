from train import main
import wandb
from options.train_options import TrainOptions
import torch
from models.model import get_model, ModelTrain
from data.dataloading import get_dataloader, save_images_dataloader, ImageDirDataset, augmentations
import os
import toml
import json
import wandb
from logzero import logger
import yaml


def main():    


    default_config = TrainOptions().parse()
    default_configs = vars(default_config)
    wandb.init(config = default_configs)

    opt = wandb.config
    logger.info('configs')
    print(opt)

    aug_dict = toml.load(opt.augmentations_toml)
    # custom_dataset = ImageDirDataset(opt.dataroot, transform=augmentations(aug_dict))



    train_dataset = ImageDirDataset(os.path.join(opt.dataroot, 'train'), transform=augmentations(aug_dict))

    #make no augmentations for val and test datasets( except for resizing and normalizing )
    aug_dict = {}

    val_dataset = ImageDirDataset(os.path.join(opt.dataroot, 'val'), transform=augmentations(aug_dict))
    test_dataset = ImageDirDataset(os.path.join(opt.dataroot, 'test'), transform=augmentations(aug_dict))

    dataloader = {
        'train': get_dataloader(train_dataset, batch_size=opt.batch_size),
        'test': get_dataloader(test_dataset, batch_size=opt.batch_size),
        'val': get_dataloader(val_dataset, batch_size=opt.batch_size)
    }

    #print mapped to ints in the dataset    
    print('classes mapped to ints in the dataset are', test_dataset.class_to_idx)

    os.makedirs(opt.model_save_path, exist_ok=True)

    #save this as a jsono to the opt.model_save_path
    with open(f'{opt.model_save_path}/class_to_idx.json', 'w') as f:
        json.dump(test_dataset.class_to_idx, f)

    os.makedirs(f'{opt.model_save_path}/images', exist_ok=True)

    #save images from the dataloader
    save_images_dataloader(dataloader['train'], f'{opt.model_save_path}/images/train.png')
    save_images_dataloader(dataloader['test'], f'{opt.model_save_path}/images/test.png')
    save_images_dataloader(dataloader['val'], f'{opt.model_save_path}/images/val.png')

    model = get_model(opt)

    #model training
    modeltrain = ModelTrain(opt)
    modeltrain.train(model, dataloader)

    #save all the opt values to a file in the opt.model_save_path
    with open(f'{opt.model_save_path}/opt.txt', 'w') as f:
        for k, v in vars(modeltrain.opt).items():
            f.write(f'{k}: {v} \n')




with open('sweeps/model_parameters.yaml') as file:
    sweep_config = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep_config, project='ClassificationModels')

# start the sweep
wandb.agent(sweep_id, function=main)