import torch
from models.model import get_model, ModelTrain

from options.train_options import TrainOptions
from data.dataloading import get_dataloader, save_images_dataloader, ImageDirDataset, augmentations
import os
import toml
import json
import wandb
from logzero import logger


def main(opt):

    # logger.info('configs')
    # print(opt)

    wandb.init(project = 'classification sweeps')
    opt = wandb.config
    
    logger.info('configs')
    print(opt)


    aug_dict = toml.load(opt.augmentations_toml)

    custom_dataset = ImageDirDataset(opt.dataroot, transform=augmentations(aug_dict))

    #shuffle the dataset
    torch.manual_seed(0)
    # custom_dataset = torch.utils.data.Subset(custom_dataset, torch.randperm(len(custom_dataset)).tolist())
    
    #train, test, val split equal to 0.8, 0.1, 0.1
    train_size = int(0.8 * len(custom_dataset))
    test_size = int(0.1 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size - test_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size, val_size])

    dataloader = {
        'train': get_dataloader(train_dataset, batch_size=opt.batch_size),
        'test': get_dataloader(test_dataset, batch_size=opt.batch_size),
        'val': get_dataloader(val_dataset, batch_size=opt.batch_size)
    }


    #print the classes in the dataset
    print('classes in the dataset are', custom_dataset.classes)

    #print mapped to ints in the dataset    
    print('classes mapped to ints in the dataset are', custom_dataset.class_to_idx)

    #save this as a jsono to the opt.model_save_path
    with open(f'{opt.model_save_path}/class_to_idx.json', 'w') as f:
        json.dump(custom_dataset.class_to_idx, f)

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

if __name__ == '__main__':
    opt = TrainOptions().parse()
    main(opt)