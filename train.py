import torch
from models.model import get_model, ModelTrain

from options.train_options import TrainOptions
from data.dataloading import get_dataloader, save_images_dataloader, ImageDirDataset, augmentations
import os

def main():

    opt = TrainOptions().parse()

    #dataloader

        #create a dictionary of augmentations
    aug_dict = {
        'blur': {'blur_limit': 3},
        # 'RandomHorizontalFlip': {},
        # 'RandomVerticalFlip': {},
        # 'RandomRotation': {'limit': 30},
        # 'GaussianBlur': {'blur_limit': 3},
        # 'ColorJitter': {}
    }

    custom_dataset = ImageDirDataset(opt.dataroot, transform=augmentations(aug_dict))

    # #shuffle the dataset
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


    os.makedirs('images', exist_ok=True)


    #save images from the dataloader
    save_images_dataloader(dataloader['train'], 'images/train.png')
    save_images_dataloader(dataloader['test'], 'images/test.png')
    save_images_dataloader(dataloader['val'], 'images/val.png')


    #model initialization
    opt.modelname = 'resnet34'
    model = get_model(opt)


    #model training
    modeltrain = ModelTrain(opt)

    modeltrain.train(model, dataloader)

    print(model)



if __name__ == '__main__':
    main()