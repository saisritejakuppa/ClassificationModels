import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2


class ImageDirDataset(ImageFolder):
    """
    PyTorch dataset for loading images from a directory.
    """
    def __init__(self, imgdirpath, transform=None):
        """
        Args:
            imgdirpath (str): Path to the directory containing the images.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        """
        self.imgdirpath = imgdirpath
        self.transform = transform or transforms.ToTensor()
        super().__init__(self.imgdirpath, transform=self.transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the image to be loaded.
        Returns:
            tuple: (image, label) where label is the index of the target class.
        """
        img_path, label = self.samples[index]
        # with open(img_path, 'rb') as f:
        #     image = Image.open(f).convert('RGB')

        #open the image with cv2
        image = cv2.imread(img_path)

        #convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image = image)['image']
        return image, label


def augmentations(aug_dict, default_prob=0.5):
    """Augmentations for the dataset.

    Args:
        aug_dict (dict): Dictionary containing the names of the augmentations as keys and their corresponding parameters.
        default_prob (float): Default probability value to use if not specified in the toml file.

    Returns:
        torchvision.transforms.Compose: Augmentations for the dataset.
    """
    custom_transforms = []

    # resize the images to 224x224
    custom_transforms.append(A.Resize(224, 224))

    # use albumentations for augmentations
    aug_map = {
        'blur': A.Blur,
        'RandomHorizontalFlip': A.HorizontalFlip,
        'RandomVerticalFlip': A.VerticalFlip,
        'RandomRotation': A.Rotate,
        'GaussianBlur': A.GaussianBlur,
        'ColorJitter': A.ColorJitter,
        'RandomCrop': A.RandomCrop,
        'RandomBrightnessContrast': A.RandomBrightnessContrast,
        'Cutout': A.Cutout,
        'RandomResizedCrop': A.RandomResizedCrop,
        'RandomRain': A.RandomRain,
        'RandomSnow': A.RandomSnow,
        'RandomSunFlare': A.RandomSunFlare,
        'RandomFog': A.RandomFog,
        'RandomSnow': A.RandomSnow
    }

    for aug_name, aug_params in aug_dict.items():
        if aug_name in aug_map:
            aug_fn = aug_map[aug_name]
            aug_fn_args = {k: v for k, v in aug_params.items() if k not in ['p']}
            aug_fn_prob = aug_params.get('p', default_prob)
            # if aug_name == 'RandomSnow':
            #     aug_fn_args['value'] = aug_params.get('value', 50)
            custom_transforms.append(aug_fn(p=aug_fn_prob, **aug_fn_args))

    # normalize the images to cifar mean and std
    custom_transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    custom_transforms.append(ToTensorV2())

    custom_transforms = A.Compose(custom_transforms)
    return custom_transforms






def save_images_dataloader(custom_dataloader, imgname):
    """Save images from the dataloader.

    Args:
        custom_dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
        imgname (str): Name of the output image file.
    """
    images, _ = next(iter(custom_dataloader))  # get a batch of images from the dataloader
    grid = torchvision.utils.make_grid(images)  # create a grid of images

    # save the grid as an image file
    torchvision.utils.save_image(grid, imgname)



def get_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True):
    """
    Returns a PyTorch DataLoader for a given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)




if __name__ == '__main__':

    datasetpath = '/home/saiteja/detectwork/helmetdetection/completedataset/helmetclassification_helmetai'

    batch_size = 4

    import toml
    aug_dict = toml.load('/home/saiteja/detectwork/ClassificationModels/metadata/augmentation.toml')


    os.makedirs('images', exist_ok=True)


    for aug in aug_dict.keys():
        print(aug)

        aug_individual = {aug: aug_dict[aug]}

        #create a custom dataset
        custom_dataset = ImageDirDataset(datasetpath, transform=augmentations(aug_individual))

        #create a dataloader for the dataset
        custom_dataloader = get_dataloader(custom_dataset, batch_size=batch_size)

        #save images from the dataloader
        save_images_dataloader(custom_dataloader, f'images/{aug}.png')