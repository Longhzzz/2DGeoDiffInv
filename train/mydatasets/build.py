import torch 
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from .seis_dataset import SeisDataset
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from .transforms_image import create_image_mask_transforms, create_seis_image_mask_transforms


def create_transforms(image_size):
    
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])



def create_dataset(dataset_name, args, split='train'):

    if dataset_name == "Seis":
        dataset = SeisDataset(args.train_data_dir, split=split, image_size=args.resolution,
                                         transform=create_seis_image_mask_transforms(args.resolution, split=='train'),
                                         )
    else:
        raise NotImplementedError

    return dataset
        
        