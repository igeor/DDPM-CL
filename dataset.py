import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from torchvision import datasets, transforms
import torch

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

def init_tvision_dataset(
        dataset_name, 
        dataset_path=None, 
        target_dir='./data', # Default target directory if dataset_path is None
        preprocess=None, 
        labels=None
    ):
    
    # Define the default preprocessing function if preprocess is None or it different type than transforms.Compose
    if preprocess is None or type(preprocess) != transforms.Compose:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat single channel to get three channels
            transforms.Resize((32), antialias=None)
        ])

    DatasetClass = getattr(datasets, dataset_name)

    # If dataset path not exist (None) define a default path
    if dataset_path:
        full_trainset = DatasetClass(dataset_path, download=False, train=True, transform=preprocess)
    else:
        full_trainset = DatasetClass(root=target_dir, download=True, train=True, transform=preprocess)

    # Filter the data to only include images with the corresponding labels
    if labels is None: labels = list(range(len(set(full_trainset.targets))))
    indices = torch.tensor([label in labels for label in full_trainset.targets])
    trainset = torch.utils.data.Subset(full_trainset, indices.nonzero().squeeze().tolist())

    if dataset_path:
        full_testset = DatasetClass(dataset_path, download=False, train=False, transform=preprocess)
    else:
        full_testset = DatasetClass(root=target_dir, download=True, train=False, transform=preprocess)
        
    # Filter the data to only include images with the corresponding labels
    indices = torch.tensor([label in labels for label in full_testset.targets])
    testset = torch.utils.data.Subset(full_testset, indices.nonzero().squeeze().tolist())
    print(f'Number of training images: {len(trainset)}, Number of test images: {len(testset)}')

    return trainset, testset


def init_folder_dataset(dataset_path, labels=None, preprocess=None):

    # Check if the dataset path exist
    if not os.path.exists(dataset_path):
        raise ValueError(f'Invalid dataset path: {dataset_path}')
    
    # Define the default preprocessing function if preprocess is None or it different type than transforms.Compose
    if preprocess is None or type(preprocess) != transforms.Compose:
        # Define transformations (optional)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128), antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # Create an ImageFolder dataset
    dataset = ImageFolder(root=dataset_path, transform=preprocess)

    # Filter the data to only include images with the corresponding labels
    if labels is not None:
        indices = torch.tensor([label in labels for label in dataset.targets])
        dataset = torch.utils.data.Subset(dataset, indices.nonzero().squeeze().tolist())

    print(f'Number of images in the dataset: {len(dataset)}')

    return dataset


if __name__ == '__main__':

    # Define transformations (optional)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64), antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Define the dataset path
    dataset_path = '/data/datasets/ImageNet/train'

    # Create an ImageFolder dataset
    # 0 to 100
    labels = list(range(100))
    trainset = init_folder_dataset(dataset_path, labels=labels, preprocess=preprocess)

    dataloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    
    print(len(dataloader))
    