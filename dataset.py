import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from torchvision import datasets, transforms
import torch

import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

def init_dataset(
        dataset_name, 
        dataset_path=None, 
        store_path='./data',
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
        full_trainset = DatasetClass(root=store_path, download=True, train=True, transform=preprocess)

    # Filter the data to only include images with the corresponding labels
    if labels is None: labels = list(range(len(set(full_trainset.targets))))
    indices = torch.tensor([label in labels for label in full_trainset.targets])
    trainset = torch.utils.data.Subset(full_trainset, indices.nonzero().squeeze().tolist())
    print(f'Number of training images: {len(trainset)}')
    

    if dataset_path:
        full_testset = DatasetClass(dataset_path, download=False, train=False, transform=preprocess)
    else:
        full_testset = DatasetClass(root=store_path, download=True, train=False, transform=preprocess)
        
    # Filter the data to only include images with the corresponding labels
    indices = torch.tensor([label in labels for label in full_testset.targets])
    testset = torch.utils.data.Subset(full_testset, indices.nonzero().squeeze().tolist())
    print(f'Number of test images: {len(testset)}')

    return trainset, testset