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

def init_dataset(dataset_name=None, dataset_path=None, preprocess='default', split=None, labels=None):
    # Define the default preprocessing function
    if isinstance(preprocess, str) and preprocess == 'default':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat single channel to get three channels
            transforms.Resize((32), antialias=None)
        ])

    # if labels is none 
    if labels is None: labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if split in [None, 'train']:
        # Check if dataset_name is provided
        if dataset_name:
            full_trainset = datasets.MNIST(dataset_name, download=True, train=True, transform=preprocess)
        # If not, check if dataset_path is provided
        elif dataset_path:
            full_trainset = datasets.MNIST(dataset_path, download=False, train=True, transform=preprocess)
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided.")

        # Filter the data to only include images with the corresponding labels
        indices = torch.tensor([label in labels for label in full_trainset.targets])
        trainset = torch.utils.data.Subset(full_trainset, indices.nonzero().squeeze().tolist())
        print(f'Number of training images: {len(trainset)}')

    if split in [None, 'test']:
        # Repeat the process for the test data
        if dataset_name:
            full_testset = datasets.MNIST(dataset_name, download=True, train=False, transform=preprocess)
        elif dataset_path:
            full_testset = datasets.MNIST(dataset_path, download=False, train=False, transform=preprocess)
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided.")

        # Filter the data to only include images with the corresponding labels
        indices = torch.tensor([label in labels for label in full_testset.targets])
        testset = torch.utils.data.Subset(full_testset, indices.nonzero().squeeze().tolist())
        print(f'Number of test images: {len(testset)}')

    if split == 'train': return trainset
    elif split == 'test': return testset
    else: return trainset, testset