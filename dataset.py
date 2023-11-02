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

def init_dataset(dataset_name, preprocess='default', split='train', labels=None):
    """
    Load the dataset and apply the provided preprocessing function.

    Example usage:
        mnist_train = mnist_dataset('mnist', split='train', labels=[0, 1, 2])
        mnist_test = mnist_dataset('mnist', split='test', labels=[0, 1, 2])
    """

    # Define the default preprocessing function
    if isinstance(preprocess, str) and preprocess == 'default':
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
        ])
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Apply the transformation and preprocessing to each example
    processed_data = []
    for example in dataset:
        image = example['image'].convert("RGB")
        image = preprocess(image)
        processed_example = {'image': image, 'label': example['label']}
        processed_data.append(processed_example)
    
    # Filter the dataset by labels if provided
    if labels is not None:
        processed_data = [example for example in processed_data if example['label'] in labels]
    
    print(f"Number of examples in the {split} split: {len(processed_data)}")
    
    return processed_data


def init_v2_dataset(dataset_name, preprocess='default', split=None, labels=None):

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
        # Download the MNIST dataset
        full_trainset = datasets.MNIST(dataset_name, download=True, train=True, transform=preprocess)
        # Filter the data to only include images with the corresponding labels
        indices = torch.tensor([label in labels for label in full_trainset.targets])
        trainset = torch.utils.data.Subset(full_trainset, indices.nonzero().squeeze().tolist())
        print(f'Number of training images: {len(trainset)}')

    if split in [None, 'test']:
        # Repeat the process for the test data
        full_testset = datasets.MNIST(dataset_name, download=True, train=False, transform=preprocess)
        # Filter the data to only include images with the corresponding labels
        indices = torch.tensor([label in labels for label in full_testset.targets])
        testset = torch.utils.data.Subset(full_testset, indices.nonzero().squeeze().tolist())
        print(f'Number of test images: {len(testset)}')

    if split == 'train': return trainset
    elif split == 'test': return testset
    else: return trainset, testset