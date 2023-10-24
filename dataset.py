import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms

import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

def init_dataset(dataset_name, preprocess, split='train', labels=None):
    """
    Load the dataset and apply the provided preprocessing function.

    Example usage:
        mnist_train = mnist_dataset('mnist', split='train', labels=[0, 1, 2])
        mnist_test = mnist_dataset('mnist', split='test', labels=[0, 1, 2])
    """

    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Apply the transformation and preprocessing to each example
    processed_data = []
    for example in dataset:
        image = preprocess(example['image'])
        processed_example = {'image': image, 'label': example['label']}
        processed_data.append(processed_example)
    
    # Filter the dataset by labels if provided
    if labels is not None:
        processed_data = [example for example in processed_data if example['label'] in labels]
    
    print(f"Number of examples in the {split} split: {len(processed_data)}")
    
    return processed_data

