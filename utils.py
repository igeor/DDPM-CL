import torch 
from torchvision import transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image

def interpolate(batch, mode='RGB', size=299):
    arr = []
    for img in batch:
        if img.shape[0] == 1: img = img.repeat(3,1,1)
        pil_img = transforms.ToPILImage(mode=mode)(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

def gather(consts: torch.Tensor, t: torch.Tensor):
      c = consts.gather(-1, t)
      return c.reshape(-1, 1, 1, 1)

# create function for selecting preprocessing function
def get_preprocess_function(dataset_name, flip=False, rotate=False):
    if dataset_name == 'MNIST':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat single channel to get three channels
            transforms.Resize((32), antialias=None),
            transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
        ])
    elif dataset_name == 'CIFAR10':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
        ])
    elif dataset_name == 'CelebA':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
        ])
    elif dataset_name == 'LSUN':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
        ])
    elif dataset_name == 'FFHQ':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
        ])
    else:
        raise NotImplementedError

    if flip:
        preprocess.transforms.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    if rotate:
        preprocess.transforms.insert(0, transforms.RandomRotation(degrees=15))

    return preprocess