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
    elif dataset_name == 'EvalCifar10':
        preprocess = transforms.Compose([
            transforms.Resize((224), antialias=None),
        ])
    else:
        raise NotImplementedError

    if flip:
        preprocess.transforms.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    if rotate:
        preprocess.transforms.insert(0, transforms.RandomRotation(degrees=15))

    return preprocess

def get_task_labels(batch_size, num_tasks):
    """
    Generate labels for the tasks.

    Args:
        batch_size (int): The total number of images in a batch.
        num_tasks (int): The number of tasks.

    Returns:
        torch.Tensor: A tensor containing the labels for each image in the batch.
    """
    # Get the number of images per task
    num_imgs_per_task = batch_size // num_tasks
    # Initialize the labels which will be used to get the task noise
    labels = torch.zeros(batch_size)
    # Create equal amount of labels for each task
    for i in range(num_tasks):
        labels[i * num_imgs_per_task : (i + 1) * num_imgs_per_task] = i
    return labels


def get_task_noise(num_imgs, size, label):
    """
    Generate task-specific noise for each image in a batch.

    Args:
        num_imgs (int): Number of images in the batch.
        size (tuple): Size of each image in the batch.
        label (int): Task label for which the noise is generated.

    Returns:
        torch.Tensor: Task-specific noise tensor of shape (num_imgs, *size).
    """
    
    # Initialize the noise to return
    output_size = (num_imgs,) + size
    noise_to_return = torch.randn(size=output_size)

    # Initialize the condition vector
    v = torch.ones_like(noise_to_return[0])

    # Iterate over the batches
    for noise_idx, i_noise in enumerate(noise_to_return):
        
        # Compute the dot product between 
        # the noise and the condition vector
        m = torch.sum(i_noise * v)

        # For different tasks, the condition is different
        if label == 0:
            # For task 0, the condition is m > 0
            while m < 0: 
                i_noise = torch.randn_like(i_noise)
                m = torch.sum(i_noise * v) 
        elif label == 1:
            # For task 1, the condition is m < 0
            while m > 0:
                i_noise = torch.randn_like(i_noise)
                m = torch.sum(i_noise * v)

        # Add the task noise to the noise_to_return
        noise_to_return[noise_idx] = i_noise
    
    return noise_to_return