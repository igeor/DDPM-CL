import torch 
from diffusers.utils.torch_utils import randn_tensor
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule, get_constant_schedule_with_warmup
from torchvision import transforms as transforms
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

def get_label_vector(batch_size, labels):
    """
    Generate a batch of labels given a list of labels.
    Function will repeat the labels to match the batch size.

    Args:
        batch_size (int): The total size of the batch.
        labels (list): The list of labels for each task.

    Returns:
        torch.Tensor: The concatenated label vector for all tasks in the batch.
    """
    n = batch_size // len(labels)
    labels_to_return = [torch.ones(n) * label for label in labels]
    return torch.cat(labels_to_return[:batch_size], dim=0)

        
def sample_task_noise(noise_shape, labels, generator=None, device=None, dtype=None):
    """
    Sample task-specific noise based on labels and condition vector.

    Args:
        noise_shape (tuple): The shape of the noise tensor to generate.
        labels (list): The labels corresponding to each noise sample.
        generator (torch.Generator, optional): Generator object for random number generation. Defaults to None.
        device (torch.device, optional): The device to place the generated noise tensor on. Defaults to None.
        dtype (torch.dtype, optional): The data type of the generated noise tensor. Defaults to None.

    Returns:
        torch.Tensor: The generated task-specific noise tensor.
    """

    # Make sure that the number of labels is equal to the number of noise samples
    assert noise_shape[0] == len(labels)
      
    # Initialize the noise to return
    noise_to_return = randn_tensor(noise_shape, 
                                   generator=generator,
                                   device=device, dtype=dtype)
    
    # If there are no labels, return the noise
    if labels == []: 
        return noise_to_return

    # Initialize the condition vector
    v = torch.ones_like(noise_to_return[0])

    # Iterate over the batches
    for noise_idx, i_noise in enumerate(noise_to_return):
        
        # Compute the dot product between 
        # the noise and the condition vector
        m = torch.sum(i_noise * v)

        # Get the label of the image
        label = labels[noise_idx]

        # For different tasks, the condition is different
        if (label == 0 and m < 0) or \
            (label == 1 and m > 0): 
            noise_to_return[noise_idx] = - i_noise

    # Return the noise
    return noise_to_return


def get_lr_scheduler(scheduler_name, optimizer, num_warm_up_steps=None, num_steps=None):
    """
    Get the learning rate scheduler based on the scheduler name.
    Args:
        scheduler_name (str): Name of the scheduler.
        optimizer: The optimizer for which the learning rate schedule is generated.
            Options are: 'const', 'const_warm', 'cos', 'cos_warm'.
        num_warm_up_steps (int, optional): Number of warm-up steps for the learning rate schedule.
            Applicable only for the 'const_warm' and 'cos_warm' learning rate schedules.
        num_steps (int, optional): Total number of training steps.
            Applicable only for the 'cos_warm' learning rate scheduler.

    Returns:
        Learning rate schedule based on the scheduler name.

    Raises:
        NotImplementedError: If the scheduler name is not supported.
    """

    if scheduler_name == "const":
        return get_constant_schedule(optimizer)
    elif scheduler_name == "const_warm":
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warm_up_steps
        )
    elif scheduler_name == "cos_warm":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warm_up_steps,
            num_training_steps=num_steps
        )
    else:
        raise NotImplementedError