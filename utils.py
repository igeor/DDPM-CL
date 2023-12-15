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
    elif dataset_name == 'ImageFolder':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128), antialias=None),
            transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
        ])
    else:
        raise NotImplementedError

    if flip:
        preprocess.transforms.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    if rotate:
        preprocess.transforms.insert(0, transforms.RandomRotation(degrees=15))

    return preprocess

def create_repeated_values_vector(int_list, size):
    """
    Create a vector of repeated values based on the given integer list and size.

    Args:
        int_list (list): The list of integers to repeat.
        size (int): The desired size of the resulting vector.

    Returns:
        torch.Tensor: The vector of repeated values.
    """
    repeat_times = size // len(int_list)
    remainder = size % len(int_list)
    repeated_list = torch.repeat_interleave(torch.tensor(int_list), repeat_times)
    remainder_list = torch.repeat_interleave(torch.tensor(int_list[:remainder]), 1)
    return torch.cat((repeated_list, remainder_list))

        
def sample_task_noise(sample_noise, labels, timesteps, t_thres=1000, generator=None, device=None, dtype=None):
    """
    Sample task-specific noise based on labels and condition vector.

    Args:
        sample_noise (torch.Tensor): The default noise which needs to be modified based on the labels.
        labels (list): The labels corresponding to each noise sample.
        timesteps (torch.Tensor): The timestep for each noise sample.
        t_thres (int, optional): The timestep threshold for the task-specific noise. Defaults to 1000.
        generator (torch.Generator, optional): Generator object for random number generation. Defaults to None.
        device (torch.device, optional): The device to place the generated noise tensor on. Defaults to None.
        dtype (torch.dtype, optional): The data type of the generated noise tensor. Defaults to None.

    Returns:
        torch.Tensor: The generated task-specific noise tensor.
    """

    # Make sure that the number of labels is equal to the number of noise samples
    assert sample_noise.shape[0] == len(labels)
    # Make sure that the number of timesteps is equal to the number of noise samples
    assert sample_noise.shape[0] == len(timesteps)
      
    # Initialize the noise to return
    noise_to_return = sample_noise
    
    # Initialize the condition vector
    v = torch.ones_like(noise_to_return[0])

    # Iterate over the batches
    for noise_idx, i_noise in enumerate(noise_to_return):
        
        # Compute the dot product between 
        # the noise and the condition vector
        m = torch.sum(i_noise * v)

        # Get the label of the image
        label = labels[noise_idx]

        # Get the timestep of the image
        t = timesteps[noise_idx]

        if t >= t_thres:
            # For different tasks, the condition is different
            if (label == 0 and m < 0) or \
                (label == 1 and m > 0): 
                noise_to_return[noise_idx] = - i_noise

    # Return the noise
    return noise_to_return


def penalty_loss(pred_noise, labels, timesteps, t_thres=1000):
    """
    Compute the penalty loss for the predicted noise based on the given labels and timesteps.

    Args:
        pred_noise (torch.Tensor): The predicted noise tensor.
        labels (torch.Tensor): The labels tensor.
        timesteps (torch.Tensor): The timesteps tensor.
        t_thres (int, optional): The threshold value for timesteps. Defaults to 1000.

    Returns:
        torch.Tensor: The computed penalty loss tensor.

    """
    loss_to_return = torch.zeros(pred_noise.shape[0])

    v = torch.ones_like(pred_noise[0])

    for pred_noise_idx, i_pred_noise in enumerate(pred_noise):
        
        # Compute the dot product between 
        # the noise and the condition vector
        m = torch.sum(i_pred_noise * v)

        # Get the label and timestep of the corresponding noisy image
        label = labels[pred_noise_idx]
        t = timesteps[pred_noise_idx]

        if t >= t_thres:
            if (label == 0 and m < 0) or (label == 1 and m > 0): 
                loss_to_return[pred_noise_idx] = -m 
        else:
            loss_to_return[pred_noise_idx] = 0
    
    # Compute the mean of the loss
    return torch.mean(loss_to_return)


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