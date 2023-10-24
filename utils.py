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


def img_to_rgb(batch: torch.Tensor, img_type: str = 'uint8', img_range: tuple = (0, 255)):
    """
    Convert a batch of tensor images to the specified data type and pixel scale.

    Args:
        batch (torch.Tensor): Batch of tensor images.
        img_type (str): Target data type for the output image, e.g., 'uint8'.
        img_range (tuple): Range of pixel values for scaling the output, e.g., (0, 255).

    Returns:
        torch.Tensor: Batch of converted images.
    """
    # Ensure the input tensor is in float32 format for scaling
    batch = batch.float()

    # Scale the pixel values based on the provided img_range
    min_val, max_val = img_range
    batch = (batch - batch.min()) / (batch.max() - batch.min())  # Scale to [0, 1]
    batch = batch * (max_val - min_val) + min_val  # Scale to the specified range

    # Convert the tensor to the specified data type
    if img_type == 'uint8':
        batch = batch.clamp(0, 255).byte()
    elif img_type == 'float32':
        pass  # No further conversion needed for float32
    else:
        raise ValueError("Unsupported img_type. Supported types are 'uint8' and 'float32'.")

    return batch

# Example usage:
# Assuming you have a batch of images in tensor format, you can use the function like this:
# converted_images = img_to_rgb(input_batch, img_type='uint8', img_range=(0, 255))

    

def fid_measure(real_images, fake_images):
    fid = FrechetInceptionDistance(feature=64)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()

def gather(consts: torch.Tensor, t: torch.Tensor):
      c = consts.gather(-1, t)
      return c.reshape(-1, 1, 1, 1)

def entropy(y_pred):
    y_pred = y_pred.sum(dim=0)
    y_pred /= y_pred.sum(dim=0)
    h = torch.sum( -y_pred * torch.log2(y_pred) )
    return h