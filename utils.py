import torch 
from torchvision import transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance

def batch_to_rgb(batch_gray: torch.Tensor):
    batch_rgb = batch_gray.repeat(1, 3, 1, 1)  # Duplicate the grayscale channel to create RGB images
    batch_rgb *= 255  # Scale the images to [0, 255]
    batch_rgb = batch_rgb.to(torch.uint8) # Convert to 8-bit integers
    return batch_rgb 

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