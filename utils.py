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