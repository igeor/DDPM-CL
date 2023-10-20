import os
import torch
from dataclasses import dataclass
from torchvision import transforms
from diffusers import DDPMScheduler
from accelerate import Accelerator
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid

@dataclass
class TrainingConfig:
    device = "cuda"
    dataset_name = "mnist"
    labels = [2]
    image_size = 32  # the generated image resolution
    train_batch_size = 128
    eval_batch_size = 128  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-mnist2s-32"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

""" *** DATASETS *** """
from datasets import load_dataset

test_dataset = load_dataset(config.dataset_name, split="test")

preprocess = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    # transforms.RandomHorizontalFlip(), # Remove Horizontal Flip (MNIST is symmetric)
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
])

# convert image back to (0, 1) range
def postprocess(images_tensor): 
    return (images_tensor + 1) / 2

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"image": images, "label": examples["label"]}

def filter_dataset(dataset, labels):
    return dataset.filter(lambda example: example["label"] in labels)

test_dataset.set_transform(transform)

if config.labels is not None:
    test_dataset = filter_dataset(test_dataset, config.labels)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config.eval_batch_size, shuffle=False)


""" *** Diffusion Model *** """
from diffusers import UNet2DModel
model = UNet2DModel.from_pretrained(f'./{config.output_dir}', subfolder="unet", use_safetensors=True)
model = model.to(config.device)

noise_scheduler = DDPMScheduler.from_pretrained(f'./{config.output_dir}', subfolder='scheduler', use_safetensors=True)

""" *** Evaluation *** """
def evaluate(config, pipeline, folder='fake_images'):
    # Create output folder if it doesn't exist
    if not os.path.exists(os.path.join(config.output_dir, folder)):
        os.makedirs(os.path.join(config.output_dir, folder))

    # Sample fake images from random noise 
    # Inverse Diffusion Process
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Save the images to the output_dir
    output_dir = os.path.join(config.output_dir, folder)
    # count the images in the folder
    n_images = len(os.listdir(output_dir))
    for image_idx, image in enumerate(images):
        image.save(f"{config.output_dir}/{folder}/{n_images + image_idx}.png")

    # Return the number of images
    return len(images)


# Initialize accelerator and tensorboard logging
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    project_dir=os.path.join(config.output_dir, "logs"),
)

pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

# Generate fake images
n_fake_images = 0
while n_fake_images < len(test_dataset):
    n_images = evaluate(config, pipeline, folder='fake_images')
    # Update the number of fake images
    n_fake_images += n_images