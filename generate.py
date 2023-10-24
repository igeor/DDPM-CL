import os
from tqdm import tqdm
from config import GenerateConfig
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DModel

# Initialize the config
config = GenerateConfig()

# Load the UNet model 
model = UNet2DModel.from_pretrained(f'./{config.pretrained_model_dir}', subfolder="unet").to(config.device)

# Load the noise scheduler
noise_scheduler = DDPMScheduler.from_pretrained(f'./{config.pretrained_model_dir}', subfolder='scheduler')

# Load the DDPM pipeline
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler, show_progress=False)

# Create output folder if it doesn't exist
output_dir = os.path.join(config.output_dir, config.folder_name)
if not os.path.exists(output_dir): os.makedirs(output_dir)

# Initialize tqdm bar
pbar = tqdm(total=config.n_images_to_generate, desc="Generating...")

# Generate images in batches
for b_idx in range(0, config.n_images_to_generate, config.eval_batch_size):
    # (Inverse Diffusion Process) Sample fake images from random noise 
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(batch_size=config.eval_batch_size).images

    # Save the images to the output_dir
    for image_idx, image in enumerate(images):
        image.save(f"{config.output_dir}/{config.folder_name}/{b_idx + image_idx}.png")

    # Update tqdm bar
    pbar.update(config.eval_batch_size)

        