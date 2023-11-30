import os
import torch
from tqdm import tqdm
import argparse
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline, DDIMPipeline
from pipelines import TS_DDIMPipeline
from args import parse_gen_args

args = parse_gen_args()

# Load the UNet model
model = UNet2DModel.from_pretrained(os.path.join(args.pretrained_model_dir, "unet")).to(args.device)

# Load the noise scheduler
noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(args.pretrained_model_dir, "scheduler"))

print(f"Model: {args.pretrained_model_dir} loaded successfully!")

# Load the Diffusion pipeline
if args.pipeline == "ddim":
    pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)
elif args.pipeline == "ts_ddim":
    pipeline = TS_DDIMPipeline(unet=model, scheduler=noise_scheduler, labels=args.labels)

# Hide the progress bar if show_gen_progress is False
pipeline.set_progress_bar_config(disable=not args.show_gen_progress)

# Initialize output folder
output_dir = os.path.join(args.pretrained_model_dir, args.folder_name)
# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# List the files in the output folder
num_exist_images = len(os.listdir(output_dir))

# Initialize tqdm bar
pbar = tqdm(total=args.n_images_to_generate, desc="Generating...")
# Generate images in batches
for b_idx in range(0, args.n_images_to_generate, args.batch_size):
    # (Inverse Diffusion Process) Sample fake images from random noise
    # The default pipeline output type is `List[PIL.Image]`
    with torch.no_grad():
        images = pipeline(
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps
        ).images

    # Save the images to the output_dir
    for image_idx, image in enumerate(images):
        image.save(f"{output_dir}/{num_exist_images+ b_idx + image_idx}.png")

    # Update tqdm bar
    pbar.update(args.batch_size)