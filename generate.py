import os
import torch
from tqdm import tqdm
import argparse
from diffusers import DDPMScheduler, DDIMPipeline, DDPMPipeline, UNet2DModel

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda", help="Device to use (e.g., 'cuda' or 'cpu')")
parser.add_argument("--pipeline", default="ddim", help="The pipeline type, 'ddpm' or 'ddim'")
parser.add_argument("--num_inference_steps", type=int, default=100, help="The pipeline type, 'ddpm' or 'ddim'")
parser.add_argument("--pretrained_model_dir", default="./results/unetXL/mnist", help="Path to the pretrained model directory")
parser.add_argument("--n_images_to_generate", type=int, default=12000, help="Number of images to generate")
parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation")
parser.add_argument("--show_gen_progress", action="store_true", help="Show generation progress")
parser.add_argument("--folder_name", default="ddim_fake_images", help="Output folder name")
args = parser.parse_args()

# Load the UNet model
model = UNet2DModel.from_pretrained(os.path.join(args.pretrained_model_dir, "unet")).to(args.device)

# Load the noise scheduler
noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(args.pretrained_model_dir, "scheduler"))

print(f"Model: {args.pretrained_model_dir} loaded successfully!")

# Load the Diffusion pipeline
if args.pipeline == "ddpm":
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    args.num_inference_steps = 1000
elif args.pipeline == "ddim":
    pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)
# Hide the progress bar if show_gen_progress is False
pipeline.set_progress_bar_config(disable=not args.show_gen_progress)

# Initialize output folder
output_dir = os.path.join(args.pretrained_model_dir, args.folder_name)
# Create output folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize tqdm bar
pbar = tqdm(total=args.n_images_to_generate, desc="Generating...")
# Generate images in batches
for b_idx in range(0, args.n_images_to_generate, args.eval_batch_size):
    # (Inverse Diffusion Process) Sample fake images from random noise
    # The default pipeline output type is `List[PIL.Image]`
    with torch.no_grad():
        images = pipeline(
            batch_size=args.eval_batch_size,
            num_inference_steps=args.num_inference_steps
        ).images

    # Save the images to the output_dir
    for image_idx, image in enumerate(images):
        image.save(f"{output_dir}/{b_idx + image_idx}.png")

    # Update tqdm bar
    pbar.update(args.eval_batch_size)