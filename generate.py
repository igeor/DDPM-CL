import os
import torch
from tqdm import tqdm
import argparse
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline, DDIMPipeline
from pipelines import DDIMPipeline as MDDIMPipeline

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda:0", help="Device to use (e.g., 'cuda' or 'cpu')")
parser.add_argument("--pipeline", default="mddim", help="The pipeline type, 'ddpm', 'ddim' or 'mddim'")
parser.add_argument("--num_inference_steps", type=int, default=50, help="The pipeline type, 'ddpm' or 'ddim'")
parser.add_argument("--pretrained_model_dir", default="/datatmp/users/igeorvasilis/ddpm-continual-learning/results/cifar10/mask_v1/m0/epoch-200", help="Path to the pretrained model directory")
parser.add_argument("--n_images_to_generate", type=int, default=10_000, help="Number of images to generate")
parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation")
parser.add_argument("--show_gen_progress", action="store_true", help="Show generation progress")
parser.add_argument("--folder_name", default="mddim_fake_images_v2", help="Output folder name")
parser.add_argument("--num_labels", type=int, default=1, help="Output folder name")
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
elif args.pipeline == "mddim":
    pipeline = MDDIMPipeline(unet=model, scheduler=noise_scheduler, num_tasks=args.num_labels)

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
        image.save(f"{output_dir}/{num_exist_images+ b_idx + image_idx}.png")

    # Update tqdm bar
    pbar.update(args.eval_batch_size)