import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from dataset import init_dataset
from config import TrainingConfig

# Initialize the config
config = TrainingConfig()

# Save the images
eval_dir = os.path.join(config.output_dir, "samples")
os.makedirs(eval_dir, exist_ok=True)

train_dataset = init_dataset(config.dataset_name, split='train', labels=config.labels)
print(f"Number of training examples: {len(train_dataset)}")

# Initialize the training dataloader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.train_batch_size, shuffle=True)

# Define the UNet model
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with an attention layer
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet downsampling block with an attention layer
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# Load pretrained model if config.pretrained_model_dir is not None
if config.pretrained_model_dir is not None:
    model = UNet2DModel.from_pretrained(
        config.pretrained_model_dir, subfolder="unet", use_safetensors=True)

# Move the model to the device
model = model.to(config.device)

# Define the noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000) # Linear by default

# Load pretrained noise scheduler if config.pretrained_model_dir is not None
if config.pretrained_model_dir is not None:
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model_dir, subfolder="scheduler", use_safetensors=True)

# Define the optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Define the learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


# Define the training loop
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    # Initialize the repository
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare the model, optimizer, and dataloader
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Initialize the global step
    global_step = 0

    # Initialize tqdm progress bar
    progress_bar = tqdm(range(config.num_epochs), desc="Epoch 0 | loss: 0.000 | lr: 0.000e+00")

    # Now you train the model
    for epoch in range(config.num_epochs):

        for step, batch in enumerate(train_dataloader):
            # Get the clean images
            clean_images = batch["image"]

            # Sample gaussian noise 
            noise = torch.randn(clean_images.shape).to(clean_images.device)

            # Get the batch size
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Forward Diffusion Process
            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Backward Diffusion Process
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                # Compute the MSE loss
                loss = F.mse_loss(noise_pred, noise)
                # Backpropagate the loss
                accelerator.backward(loss)
                # Clip the gradients
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                # Update the model parameters
                optimizer.step()
                # Update the learning rate
                lr_scheduler.step()
                # Zero the gradients
                optimizer.zero_grad()

            # Log the loss and learning rate
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            # accelerator.log(logs, step=global_step)
            
            # Update the global step
            global_step += 1
        
            # Update description of the progress bar
            progress_bar.set_description(f"Epoch {epoch} | loss: {loss.item():.3f} | lr: {lr_scheduler.get_last_lr()[0]:.3e}")

        # Update progress bar
        progress_bar.update(1)

        # Sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:

            # Initialize the pipeline
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), 
                scheduler=noise_scheduler, 
                show_progress=config.show_gen_progress
            )

            # Evaluate the model 
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # Generate sample images
                images = pipeline(
                    batch_size=config.eval_batch_size,
                    generator=torch.manual_seed(config.seed),
                ).images
                
                image_grid = make_image_grid(images[:16], rows=4, cols=4)
                image_grid.save(f"{eval_dir}/{epoch}.png")

            # Save the model
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(f'{config.output_dir}/epoch-{epoch}')

from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)


