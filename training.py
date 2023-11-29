import os
import torch
import torch.nn.functional as F

from diffusers.utils import make_image_grid
from diffusers import UNet2DModel, DDIMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from tqdm.auto import tqdm

from dataset import init_dataset
from utils import get_preprocess_function
from args import parse_args

""" 
Initialize the training script
"""
args = parse_args()

# Initialize the output directory
os.makedirs(args.output_dir, exist_ok=True)

# Store args in the output directory
if args.output_dir is not None:
    torch.save(args, os.path.join(args.output_dir, "args.pt"))

# Initialize the directory to save the generated images during training
eval_dir = os.path.join(args.output_dir, "samples")
os.makedirs(eval_dir, exist_ok=True)

# Initialize the preprocessing function
preprocess = get_preprocess_function(
    args.dataset_name, flip=args.pr_flip, rotate=args.pr_rotate)

# Get the training and test datasets
trainset, testset = init_dataset(
    args.dataset_name, dataset_path=args.dataset_path, target_dir=args.target_dir,
    labels=args.labels, preprocess=preprocess)

# Initialize the training dataloader
train_dataloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch_size, shuffle=True)

# Define the UNet model
model = UNet2DModel(
    sample_size=args.image_size,  # the target image resolution
    in_channels=args.in_channels,  # the number of input channels, 3 for RGB images
    out_channels=args.out_channels,  # the number of output channels
    layers_per_block=args.layers_per_block,  # how many ResNet layers to use per UNet block
    block_out_channels=args.block_out_channels,  # the number of output channels for each UNet block
    down_block_types=args.down_block_types,
    up_block_types=args.up_block_types
)

# Load pretrained model if args.pretrained_model_dir is not None
if args.pretrained_model_dir is not None:
    model = UNet2DModel.from_pretrained(
        args.pretrained_model_dir, subfolder="unet", use_safetensors=True)
    
# Move the model to the device
model = model.to(args.device)

# Define the optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# Define the learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)

# Define the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=args.num_train_timesteps,
    beta_start=args.beta_start,
    beta_end=args.beta_end,
    beta_schedule=args.beta_schedule
)

# Load pretrained noise scheduler if args.pretrained_model_dir is not None
if args.pretrained_model_dir is not None:
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_dir, subfolder="scheduler", use_safetensors=True)


""" 
The training loop
"""
# Initialize tqdm progress bar
progress_bar = tqdm(range(args.num_epochs), desc="Epoch 1 | loss: 0.000 | lr: 0.000e+00")
# Initialize the train log
train_log = {"loss": [], "lr": [], "step": []}

global_step = 0

for epoch in range(args.num_epochs):

    # Iterate over the training batches
    for batch_idx, (clean_images, clean_images_labels) in enumerate(train_dataloader):
        # Move the images to the device
        clean_images = clean_images.to(args.device)

        # Get the batch size
        bs = clean_images.shape[0]

        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(args.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=args.device
        ).long()

        # Forward Diffusion Process
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Backward Diffusion Process
        # Predict the noise residual
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        
        # Compute the MSE loss between the predicted noise and the actual noise
        loss = F.mse_loss(noise, noise_pred)

        # Backpropagate the loss
        loss.backward()
        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update the model parameters
        optimizer.step()
        # Update the learning rate
        lr_scheduler.step()
        # Zero the gradients
        optimizer.zero_grad()

        # Log the loss and learning rate
        train_log['loss'] += [loss.item()]
        train_log['lr'] += [lr_scheduler.get_last_lr()[0]]
        train_log['step'] += [global_step]
        global_step += 1
    
        # Update description of the progress bar
        progress_bar.set_description(f"Epoch {epoch + 1} | loss: {loss.item():.3f} | lr: {lr_scheduler.get_last_lr()[0]:.3e}")

    # Update progress bar
    progress_bar.update(1)

     # Update the train log file
    torch.save(train_log, os.path.join(args.output_dir, "train_log.pt"))

    # Evaluate the model 
    if (epoch + 1) % args.sample_image_epochs == 0:
        
        pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)

        # Set progress to false
        pipeline.set_progress_bar_config(disable=True)
        
        with torch.no_grad():
            # Generate images
            images = pipeline(
                batch_size=args.eval_batch_size,
                generator=torch.manual_seed(args.gen_seed),
                num_inference_steps=args.num_inference_steps,
            ).images
    
        # Save the images
        image_grid = make_image_grid(images[:16], rows=4, cols=4)
        image_grid.save(f"{eval_dir}/{epoch + 1}.png")

    # Save the model
    if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
        pipeline.save_pretrained(f'{args.output_dir}/epoch-{epoch + 1}')
    
    # Generate samples from the model
    if (epoch + 1) % args.generate_image_epochs == 0 or epoch == args.num_epochs - 1:

        # Initialize output folder
        output_dir = f"{args.output_dir}/epoch-{epoch + 1}/{args.pipeline}_fake_images"
        # Create output folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # List the files in the output folder
        num_exist_images = len(os.listdir(output_dir))

        # Set progress to false
        pipeline.set_progress_bar_config(disable=True)

        # Initialize tqdm bar
        pbar = tqdm(total=args.n_fake_images, desc=f"Generating fake images for epoch {epoch + 1}...")
        # Generate images in batches
        for b_idx in range(0, args.n_fake_images, args.eval_batch_size):
            # (Inverse Diffusion Process) Sample fake images from random noise
            # The default pipeline output type is `List[PIL.Image]`
            with torch.no_grad():
                images = pipeline(
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.num_inference_steps
                ).images

            # Save the images to the output_dir
            for image_idx, image in enumerate(images):
                image.save(f"{output_dir}/{num_exist_images + b_idx + image_idx}.png")

            # Update tqdm bar
            pbar.update(args.eval_batch_size)

    
# See results:
# tensorboard --logdir path/to/your/logs (i.e. results/unetSM/mnist/logs)