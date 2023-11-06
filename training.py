import os
import torch
import argparse
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator, notebook_launcher
from diffusers.utils import make_image_grid
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset import init_dataset, init_v2_dataset

def list_of_strings(arg): return arg.split(',')
def list_of_ints(arg): return [int(x) for x in arg.split(',')]

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda", help="Device to use (e.g., 'cuda' or 'cpu')")
parser.add_argument("--dataset_name", default="~/.pytorch/MNIST_data/", help="Dataset name")
parser.add_argument("--pipeline", default="ddim", help="The pipeline type, 'ddpm' or 'ddim'")
parser.add_argument("--pretrained_model_dir", default="No", help="Path to the pretrained model directory")
parser.add_argument("--labels", type=list_of_ints, default=[2], help="Labels to train on")
parser.add_argument("--image_size", type=int, default=32, help="Image size")
parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--lr_warmup_steps", type=int, default=5, help="Number of learning rate warmup steps")
parser.add_argument("--save_image_epochs", type=int, default=50, help="Number of epochs to save images")
parser.add_argument("--save_model_epochs", type=int, default=50, help="Number of epochs to save model")
parser.add_argument("--mixed_precision", default="fp16", help="Mixed precision mode, 'no' for float32, 'fp16' for automatic mixed precision")
parser.add_argument("--output_dir", default="experiment", help="Output directory")
parser.add_argument("--show_gen_progress", action="store_true", help="Show generation progress")
parser.add_argument("--train_on", default="train", help="In which dataset to train on (e.g., 'train' or 'test')")
parser.add_argument('--gen_seed', type=int, default=0, help='Seed for generation')

args = parser.parse_args()

# Initialize the output evaluation directory
eval_dir = os.path.join(args.output_dir, "samples")
os.makedirs(eval_dir, exist_ok=True)

# Initialize the training dataloader
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat single channel to get three channels
    transforms.Resize((32), antialias=None),
    transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
])

train_dataloader = torch.utils.data.DataLoader(
    init_v2_dataset(args.dataset_name, split=args.train_on, preprocess=preprocess, labels=args.labels), 
    batch_size=args.train_batch_size, shuffle=True)

# Define the UNet model
model = UNet2DModel(
    sample_size=args.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 256, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "UpBlock2D",
        "UpBlock2D"
    ),
)

# Load pretrained model if args.pretrained_model_dir is not None
if args.pretrained_model_dir != "No":
    model = UNet2DModel.from_pretrained(
        args.pretrained_model_dir, subfolder="unet", use_safetensors=True)

# Move the model to the device
model = model.to(args.device)

# Define the noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000) # Linear by default

# Load pretrained noise scheduler if args.pretrained_model_dir is not None
if args.pretrained_model_dir != "No":
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_dir, subfolder="scheduler", use_safetensors=True)

# Define the optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# Define the learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * args.num_epochs),
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
        accelerator.init_trackers("train_log")

    # Prepare the model, optimizer, and dataloader
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Initialize the global step
    global_step = 0

    # Initialize tqdm progress bar
    progress_bar = tqdm(range(config.num_epochs), desc="Epoch 1 | loss: 0.000 | lr: 0.000e+00")

    # Now you train the model
    for epoch in range(config.num_epochs):

        for step, (clean_images, _) in enumerate(train_dataloader):

            # Sample gaussian noise 
            noise = torch.randn(clean_images.shape).to(accelerator.device)

            # Get the batch size
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=accelerator.device
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
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(logs, step=global_step)
            
            # Update the global step
            global_step += 1
        
            # Update description of the progress bar
            progress_bar.set_description(f"Epoch {epoch + 1} | loss: {loss.item():.3f} | lr: {lr_scheduler.get_last_lr()[0]:.3e}")

        # Update progress bar
        progress_bar.update(1)

        # Sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:

            # Evaluate the model 
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                
                # Initialize the pipeline
                pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                if config.pipeline == "ddpm":
                    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                # Set progress false
                pipeline.set_progress_bar_config(disable=True)
                
                # Generate sample images
                images = pipeline(
                    batch_size=config.eval_batch_size,
                    generator=torch.manual_seed(config.gen_seed),
                ).images
                
                image_grid = make_image_grid(images[:16], rows=4, cols=4)
                image_grid.save(f"{eval_dir}/{epoch + 1}.png")

            # Save the model
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(f'{config.output_dir}/epoch-{epoch + 1}')


# Create a dataclass config 
config = argparse.Namespace(
    device=args.device,
    dataset_name=args.dataset_name,
    pipeline=args.pipeline,
    pretrained_model_dir=args.pretrained_model_dir,
    labels=args.labels,
    image_size=args.image_size,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    num_epochs=args.num_epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    lr_warmup_steps=args.lr_warmup_steps,
    save_image_epochs=args.save_image_epochs,
    save_model_epochs=args.save_model_epochs,
    mixed_precision=args.mixed_precision,
    output_dir=args.output_dir,
    gen_seed=args.gen_seed,
)

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

# See results:
# tensorboard --logdir path/to/your/logs (i.e. results/unetSM/mnist/logs)