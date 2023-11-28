import argparse

def list_of_strings(arg): return arg.split(',')
def list_of_ints(arg): return [int(x) for x in arg.split(',')]

def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1", help="Device to use (e.g., 'cuda' or 'cpu')")

    parser.add_argument("--dataset_name", default='MNIST', help="Dataset name")
    parser.add_argument("--dataset_path", default=None, help="Dataset path")
    parser.add_argument("--target_dir", default='./dataset', help="The target directory to download the dataset")
    parser.add_argument("--pr_flip", action="store_true", help="Apply random horizontal flip to training set")
    parser.add_argument("--pr_rotate", action="store_true", help="Apply random rotation to training set")
    parser.add_argument("--labels", type=list_of_ints, default=[1], help="Labels to train on")

    parser.add_argument("--num_train_timesteps", type=int, default=1_000, help="Number of training timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Number of training timesteps")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Number of training timesteps")
    parser.add_argument("--beta_schedule", default="squaredcos_cap_v2", help="Number of training timesteps")
    parser.add_argument("--mask", type=str, default=None, help="The type of the mask")
    parser.add_argument("--num_tasks", type=int, default=None, help="The number of the tasks")

    parser.add_argument("--pipeline", default="ddim", help="The pipeline type, 'ddpm' or 'ddim'")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="The pipeline type, 'ddpm' or 'ddim'")

    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    parser.add_argument("--in_channels", type=int, default=3, help="The number of input channels, 3 for RGB images")
    parser.add_argument("--out_channels", type=int, default=3, help="The number of output channels")
    parser.add_argument("--layers_per_block", type=int, default=2, help="How many ResNet layers to use per UNet block")
    parser.add_argument("--block_out_channels", type=list_of_ints, 
                        default=[128, 128, 256, 256, 512, 512], help="The number of output channels for each UNet block")
    parser.add_argument("--down_block_types", type=list_of_strings, 
                        default=["DownBlock2D",  "DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"], 
                        help="The type of the downsampling block")
    parser.add_argument("--up_block_types", type=list_of_strings,
                        default=["UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
                        help="The type of the upsampling block")

    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=5, help="Number of learning rate warmup steps")

    parser.add_argument("--pretrained_model_dir", default=None, help="Path to the pretrained model directory")
    parser.add_argument("--sample_image_epochs", type=int, default=1, help="Number of epochs to save images")
    parser.add_argument("--generate_image_epochs", type=int, default=100, help="Number of epochs to generate images")
    parser.add_argument("--n_fake_images", type=int, default=10_000, help="Number of fake images to generate")
    parser.add_argument("--save_model_epochs", type=int, default=50, help="Number of epochs to save model")
    parser.add_argument("--output_dir", default="experiment", help="Output directory")

    parser.add_argument("--mixed_precision", default="fp16", help="Mixed precision mode, 'no' for float32, 'fp16' for automatic mixed precision")
    parser.add_argument("--show_gen_progress", action="store_true", help="Show generation progress")
    parser.add_argument('--gen_seed', type=int, default=0, help='Seed for generation')

    args = parser.parse_args()
    return args