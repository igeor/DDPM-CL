from dataclasses import dataclass

@dataclass
class TrainingConfig:
    device = "cuda"
    dataset_name = "mnist"
    pipeline = "ddim"  # the pipeline type, `ddpm` or `ddim`
    pretrained_model_dir = "results/unetSM/mnist-2/epoch-99"
    labels = [2] 
    image_size = 32  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 32  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 5
    save_image_epochs = 50
    save_model_epochs = 50
    show_gen_progress = False
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results/unetSM/mnist-2-pr1"  # the model name locally and on the HF Hub
    gen_seed = 0 

@dataclass
class GenerateConfig:
    device = "cuda"
    pipeline = "ddim"  # the pipeline type, `ddpm` or `ddim`
    pretrained_model_dir = "results/unetSM/mnist-1n2-pr1/epoch-49"
    n_images_to_generate = 12_000
    eval_batch_size = 128  # how many images to sample during evaluation
    show_gen_progress = False
    folder_name = "ddim_fake_images"

class EvalConfig:
    device = "cuda"
    dataset_name = "mnist"
    pretrained_model_dir = "results/unetXL/mnist-1n2/epoch-49"
    labels = [1, 2]
    image_size = 32  # the generated image resolution
    n_images_to_evaluate = 4_000
    eval_batch_size = 32  # how many images to sample during evaluation
    folder_name = "ddim_fake_images"
    seed = 0