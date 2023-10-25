from dataclasses import dataclass

@dataclass
class TrainingConfig:
    device = "cuda"
    dataset_name = "mnist"
    pretrained_model_dir = None 
    labels = [1, 2]
    image_size = 32  # the generated image resolution
    train_batch_size = 128
    eval_batch_size = 128  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 5
    save_image_epochs = 10
    save_model_epochs = 10
    show_gen_progress = False
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results/mnist-1n2"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

@dataclass
class GenerateConfig:
    device = "cuda"
    pretrained_model_dir = "results/mnist-1n2/epoch-99"
    n_images_to_generate = 4_000
    eval_batch_size = 128  # how many images to sample during evaluation
    show_gen_progress = False
    output_dir = "results/mnist-1n2/epoch-99"  # the model name locally and on the HF Hub
    folder_name = "fake_images"

class EvalConfig:
    device = "cuda"
    dataset_name = "mnist"
    pretrained_model_dir = "results/mnist-1n2-pr1/epoch-99"
    labels = [1, 2]
    image_size = 32  # the generated image resolution
    n_images_to_evaluate = 4_000
    eval_batch_size = 128  # how many images to sample during evaluation
    folder_name = "fake_images"