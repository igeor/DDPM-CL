from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from diffusers.schedulers import DDIMScheduler

from utils import sample_task_noise, get_label_vector


class TS_DDIMPipeline(DiffusionPipeline):
	r"""
	Pipeline for Task-Specific (TS) image generation.

	This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
	implemented for all pipelines (downloading, saving, running on a particular device, etc.).

	Parameters:
		unet ([`UNet2DModel`]):
			A `UNet2DModel` to denoise the encoded image latents.
		scheduler ([`SchedulerMixin`]):
			A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
			[`DDPMScheduler`], or [`DDIMScheduler`].
	"""
	model_cpu_offload_seq = "unet"

	def __init__(self, unet, scheduler, labels=[]):
		super().__init__()

		# make sure scheduler can always be converted to DDIM
		scheduler = DDIMScheduler.from_config(scheduler.config)
		self.labels = labels

		self.register_modules(unet=unet, scheduler=scheduler)

	@torch.no_grad()
	def __call__(
		self,
		batch_size: int = 1,
		generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
		eta: float = 0.0,
		num_inference_steps: int = 50,
		use_clipped_model_output: Optional[bool] = None,
		output_type: Optional[str] = "pil",
		return_dict: bool = True,
	) -> Union[ImagePipelineOutput, Tuple]:
		r"""
		The call function to the pipeline for generation.

		Args:
			batch_size (`int`, *optional*, defaults to 1):
				The number of images to generate.
			labels (`List[int]`, *optional*, defaults to []):
				The labels of the tasks to generate images for.
				Every label (task) corresponds to a different initial noise vector.
			generator (`torch.Generator`, *optional*):
				A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
				generation deterministic.
			eta (`float`, *optional*, defaults to 0.0):
				Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
				to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
				DDIM and `1` corresponds to DDPM.
			num_inference_steps (`int`, *optional*, defaults to 50):
				The number of denoising steps. More denoising steps usually lead to a higher quality image at the
				expense of slower inference.
			use_clipped_model_output (`bool`, *optional*, defaults to `None`):
				If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
				downstream to the scheduler (use `None` for schedulers which don't support this argument).
			output_type (`str`, *optional*, defaults to `"pil"`):
				The output format of the generated image. Choose between `PIL.Image` or `np.array`.
			return_dict (`bool`, *optional*, defaults to `True`):
				Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

		Example:

		```py
		>>> from diffusers import TS_DDIMPipeline
		>>> import PIL.Image
		>>> import numpy as np

		>>> # load model and scheduler
		>>> pipe = TS_DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

		>>> # run pipeline in inference (sample random noise and denoise)
		>>> image = pipe(eta=0.0, num_inference_steps=50)

		>>> # process image to PIL
		>>> image_processed = image.cpu().permute(0, 2, 3, 1)
		>>> image_processed = (image_processed + 1.0) * 127.5
		>>> image_processed = image_processed.numpy().astype(np.uint8)
		>>> image_pil = PIL.Image.fromarray(image_processed[0])

		>>> # save image
		>>> image_pil.save("test.png")
		```

		Returns:
			[`~pipelines.ImagePipelineOutput`] or `tuple`:
				If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
				returned where the first element is a list with the generated images
		"""

		# Sample gaussian noise to begin loop
		if isinstance(self.unet.config.sample_size, int):
			image_shape = (
				batch_size,
				self.unet.config.in_channels,
				self.unet.config.sample_size,
				self.unet.config.sample_size,
			)
		else:
			image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

		if isinstance(generator, list) and len(generator) != batch_size:
			raise ValueError(
				f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
				f" size of {batch_size}. Make sure the batch size matches the length of the generators."
			)

		# Sample gaussian noise to begin loop
		image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

		if self.labels != []:
			img_labels = get_label_vector(batch_size, labels=self.labels)
			image = sample_task_noise(
				noise_shape=image_shape,
				labels=img_labels,
				generator=generator,
				device=self._execution_device,
				dtype=self.unet.dtype
			)

		# set step values
		self.scheduler.set_timesteps(num_inference_steps)

		for t in self.progress_bar(self.scheduler.timesteps):
			# 1. predict noise model_output
			model_output = self.unet(image, t).sample

			# 2. predict previous mean of image x_t-1 and add variance depending on eta
			# eta corresponds to η in paper and should be between [0, 1]
			# do x_t -> x_t-1
			image = self.scheduler.step(
				model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
			).prev_sample

		image = (image / 2 + 0.5).clamp(0, 1)
		image = image.cpu().permute(0, 2, 3, 1).numpy()
		if output_type == "pil":
			image = self.numpy_to_pil(image)

		if not return_dict:
			return (image,)

		return ImagePipelineOutput(images=image) 