import torch 
from torch import nn
from diffusers import UNet2DModel

torch.manual_seed(0)

class ClassConditionedUnet(nn.Module):
	def __init__(self, model, num_tasks=2):
		super().__init__()
		
		self.num_tasks = num_tasks
		self.class_emb_size = 1 

		# The embedding layer will map the class label to a vector of size class_emb_size
		self.class_emb = nn.Embedding(num_tasks, self.class_emb_size)

		# Self.model is an unconditional UNet with extra input channels to accept 
		# the conditioning information (the class embedding)
		self.model = model

	# Our forward method now takes the class labels as an additional argument
	def forward(self, x, t, task_labels):
		# Shape of x:
		bs, ch, w, h = x.shape
		
		# class conditioning in right shape to add as additional input channels
		# x is shape (bs, ch, w, h) and class_cond will be (bs, class_emb_size, w, h)
		class_emb = self.class_emb(task_labels) # Map to embedding dimension
		# repeat class_cond from (bs, 1) to (bs, num_tasks)
		class_cond = torch.randn(bs, self.num_tasks)
		# set the class_cond of the correct class to 1
		class_cond[:, task_labels] = class_emb
		# print(class_cond.shape)
		# convert from (bs, num_tasks) to (bs, class_emb_size, w, h)
		class_cond = class_cond.view(bs, self.num_tasks, 1, 1).repeat(1, 1, w, h)
		# print(class_cond.shape)
		# print(class_cond)
		# Net input is now x and class cond concatenated together along dimension 1
		net_input = torch.cat((x, class_cond), 1) # (bs, ch + class_emb_size, 28, 28)

		# Feed this to the UNet alongside the timestep and return the prediction
		return self.model(net_input, t).sample # (bs, ch, 28, 28)


if __name__ == "__main__":
		num_tasks = 2

		model = UNet2DModel(
				sample_size=32,  # the target image resolution
				in_channels=3 + num_tasks,  # the number of input channels, 3 for RGB images
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
				)
)

		x_in = torch.randn(1, 3, 32, 32)
		y_in = torch.Tensor([1]).type(torch.long)
		t_in = torch.Tensor([0]).type(torch.long) 
		model = ClassConditionedUnet(model, num_tasks=2)
		pred = model(x_in, t_in, y_in)