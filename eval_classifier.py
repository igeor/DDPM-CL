import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets import load_dataset
from config import TrainingConfig

config = TrainingConfig()

train_dataset = load_dataset(config.dataset_name, split="train")
test_dataset = load_dataset(config.dataset_name, split="test")

preprocess = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]), # Convert images from (0,1) to (-1, 1)
])

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images, "labels": examples["label"]}

train_dataset.set_transform(transform)
test_dataset.set_transform(transform)

if config.labels is not None:
    train_dataset = train_dataset.filter(lambda example: example["labels"] in config.labels)
    test_dataset = test_dataset.filter(lambda example: example["labels"] in config.labels)

train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

# Define the model
model = models.resnet50(pretrained=True)
# Change the first and last layer
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, 10, bias=True)
model = model.to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

display_step = 100
total = 0 
correct = 0
for epoch in range(10):
    for i, data in enumerate(train_dataloader):
        # Get the inputs
        images, labels = data['images'], data['labels']
        # Send them to device
        images = images.to(config.device)
        labels = labels.to(config.device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize  
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % display_step == 0:
            print('Epoch: {} Batch: {} loss: {}'.format(epoch, i, loss.item()))

        # Compute training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Training accuracy: {} %'.format((correct / total) * display_step))

# save the model
torch.save(model.state_dict(), 'mnist-classifier.pth')

# Test the model
model.eval()
total = 0
correct = 0
y_preds = []
for i, data in enumerate(test_dataloader, 1):
    # Get the inputs
    images, labels = data['images'], data['labels']
    # Send them to device
    images = images.to(config.device)
    labels = labels.to(config.device)

    outputs = model(images)
    y_preds += [ outputs ]

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Testing accuracy: {} %'.format((correct/total)*100))