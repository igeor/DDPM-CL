import os
import torch
import argparse
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim

def list_of_strings(arg): return arg.split(',')
def list_of_ints(arg): return [int(x) for x in arg.split(',')]

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda", help="Device to use (e.g., 'cuda' or 'cpu')")
parser.add_argument("--dataset_name", default="~/.pytorch/MNIST_data/", help="Dataset name")
parser.add_argument("--labels", type=list_of_ints, default=[1,2,3], help="Labels to train on")
parser.add_argument("--image_size", type=int, default=32, help="Image size")
parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--test_batch_size", type=int, default=32, help="Batch size for evaluation")
parser.add_argument("--num_epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--save_model_epochs", type=int, default=1, help="Number of epochs to save model")
parser.add_argument("--output_dir", default="results/eval_classifier", help="Output directory")
parser.add_argument('--seed', type=int, default=0, help='Seed for generation')
args = parser.parse_args()

# Initialize the output evaluation directory
os.makedirs(args.output_dir, exist_ok=True)

# Define a transform 
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.image_size), antialias=None),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat single channel to get three channels
])

# Map the labels to the indices (i.e. labels[2,7] -> [0,1])
label_map = {label: i for i, label in enumerate(args.labels)}

# Download the MNIST dataset
full_trainset = datasets.MNIST(args.dataset_name, download=True, train=True, transform=preprocess)

# Filter the data to only include images with labels 2 and 7
indices = torch.tensor([label in args.labels for label in full_trainset.targets])
trainset = torch.utils.data.Subset(full_trainset, indices.nonzero().squeeze().tolist())
# Create a DataLoader for the filtered training data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)
print(f'Number of training images: {len(trainset)}')

# Repeat the process for the test data
full_testset = datasets.MNIST(args.dataset_name, download=True, train=False, transform=preprocess)
indices = torch.tensor([label in args.labels for label in full_testset.targets])
testset = torch.utils.data.Subset(full_testset, indices.nonzero().squeeze().tolist())
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
print(f'Number of test images: {len(testset)}')

# Define the pretrained model
model = models.resnet18(weights=True)
# Modify the last layer to have only 10 output features for MNIST
model.fc = nn.Linear(model.fc.in_features, len(args.labels))
# Set model to device
model.to(args.device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

for epoch in range(args.num_epochs):
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        # Map labels to indices
        labels = torch.tensor([label_map[label.item()] for label in labels]).to(args.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item()
    
    epoch_loss /= len(trainloader) * trainloader.batch_size
    print(f'Epoch {epoch + 1} loss: {epoch_loss:.3f}', end=' | ')

    if (epoch + 1) % args.save_model_epochs == 0:
        # Validation
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(args.device), data[1].to(args.device)
                # Map labels to indices
                labels = torch.tensor([label_map[label.item()] for label in labels]).to(args.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        
    print(f'Accuracy: {accuracy}')
    
    # Create model checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': epoch_loss,
        'mapping': label_map,
    }

    # Save checkpoint

    # Convert args.labels to string for filename
    labels_str = 'n'.join([str(label) for label in args.labels])
    torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_{labels_str}_ep{epoch + 1}.pth'))