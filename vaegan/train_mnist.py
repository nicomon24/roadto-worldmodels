'''
    Train a VAEGAN model on the MNIST dataset.
'''

from tqdm import trange, tqdm
import numpy as np
import argparse, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
print(torch.__version__)
print(torchvision.__version__)

from tensorboardX import SummaryWriter

from vaegan import VAEGAN

# Parse arguments
parser = argparse.ArgumentParser(description='VAEGAN learning MNIST')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
parser.add_argument('--latent_size', type=int, default=32, help='Latent vector size.')
parser.add_argument('--epochs', type=int, default=10, help='Training epochs.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. (default: 0.001)')
parser.add_argument('--logdir', type=str, default='logs', help='Logging directory.')
parser.add_argument('--name', type=str, default='base', help='Name for the run.')
parser.add_argument('--savedir', type=str, default='weights', help='Directory to use for saving the model.')
args = parser.parse_args()

# Find the available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    # Space for other transformations
    transforms.ToTensor() # We need this to get a tensor instead of a PIL image
])

# Datasets
trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

# Loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Create VAEGAN
vaegan = VAEGAN(latent_size=args.latent_size).to(device)
# Optimizer
optimizer = optim.SGD(vaegan.parameters(), lr=args.lr)

# Tensorboard writer
writer = SummaryWriter(args.logdir + '/' + args.name)

global_i = 0
for epoch in trange(args.epochs):  # loop over the dataset multiple times

    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainset)//args.batch_size):
        # Get the inputs
        inputs, labels = data
        # Optimize and compute losses
        reco_loss, norm_loss, total_loss = vaegan.optimize(inputs.to(device), optimizer)
        # Log
        global_i += 1
        writer.add_scalar('data/reco_loss', reco_loss, global_i)
        writer.add_scalar('data/norm_loss', norm_loss, global_i)
        writer.add_scalar('data/total_loss', total_loss, global_i)

# Save the model for inference
state = {
    'state_dict': vaegan.state_dict(),
    'latent_size': vaegan.latent_size
}
# Create savedir if necessary
if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)
torch.save(state, args.savedir + '/' + args.name + '.torch')
print("Saved final model, closing...")
