'''
    Loads a saved model and uses it in 2 different modes:
    - Displays NxN random images
    - Displays N images with their reconstruction from the dataset.
'''

import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
print(torch.__version__)
print(torchvision.__version__)

from vaegan import VAEGAN

# Parse arguments
parser = argparse.ArgumentParser(description='VAEGAN playing MNIST')
parser.add_argument('--weights', type=str, default='weights/base.torch', help='Weight file to load.')
parser.add_argument('--mode', type=str, default='generator', choices=['generator', 'reconstructor'], help='Mode for the script')
parser.add_argument('--N', type=int, default=4, help='Images to display, depends on mode.')
args = parser.parse_args()

# Find the available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
state = torch.load(args.weights, map_location='cpu')
input_shape = (3, 32, 32)
vaegan = VAEGAN(input_shape, latent_size=state['latent_size'], convs=state['convs']).to(device)
vaegan.load_state_dict(state['state_dict'])
vaegan.eval()

if args.mode == 'generator':
    # Create an NxN vector of latent_size
    x = np.random.randn(args.N, args.N, state['latent_size']).astype(np.float32)
    # Decode using the VAE decoder
    generated = vaegan.decode(torch.from_numpy(np.reshape(x, (-1, state['latent_size']))))
    # Reshape to display [has shape (batch, 3, 32, 32)]
    generated = torch.permute(0, 2, 3, 1) # shape: (batch, 32, 32, 3)
    generated = np.reshape(generated.detach().numpy(), (args.N, args.N, 32, 32, 3))
    # Create a window with NxN subplots
    fig, ax = plt.subplots(args.N, args.N)
    for i in range(args.N):
        for k in range(args.N):
            ax[i, k].imshow(generated[i, k])
    plt.show()
elif args.mode == 'reconstructor':
    # Get N images from MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.N, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # Get the reconstructed images
    mu, log_sigma, z, rebuild = vaegan(images)
    # Transform for numpy
    ground = np.reshape(images.permute(0, 2, 3, 1).detach().numpy(), (args.N, 32, 32, 3))
    rebuild = np.reshape(rebuild.permute(0, 2, 3, 1).detach().numpy(), (args.N, 32, 32, 3))
    # Plot
    fig, ax = plt.subplots(args.N, 2)
    for i in range(args.N):
        ax[i, 0].imshow(ground[i])
        ax[i, 1].imshow(rebuild[i])
    plt.show()
else:
    raise Exception('Unrecognized mode.')
