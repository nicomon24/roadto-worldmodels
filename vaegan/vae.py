'''
    Define a class for the VAE/GAN model.
'''

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
    Create the VAE model. The architecture is specified using the 'convs'
    parameter, an array of tuple:
    (channels, kernel_size, stride)
    The deconvolutional layers are automatically adjusted based on the input_shape
'''
def conv_output_size(size, kernel_size, stride=1, dilation=1, padding=0):
    return math.floor((size + 2*padding - dilation*(kernel_size-1)-1)/stride +1)

class VAE(nn.Module):

    def __init__(self, input_shape, latent_size=32, convs=[], beta=1.0):
        super(VAEGAN, self).__init__()
        self.latent_size = latent_size
        self.beta = beta
        # Check input shape
        assert len(input_shape) == 3, "Input shape must be 3D for images."

        # Create convolutional layers
        self.convolutional_layers = []
        prev_channels = input_shape[0]
        prev_shape = input_shape
        for channels, kernel_size, stride in convs:
            layer = nn.Sequential(nn.Conv2d(prev_channels, channels, kernel_size, stride=stride), nn.ReLU())
            self.convolutional_layers.append(layer)
            prev_channels = channels
            prev_shape = [channels, conv_output_size(prev_shape[1], kernel_size, stride=stride), conv_output_size(prev_shape[2], kernel_size, stride=stride)]
        # Create module list
        self.convolutional_layers = nn.ModuleList(self.convolutional_layers)

        # Encoder last layers
        flattened_final_shape = np.prod(prev_shape)
        self.encoder_mean_layer = nn.Linear(flattened_final_shape, self.latent_size)
        self.encoder_sigma_layer = nn.Linear(flattened_final_shape, self.latent_size)

        # Decoder first layers
        self.decoder_widen = nn.Linear(self.latent_size, flattened_final_shape)
        self.decoder_reconv = nn.Sequential(nn.ConvTranspose2d(flattened_final_shape, prev_shape[0], (prev_shape[1], prev_shape[2]), stride=1), nn.ReLU())

        # Deconvolutional layers
        self.deconvolutional_layers = []
        for i, (channels, kernel_size, stride) in enumerate(convs[:0:-1]):
            next_channels = convs[len(convs) - i - 2][0]
            layer = nn.Sequential(nn.ConvTranspose2d(channels, next_channels, kernel_size, stride=stride), nn.ReLU())
            self.deconvolutional_layers.append(layer)
        # Last deconv layer
        layer = nn.Sequential(nn.ConvTranspose2d(convs[0][0], input_shape[0], convs[0][1], stride=convs[0][2]), nn.Sigmoid())
        self.deconvolutional_layers.append(layer)
        # Create module list
        self.deconvolutional_layers = nn.ModuleList(self.deconvolutional_layers)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def encode(self, x):
        # Pass the convolutional layers
        for conv_layer in self.convolutional_layers:
            x = conv_layer(x)
        # Reshape to 2D for linear layers
        x = x.view(x.shape[0], -1)
        return self.encoder_mean_layer(x), self.encoder_sigma_layer(x)

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.decoder_widen(z)
        # Unflatten to 4D for deconvs
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        # Re-convolution
        z = self.decoder_reconv(z)
        # Deconvs
        for deconv_layer in self.deconvolutional_layers:
            z = deconv_layer(z)
        return z

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        rebuild = self.decode(z)
        assert x.shape == rebuild.shape, 'Image dimension not aligned.'
        return mu, log_sigma, z, rebuild

    def losses(self, x):
        # Forward pass
        mu, log_sigma, z, rebuild = self(x)
        # Reconstruction loss
        mse = ((x-rebuild)**2)
        mse = mse.view(mse.size(0), -1)
        reco_loss = torch.mean(torch.sum(mse, dim=1))
        # Normalization loss
        norm_loss = -0.5 * torch.sum(1 + log_sigma - mu**2 - log_sigma.exp(), dim=1)
        # KL tolerance
        norm_loss = torch.mean(norm_loss)
        # Total loss
        total_loss = reco_loss + self.beta * norm_loss
        return reco_loss, norm_loss, total_loss

    def optimize(self, x, optimizer):
        optimizer.zero_grad()
        reco_loss, norm_loss, total_loss = self.losses(x)
        total_loss.backward()
        optimizer.step()
        return reco_loss, norm_loss, total_loss

if __name__ == '__main__':
    # Perform tests
    test_shapes = [(3, 32, 34)]
    CONVS = [(32, 4, 2), (64, 4, 1), (128, 4, 1), (256, 4, 1)]
    BATCH = 32
    # Perform each test shape
    for test_shape in test_shapes:
        # Declare model
        vaegan = VAEGAN(test_shape, latent_size=32, convs=CONVS)
        # Forward pass
        result = vaegan(torch.from_numpy(np.zeros((BATCH,) + test_shape, dtype=np.float32)))
