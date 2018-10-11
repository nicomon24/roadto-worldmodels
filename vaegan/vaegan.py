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
    Create the VAEGAN model. The architecture is specified using the 'convs'
    parameter, an array of tuple:
    (channels, kernel_size, stride)
    The deconvolutional layers are automatically adjusted based on the input_shape
'''
def conv_output_size(size, kernel_size, stride=1, dilation=1, padding=0):
    return math.floor((size + 2*padding - dilation*(kernel_size-1)-1)/stride +1)

class EncoderModule(nn.Module):

    def __init__(self, input_shape, latent_size, convs):
        super(EncoderModule, self).__init__()
        self.latent_size = latent_size

        # Create encode layers
        self.encoder_layers = []
        prev_channels = input_shape[0]
        prev_shape = input_shape
        for channels, kernel_size, stride in convs:
            layer = nn.Sequential(nn.Conv2d(prev_channels, channels, kernel_size, stride=stride), nn.ReLU())
            self.encoder_layers.append(layer)
            prev_channels = channels
            prev_shape = [channels, conv_output_size(prev_shape[1], kernel_size, stride=stride), conv_output_size(prev_shape[2], kernel_size, stride=stride)]
        # Create module list
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

        # Save dimensions
        self.last_layer = prev_shape
        self.flattened_final_shape = np.prod(prev_shape)

        # Encoder last layers
        self.encoder_mean_layer = nn.Linear(self.flattened_final_shape, self.latent_size)
        self.encoder_sigma_layer = nn.Linear(self.flattened_final_shape, self.latent_size)

    def forward(self, x):
        # Pass the convolutional layers
        for conv_layer in self.encoder_layers:
            x = conv_layer(x)
        # Reshape to 2D for linear layers
        x = x.view(x.shape[0], -1)
        return self.encoder_mean_layer(x), self.encoder_sigma_layer(x)

class DecoderModule(nn.Module):

    def __init__(self, input_shape, flattened_final_shape, prev_shape, latent_size, convs):
        super(DecoderModule, self).__init__()
        self.latent_size = latent_size

        # Decoder first layers
        self.decoder_widen = nn.Linear(self.latent_size, flattened_final_shape)
        self.decoder_reconv = nn.Sequential(nn.ConvTranspose2d(flattened_final_shape, prev_shape[0], (prev_shape[1], prev_shape[2]), stride=1), nn.ReLU())

        # Decoder layers
        self.decoder_layers = []
        for i, (channels, kernel_size, stride) in enumerate(convs[:0:-1]):
            next_channels = convs[len(convs) - i - 2][0]
            layer = nn.Sequential(nn.ConvTranspose2d(channels, next_channels, kernel_size, stride=stride), nn.ReLU())
            self.decoder_layers.append(layer)
        # Last deconv layer
        layer = nn.Sequential(nn.ConvTranspose2d(convs[0][0], input_shape[0], convs[0][1], stride=convs[0][2]), nn.Sigmoid())
        self.decoder_layers.append(layer)
        # Create module list
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

    def forward(self, z):
        z = self.decoder_widen(z)
        # Unflatten to 4D for deconvs
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        # Re-convolution
        z = self.decoder_reconv(z)
        # Deconvs
        for deconv_layer in self.decoder_layers:
            z = deconv_layer(z)
        return z

class DiscriminatorModule(nn.Module):

    def __init__(self, input_shape, flattened_final_shape, latent_size, convs):
        super(DiscriminatorModule, self).__init__()
        self.latent_size = latent_size

        # Discriminator layers (same as encoder)
        self.discriminator_layers = []
        prev_channels = input_shape[0]
        prev_shape = input_shape
        for channels, kernel_size, stride in convs:
            layer = nn.Sequential(nn.Conv2d(prev_channels, channels, kernel_size, stride=stride), nn.ReLU())
            self.discriminator_layers.append(layer)
            prev_channels = channels
        # Create module list
        self.discriminator_layers = nn.ModuleList(self.discriminator_layers)
        self.discriminator_proba = nn.Sequential(nn.Linear(flattened_final_shape, 1), nn.Sigmoid())

    def forward(self, x):
        # Pass the convolutional layers
        for conv_layer in self.discriminator_layers:
            x = conv_layer(x)
        last_layer_data = x
        # Reshape to 2D for linear layers
        x = x.view(x.shape[0], -1)
        return self.discriminator_proba(x), last_layer_data

class VAEGAN(nn.Module):

    def __init__(self, input_shape, latent_size=32, convs=[]):
        super(VAEGAN, self).__init__()
        self.latent_size = latent_size
        # Check input shape
        assert len(input_shape) == 3, "Input shape must be 3D for images."

        # Declare modules
        self.encoder = EncoderModule(input_shape, latent_size, convs)
        self.decoder = DecoderModule(input_shape, self.encoder.flattened_final_shape, self.encoder.last_layer, latent_size, convs)
        self.discriminator = DiscriminatorModule(input_shape, self.encoder.flattened_final_shape, latent_size, convs)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                #nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def set_device(self, device):
        self.default_device = device

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def forward(self, x):
        # Encode-decode the samples
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        rebuild = self.decode(z)
        assert x.shape == rebuild.shape, 'Image dimension not aligned.'
        # Discriminator on sample and sample's reconstruction
        dis_true, dis_l_true = self.discriminate(x)
        dis_rebuild, dis_l_rebuild = self.discriminate(rebuild)
        # Generate noise and decode
        noise = torch.randn(z.shape[0], z.shape[1]).to(self.default_device)
        rebuild_noise = self.decode(noise)
        dis_noise, dis_l_noise = self.discriminate(rebuild_noise)
        return mu, log_sigma, z, rebuild, dis_true, dis_l_true, dis_rebuild, dis_l_rebuild, dis_noise, dis_l_noise

    def losses(self, x):
        # Forward pass
        mu, log_sigma, z, rebuild, dis_true, dis_l_true, dis_rebuild, dis_l_rebuild, dis_noise, dis_l_noise = self(x)

        # Prior loss: KL-divergence
        prior_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu**2 - log_sigma.exp(), dim=1))
        if torch.isnan(prior_loss):
            print(self.encoder.encoder_mean_layer.weight)
            print(log_sigma)
            print(mu)
            exit(0)

        # Dis-like loss
        mse = ((dis_l_true-dis_l_rebuild)**2)
        mse = mse.view(mse.size(0), -1)
        dislike_loss = torch.mean(-0.5 * torch.sum(mse, dim=1))

        print("DIS_MAX", torch.max(dis_true), torch.max(dis_rebuild), torch.max(dis_noise))
        print("DIS_MIN", torch.min(dis_true), torch.min(dis_rebuild), torch.min(dis_noise))
        # GAN loss
        gan_loss = torch.mean(torch.log(dis_true) + torch.log(1-dis_rebuild) + torch.log(1-dis_noise))

        return prior_loss, dislike_loss, gan_loss

    def optimize(self, x, lr=1e-4, gamma=1.0):
        # Create the 3 optimizers
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        # Get losses
        prior_loss, dislike_loss, gan_loss = self.losses(x)
        # Module losses
        encoder_loss = prior_loss + dislike_loss
        decoder_loss = gamma * dislike_loss - gan_loss
        discriminator_loss = gan_loss
        print("LOSS1", prior_loss, dislike_loss, gan_loss)
        print("LOSS2", encoder_loss, decoder_loss, discriminator_loss)

        # Backward
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        encoder_loss.backward(retain_graph=True)
        encoder_optimizer.step()

        decoder_loss.backward(retain_graph=True)
        decoder_optimizer.step()

        discriminator_loss.backward()
        discriminator_optimizer.step()

        return prior_loss, dislike_loss, gan_loss

if __name__ == '__main__':
    # Perform tests
    test_shapes = [(3, 32, 32)]
    CONVS = [(32, 4, 2), (64, 4, 1), (128, 4, 1), (256, 4, 1)]
    BATCH = 32
    # Perform each test shape
    for test_shape in test_shapes:
        # Declare model
        vaegan = VAEGAN(test_shape, latent_size=32, convs=CONVS)
        # Forward pass
        result = vaegan.optimize(torch.from_numpy(np.zeros((BATCH,) + test_shape, dtype=np.float32)))
