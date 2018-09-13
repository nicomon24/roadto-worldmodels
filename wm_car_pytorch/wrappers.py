'''
    Collection of environment wrappers
'''

import gym, cv2
import numpy as np

from utils import NCHW, NHWC
from vae import VAE
import torch

'''
    Resize the observation to a specific size.
'''
class ResizeObservation(gym.ObservationWrapper):

    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        self.env.observation_space.shape = size
        self.size = size

    def _observation(self, obs):
        return cv2.resize(obs, dsize=self.size[:2])

'''
    Crop the lower black bar in the car racing environment.
'''
class CropCarRacing(gym.ObservationWrapper):

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space.shape = (84, 96, 3)

    def _observation(self, obs):
        return obs[:84,:,:]

'''
    Returns a Black/White observation instead of an RGB one.
'''
class Scolorized(gym.ObservationWrapper):

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space.shape = self.observation_space.shape[:2] + (1,) # Change the channels to 1

    def _observation(self, obs):
        return np.reshape(np.mean(obs, axis=2), (obs.shape[0], obs.shape[1], 1))

'''
    Set the RGB values from [0-255] to [0-1] floats.
'''
class NormalizeRGB(gym.ObservationWrapper):

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.env.observation_space.shape)

    def _observation(self, obs):
        return obs / 255

'''
    Return the compressed representation of the selected VAE model.
'''
class VAEObservation(gym.ObservationWrapper):

    def __init__(self, env, vae_filepath, arch='base_car_racing', latent_size=32, device=None):
        gym.ObservationWrapper.__init__(self, env)
        # Load VAE
        self.vae = VAE(arch=arch, latent_size=latent_size)
        self.device = device
        assert vae_filepath, "No VAE checkpoint provided."
        self.vae.load_state_dict(torch.load(vae_filepath))
        self.vae.eval()
        # Observation space
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(self.vae.latent_size,))

    def _observation(self, obs):
        # First convert to torch notation and type
        obs_torch = torch.from_numpy(NCHW([obs])).float()
        if self.device is not None:
            obs_torch = obs_torch.to(self.device)
        # Get the compressed
        mu, _ = self.vae.encode(obs_torch)
        return mu.detach().numpy()[0]
