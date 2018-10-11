'''
    Sampling script.
    Simulate N episodes of a particular environment using the specified policy.

    TODO: add multi-threading
'''

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import NCHW

class Sampler():

    def __init__(self, env, horizon):
        self.env = env
        self.horizon = horizon

    def sample(self, n_episodes, vae_policy, render=False):
        trajectories = {
            'rewards': np.zeros((n_episodes, self.horizon)),
            'mask' : np.zeros((n_episodes, self.horizon)),
            'logp': torch.zeros(n_episodes, self.horizon, dtype=torch.double)
        }
        losses_and_info = []
        for i in range(n_episodes):
            obs = self.env.reset()
            for t in range(self.horizon):
                # Transform obs for PyTorch and optimize the observation (save to VAE)
                obs_torch = torch.from_numpy(NCHW([obs])).float()
                results = vae_policy.optimize_vae(obs_torch)
                losses_and_info.append(results)
                # Act
                a, logp = vae_policy.act(obs_torch)
                obs, r, done, _ = self.env.step(a)
                # Add to trajectories
                trajectories['rewards'][i, t] = r
                trajectories['logp'][i, t] = logp
                trajectories['mask'][i, t] = 1
                if render:
                    self.env.render()
                if done:
                    break
        return trajectories, losses_and_info
