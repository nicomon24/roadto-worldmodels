'''
    Guess what
'''

import argparse, gym, time, os
import numpy as np
from collections import deque
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from wrappers import CropCarRacing, ResizeObservation, Scolorized, NormalizeRGB

from vae_policy import VAEPolicy
from utils import side_by_side, NCHW, NHWC, imshow_bw_or_rgb

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='REINFORCE using PyTorch')
    # Logging
    parser.add_argument('--alias', type=str, default='base', help="""Alias of the model.""")
    parser.add_argument('--render_interval', type=int, default=100, help='interval between rendered epochs (default: 100)')
    # Learning parameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
    parser.add_argument('--eb', type=int, default=1, help='episode batch (default: 1)')
    parser.add_argument('--episodes', type=int, default=10000, help='simulated episodes (default: 10000)')
    parser.add_argument('--policy', type=str, default=None, help="""Policy checkpoint to restore.""")
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--horizon', type=int, default=1000, help='horizon (default: 1000)')
    parser.add_argument('--baseline', action='store_true', help='use the baseline for the REINFORCE algorithm')
    args = parser.parse_args()
    # Check cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    # Initialize environment
    env = gym.make('CarRacing-v0')
    env = CropCarRacing(env)
    env = ResizeObservation(env, (32, 32, 3))
    env = Scolorized(env, weights=[0.0, 1.0, 0.0])
    env = NormalizeRGB(env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Env final goal:", env.spec.reward_threshold)
    # Create the alias for the run
    alias = '%s_%s' % (args.alias, time.time())
    # Use alias for checkpoints
    checkpoint_best_filename = 'policy_weights/' + alias + '_best.torch'
    checkpoint_final_filename = 'policy_weights/' + alias + '_final.torch'
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    # Tensorboard writer
    writer = SummaryWriter('logs/' + alias)
    # Create VAE policy
    vape = VAEPolicy()

    def show_centroids():
        c = NHWC(vape.centroids.detach().numpy())
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0,0].imshow(c[0,:,:,0], cmap='Greys')
        ax[0,1].imshow(c[1,:,:,0], cmap='Greys')
        ax[1,0].imshow(c[2,:,:,0], cmap='Greys')
        ax[1,1].imshow(c[3,:,:,0], cmap='Greys')
        plt.show()

    optimizer = optim.Adam(vape.parameters(), lr=1e-03)

    def transform_observation(obs):
        obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
        rebuild = vape.encode_decode(obs_torch)
        rebuild = NHWC(rebuild.detach().numpy()[0])
        return rebuild, obs_torch

    obs = env.reset()
    done = False
    HORIZON = 200
    timestep = 0

    rebuild, obs_torch = transform_observation(obs)

    fig1 = plt.figure()
    if len(obs.shape) == 3 and (obs.shape[-1]==1):
        c = NHWC(vape.centroids.detach().numpy())
        im = plt.imshow(side_by_side(side_by_side(side_by_side(obs, rebuild), c[0,:,:,0]), c[1,:,:,0]), cmap="Greys")
    else:
        im = plt.imshow(side_by_side(obs, rebuild))

    # Setting animation update function
    def updatefig(*args):
        nonlocal done
        nonlocal obs
        nonlocal HORIZON
        nonlocal timestep
        rebuild, obs_torch = transform_observation(obs)
        if not done and timestep < HORIZON:
            #action, action_proba = vape.act(obs_torch)
            #action = action[0].detach().numpy()
            action = env.action_space.sample()
            action = [action[0], 0.3, 0.0]
            obs, reward, done, info = env.step(action)
            #env.render(mode='human')
            timestep += 1
        else:
            done = False
            obs = env.reset()
            timestep = 0
        c = NHWC(vape.centroids.detach().numpy())
        im.set_array(side_by_side(side_by_side(side_by_side(obs, rebuild), c[0,:,:,0]), c[1,:,:,0]))
        vape.optimize_vae(obs_torch, optimizer)
        time.sleep(0.01)
        return im,

    # Start animation
    ani = animation.FuncAnimation(fig1, updatefig, interval=50, blit=True)
    plt.show()
    # Close env and writer
    env.close()
    writer.close()

if __name__ == '__main__':
    main()
