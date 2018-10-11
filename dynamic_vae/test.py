'''
    Test the VAE reconstruction results on a specified dataset (not the environment as in render)
'''

'''
    This script, after loading a VAE checkpoints, shows a snek environment and
    its VAE reconstruction over time.
'''

import gym, argparse, pickle, time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from vae_policy import VAEPolicy
from wrappers import CropCarRacing, ResizeObservation, Scolorized, NormalizeRGB
from utils import side_by_side, NCHW, NHWC, imshow_bw_or_rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=str, default=None, help="""Controller checkpoint to restore.""")
    parser.add_argument('--seed', type=int, default=42, help="""Seed used in the environment initialization.""")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
    args, unparsed = parser.parse_known_args()
    # Check cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make('CarRacing-v0')
    env = CropCarRacing(env)
    env = ResizeObservation(env, (64, 64, 3))
    env = NormalizeRGB(env)
    env = Scolorized(env, weights=[0.0, 1.0, 0.0])
    env.seed(args.seed)

    vape = VAEPolicy()
    # Restore checkpoint
    assert args.controller, "No checkpoint provided."
    vape.load_state_dict(torch.load(args.controller))
    vape.eval()

    # Animation of environment
    obs = env.reset()
    obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
    rebuild, mu, log_sigma, z = vape.encode_decode(obs_torch)
    rebuild = NHWC(rebuild.detach().numpy()[0])

    fig1 = plt.figure()
    if len(obs.shape) == 3 and (obs.shape[-1]==1):
        im = plt.imshow(side_by_side(obs, rebuild), cmap="Greys")
    else:
        im = plt.imshow(side_by_side(obs, rebuild))
    done = False

    # Setting animation update function
    def updatefig(*args):
        global done
        global obs
        if not done:
            obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
            mu, _ = vape.encode(obs_torch)
            action, logp = vape.act(obs_torch)
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
        else:
            done = False
            obs = env.reset()
        obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
        rebuild, mu, log_sigma, z = vape.encode_decode(obs_torch)
        rebuild = NHWC(rebuild.detach().numpy()[0])
        im.set_array(side_by_side(obs, rebuild))
        time.sleep(0.01)
        return im,

    # Start animation
    ani = animation.FuncAnimation(fig1, updatefig, interval=50, blit=True)
    plt.show()
