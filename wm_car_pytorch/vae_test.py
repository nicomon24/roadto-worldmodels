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

from vae import VAE
from wrappers import CropCarRacing, ResizeObservation, Scolorized, NormalizeRGB
from utils import side_by_side, NCHW, NHWC, imshow_bw_or_rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, default='', help="""Path of a checkpoint to restore.""")
    parser.add_argument('--latent_size', type=int, default=32, help="""Size of the latent vector.""")
    parser.add_argument('--sample', type=int, default=0, help="""Specify the sample to visualize.""")
    parser.add_argument('--dataset', type=str, default='', help="""Dataset file to load.""")
    parser.add_argument('--arch', type=str, default='base_car_racing', help="""Model architecture.""")
    parser.add_argument('--seed', type=int, default=42, help="""Seed used in the environment initialization.""")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
    args, unparsed = parser.parse_known_args()
    # Check cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    # Loading the dataset
    if args.dataset:
        dataset = np.array(pickle.load(open(args.dataset, 'rb')))
        N_SAMPLES, W, H, CHANNELS = dataset.shape
        print("Dataset size:", N_SAMPLES)
        print("Channels:", CHANNELS)
        print("Image dim: (%d,%d)" % (W,H))
        dataset_torch = torch.from_numpy(NCHW(dataset)).float().to(device)
    else:
        print("Using gym environment directly.")
        env = gym.make('CarRacing-v0')
        env = CropCarRacing(env)
        env = ResizeObservation(env, (64, 64, 3))
        env = NormalizeRGB(env)
        # env = Scolorized(env)
        env.seed(args.seed)

    # Network creation
    vae = VAE(arch=args.arch, latent_size=args.latent_size)
    # Restore checkpoint
    assert args.vae, "No checkpoint provided."
    vae.load_state_dict(torch.load(args.vae))
    vae.eval()

    if args.dataset:
        # Single observation display
        mu, log_sigma, z, rebuild = vae(dataset_torch[args.sample:args.sample+1])
        rebuild = rebuild.detach().numpy()[0]
        imshow_bw_or_rgb(side_by_side(dataset[args.sample], NHWC(rebuild)))
        plt.show()
    else:
        # Animation of environment
        obs = env.reset()
        obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
        mu, log_sigma, z, rebuild = vae(obs_torch)
        rebuild = NHWC(rebuild.detach().numpy()[0])

        fig1 = plt.figure()
        im = plt.imshow(side_by_side(obs, rebuild))
        done = False

        # Setting animation update function
        def updatefig(*args):
            global done
            if not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                env.render(mode='human')
            else:
                done = False
                obs = env.reset()
            obs_torch = torch.from_numpy(NCHW([obs])).float().to(device)
            mu, log_sigma, z, rebuild = vae(obs_torch)
            rebuild = NHWC(rebuild.detach().numpy()[0])
            im.set_array(side_by_side(obs, rebuild))
            time.sleep(0.01)
            return im,

        # Start animation
        ani = animation.FuncAnimation(fig1, updatefig, interval=50, blit=True)
        plt.show()
