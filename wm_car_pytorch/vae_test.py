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

def NCHW(x):
    x = np.array(x)
    if len(x.shape) == 4:
        return np.transpose(x, (0, 3, 1, 2))
    elif len(x.shape) == 3:
        return np.transpose(x, (2, 0, 1))
    else:
        raise Exception("Unrecognized shape.")

def NHWC(x):
    x = np.array(x)
    if len(x.shape) == 4:
        return np.transpose(x, (0, 2, 3, 1))
    elif len(x.shape) == 3:
        return np.transpose(x, (1, 2, 0))
    else:
        raise Exception("Unrecognized shape.")

def imshow_bw_or_rgb(img):
    if img.shape[-1] == 1:
        plt.imshow(img[:,:,0], cmap="Greys")
    elif img.shape[-1] == 3:
        plt.imshow(img)
    else:
        raise Exception('Unrecognized image format')

def side_by_side(img1, img2):
    SIZE = 4
    if len(img1.shape) == 2:
        return np.concatenate([img1, np.ones((img1.shape[0], SIZE)), img2], axis=1)
    elif len(img1.shape) == 3:
        return np.concatenate([img1, np.ones((img1.shape[0], SIZE, img1.shape[2])), img2], axis=1)
    else:
        raise Exception("Unrecognized observation format!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='', help="""Path of a checkpoint to restore.""")
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
    assert args.checkpoint, "No checkpoint provided."
    vae.load_state_dict(torch.load(args.checkpoint))
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
