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
import tensorflow as tf
import numpy as np

from vae import create_vae
from wrappers import CropCarRacing, ResizeObservation, Scolorized, NormalizeRGB

def imshow_bw_or_rgb(img):
    if img.shape[-1] == 1:
        plt.imshow(img[:,:,0], cmap="Greys")
    elif img.shape[-1] == 3:
        plt.imshow(img)
    else:
        raise Exception('Unrecognized image format')

def side_by_side(img1, img2):
    SIZE = 4
    print(len(img1.shape))
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
    args, unparsed = parser.parse_known_args()
    # Loading the dataset
    if args.dataset:
        dataset = np.array(pickle.load(open(args.dataset, 'rb')))
        N_SAMPLES, W, H, CHANNELS = dataset.shape
        print("Dataset size:", N_SAMPLES)
        print("Channels:", CHANNELS)
        print("Image dim: (%d,%d)" % (W,H))
    else:
        print("Using gym environment directly.")
        env = gym.make('CarRacing-v0')
        env = CropCarRacing(env)
        env = ResizeObservation(env, (64, 64, 3))
        env = NormalizeRGB(env)
        # env = Scolorized(env)
        W, H, CHANNELS = env.observation_space.shape
        env.seed(args.seed)

    # Network creation
    tf.reset_default_graph()
    X, z, rebuild, batch_size, keep_prob = create_vae((W,H,CHANNELS), arch=args.arch, latent_size=args.latent_size, is_training=False)
    # Session and init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Restore checkpoint
    assert args.checkpoint, "No checkpoint provided."
    _saver = tf.train.Saver(tf.global_variables())
    _saver.restore(sess, args.checkpoint)

    if args.dataset:
        # Single observation display
        obs = dataset[args.sample]
        reco = sess.run(rebuild, feed_dict={X: [obs], keep_prob:1.0, batch_size:1})[0]
        imshow_bw_or_rgb(side_by_side(obs, reco))
        plt.show()
    else:
        # Animation of environment
        obs = env.reset()
        reco = sess.run(rebuild, feed_dict={X: [obs], keep_prob:1.0, batch_size:1})[0]
        fig1 = plt.figure()
        im = plt.imshow(side_by_side(obs, reco))
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
            reco = sess.run(rebuild, feed_dict={X: [obs], keep_prob:1.0, batch_size:1})[0]
            im.set_array(side_by_side(obs, reco))
            time.sleep(0.01)
            return im,

        # Start animation
        ani = animation.FuncAnimation(fig1, updatefig, interval=50, blit=True)
        plt.show()
