'''
    Dataset generator script.
    This script runs the car racing environment with a random policy.
    TODO: add non-random policy

    It produces a pickle file containing a dataset of resized frames, already
    shuffled.
'''

import gym, time, argparse, cv2
from gym import wrappers, logger
import numpy as np
from tqdm import trange
import pickle

from wrappers import CropCarRacing, ResizeObservation, NormalizeRGB

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1000, help="""Number of frames to generate.""")
parser.add_argument('--output', type=str, default='car_dataset.pkl', help="""Output filename.""")
parser.add_argument('--seed', type=int, default=42, help="""Output filename.""")
args, unparsed = parser.parse_known_args()

env = gym.make('CarRacing-v0')
env = CropCarRacing(env)
env = ResizeObservation(env, (64, 64, 3))
env = NormalizeRGB(env)
env.seed(args.seed)
np.random.seed(args.seed)

dataset = []
BUFFER = 8
#env = wrappers.Monitor(env, directory='tmp', force=True)

obs = env.reset()
dataset.append(obs)
for i in trange(args.size + BUFFER):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    dataset.append(obs)

dataset = dataset[BUFFER:]

env.close()

np.random.shuffle(dataset)
print("Generated dataset.")

# Save to file
with open(args.output, 'wb') as outfile:
    pickle.dump(dataset, outfile)

print("Saved dataset to", FLAGS.output)
