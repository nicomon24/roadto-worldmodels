'''
    Dataset generator script.
    This script runs the car racing environment with a random policy.
    TODO: add non-random policy

    It produces a pickle file containing a dataset of resized frames, already
    shuffled.
'''

import gym, time, argparse, cv2
import numpy as np
from tqdm import trange
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1000,
  help="""\
  Number of frames to generate.
""")
parser.add_argument('--output', type=str, default='car_dataset.pkl',
  help="""\
  Output filename.
""")
parser.add_argument('--seed', type=int, default=42,
  help="""\
  Output filename.
""")
FLAGS, unparsed = parser.parse_known_args()

# We need a function to crop from 96x96 down to 84x96 and resize from 84x96 to 64x64
def resize(img):
    img = img[:84,:,:]
    img = cv2.resize(img, dsize=(80, 80))
    return img

env = gym.make('CarRacing-v0')
env.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

dataset = []

obs = env.reset()
dataset.append(resize(obs) / 255)
for i in trange(FLAGS.size - 1):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    dataset.append(resize(obs) / 255)

np.random.shuffle(dataset)
print("Generated dataset.")

# Save to file
with open(FLAGS.output, 'wb') as outfile:
    pickle.dump(dataset, outfile)

print("Saved dataset to", FLAGS.output)
