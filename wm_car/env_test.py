import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tqdm import trange

env = gym.make('CarRacing-v0')

dataset = []
env.seed(42)
obs = env.reset()
done = False

for i in trange(20):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    dataset.append(obs)
    env.render()
env.close()

plt.imshow(dataset[-1])
plt.show()
