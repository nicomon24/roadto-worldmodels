'''
    Collection of environment wrappers
'''

import gym, cv2
import numpy as np

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

class NormalizeRGB(gym.ObservationWrapper):

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.env.observation_space = gym.spaces.Box(low=0, high=1, shape=self.env.observation_space.shape)

    def _observation(self, obs):
        return obs / 255
