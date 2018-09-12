'''
    Specification of the policy. This policy is specifically created for
    the CarRacing environment, so the action space is taken as granted
    Action space has 3 float values: steer[-1,1], gas[0,1], brake[0,1].

    Future TODO: environment independence
'''

import tensorflow as tf

def create_policy(input_shape, arch='base_car_racing', distribution='gaussian', fixed_variance=True):
    input = tf.placeholder(dtype=tf.float32, shape=input_shape)
    if arch == 'base_car_racing':
        _Y = input
    else:
        raise Exception('Unrecognized architecture.')
    # Define output distributions
    if distribution == 'gaussian':
        mu = tf.layers.dense(_Y, 3, activation=None)
        variance = 0.1
        # TODO: add variance if not fixed
        
    return None
