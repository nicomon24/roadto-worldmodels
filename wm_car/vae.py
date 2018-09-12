'''
    Specification of the Variational AutoEncoder in tensorflow.
    Change this for changes on the architecture.
'''

#TODO: allow different architectures

import tensorflow as tf
import numpy as np

def create_vae(input_shape, arch='base_car_racing', latent_size=32, beta=1.0, is_training=True, optimizer=None):
    # Get the input shape parameters
    W, H, CHANNELS = input_shape
    # Input image placeholder
    X = tf.placeholder(dtype=tf.float32, shape=(None, W, H, CHANNELS))
    # Param placeholders
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
    # Add learning rate in the symmaries
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    if arch == 'base_car_racing':
        conv_layers = [(32, 4, 2, tf.nn.relu),
                        (64, 4, 2, tf.nn.relu),
                        (128, 4, 2, tf.nn.relu),
                        (256, 4, 2, tf.nn.relu)] #(filters, size, stride, activation)
        deconv_layers = [(128, 5, 2, tf.nn.relu),
                        (64, 5, 2, tf.nn.relu),
                        (32, 6, 2, tf.nn.relu),
                        (CHANNELS, 6, 2, tf.sigmoid)] #(filters, size, stride, activation)
    else:
        raise Exception('Unknown architecture.')

    # ENCODER GRAPH
    with tf.variable_scope("encoder", reuse=None):
        _Y = X
        for filters, kernel_size, strides, activation in conv_layers:
            _Y = tf.layers.conv2d(_Y, filters=filters, kernel_size=kernel_size, strides=strides, activation=activation)
            _Y = tf.nn.dropout(_Y, keep_prob)
        flat = tf.layers.flatten(_Y)
        z_mu = tf.layers.dense(flat, units=latent_size)
        z_log_sigma_sq = tf.layers.dense(flat, units=latent_size)
        z_noise = tf.random_normal(shape=(batch_size, latent_size))
        z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * z_noise

    # DECODER GRAPH
    with tf.variable_scope("decoder", reuse=None):
        deflat = tf.layers.dense(z, units=flat.shape[1])
        deflat4d = tf.reshape(deflat, shape=(-1, 1, 1, flat.shape[1]))
        _Y = deflat4d
        for filters, kernel_size, strides, activation in deconv_layers:
            _Y = tf.layers.conv2d_transpose(_Y, filters=filters, kernel_size=kernel_size, strides=strides, activation=activation)
            _Y = tf.nn.dropout(_Y, keep_prob)
        rebuild = tf.reshape(_Y, shape=(-1, W, H, CHANNELS))
        assert rebuild.shape[1:]==X.shape[1:], 'Incompatible architecture!'

    if not is_training:
        return X, z, rebuild, batch_size, keep_prob

    # Reconstruction loss
    reconstruction_loss  = tf.losses.mean_squared_error(X, rebuild)
    # Regularization loss
    reg_loss = -tf.reduce_mean(0.5 * (1 + z_log_sigma_sq - z_mu**2 - tf.exp(z_log_sigma_sq)))
    # Complete loss
    complete_loss = reconstruction_loss + reg_loss * beta
    # Optimizer
    assert optimizer is not None, 'Must specify an optimizer in training mode!'
    tf_optimizer = optimizer(learning_rate).minimize(complete_loss)
    # Aggregate losses for summaries
    losses = [
        ('reconstruction_loss', reconstruction_loss),
        ('normalization_loss', reg_loss),
        ('complete_loss', complete_loss)
    ]
    # Return all needed
    return X, z, rebuild, tf_optimizer, batch_size, learning_rate, keep_prob, losses
