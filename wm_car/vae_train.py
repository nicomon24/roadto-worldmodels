'''
    Train a Variational AutoEncoder using a specific dataset.
    It writes summaries for tensorboard and checkpoint to reuse the model.
'''

import numpy as np
import tensorflow as tf
import gym, pickle, os, argparse
from tqdm import trange

from vae import create_vae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="""Size of the image batches.""")
    parser.add_argument('--latent_size', type=int, default=32, help="""Size of the latent vector.""")
    parser.add_argument('--save_interval', type=int, default=100, help="""How frequently to save a checkpoint.""")
    parser.add_argument('--dropout', type=float, default=0.9, help="""Keep probability of dropout.""")
    parser.add_argument('--beta', type=float, default=1.0, help="""Beta parameter.""")
    parser.add_argument('--epochs', type=str, default='100', help="""Comma separated epochs of training.""")
    parser.add_argument('--learning_rates', type=str, default='1e-05', help="""Comma separated learning rates. (Must be of the same size of epochs)""")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help="""Directory in which checkpoints are saved.""")
    parser.add_argument('--log_dir', type=str, default='logs', help="""Directory in which logs for tensorboard are saved.""")
    parser.add_argument('--checkpoint', type=str, default='', help="""Path of a checkpoint to restore.""")
    parser.add_argument('--start_epoch', type=int, default=0, help="""Start epoch when loading a checkpoint.""")
    parser.add_argument('--alias', type=str, default='base', help="""Alias of the model.""")
    parser.add_argument('--arch', type=str, default='base_car_racing', help="""Model architecture.""")
    parser.add_argument('--dataset', type=str, default=None, help="""Dataset file to load.""")
    args, unparsed = parser.parse_known_args()
    # Parse comma-separated learning rates and epochs
    LEARNING_RATES = list(map(float, args.learning_rates.split(',')))
    EPOCHS = list(map(int, args.epochs.split(',')))
    assert len(LEARNING_RATES) == len(EPOCHS), 'Epochs and Learning rates must be of the same size!'
    # Loading the dataset
    assert args.dataset is not None, 'No dataset provided!'
    dataset = np.array(pickle.load(open(args.dataset, 'rb')))
    N_SAMPLES, W, H, CHANNELS = dataset.shape
    print("Dataset size:", N_SAMPLES)
    print("Channels:", CHANNELS)
    print("Image dim: (%d,%d)" % (W,H))
    # Network creation
    tf.reset_default_graph()
    X, z, rebuild, tf_optimizer, batch_size, learning_rate, keep_prob, losses = create_vae((W,H,CHANNELS),
                                                        arch=args.arch, latent_size=args.latent_size, beta=args.beta,
                                                        is_training=True, optimizer=tf.train.AdamOptimizer)
    loss_names, losses_tf = zip(*losses)
    losses_with_optimizer = [tf_optimizer] + list(losses_tf)
    # Session and init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Saver and summary writer
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)
    train_writer = tf.summary.FileWriter(os.path.join(args.log_dir, args.alias + '_train'))
    # Check if we need to load a checkpoint
    if args.checkpoint:
        _saver = tf.train.Saver(tf.global_variables())
        _saver.restore(sess, args.checkpoint)
    # Training
    print("Start training...")
    # Setting index of the learning_rate/epochs
    lr_index = 0
    for epoch in trange(args.start_epoch, sum(EPOCHS)):
        # Update index and select current learning rate
        if epoch > sum(EPOCHS[:lr_index+1]):
            lr_index += 1
        lr = LEARNING_RATES[lr_index]
        losses_dict = {key: [] for key in loss_names}
        for bindex in range(0, len(dataset), args.batch_size):
            batch = dataset[bindex:bindex+args.batch_size]
            res = sess.run(losses_with_optimizer, feed_dict = {X: batch, keep_prob:args.dropout, learning_rate:lr, batch_size:len(batch)})[1:] # Discard optimizer
            for i, loss_name in enumerate(loss_names):
                losses_dict[loss_name].append(res[i])
        # Write summaries
        summary = tf.Summary()
        summary.value.add(tag='learning_rate', simple_value=lr)
        for loss_name in loss_names:
            summary.value.add(tag=loss_name, simple_value=np.mean(losses_dict[loss_name]))
        train_writer.add_summary(summary, epoch)
        # Check if we need to save
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, args.alias + '-' + str(epoch) + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, epoch)
            saver.save(sess, checkpoint_path, write_meta_graph=True)

    # Save the last model
    checkpoint_path = os.path.join(args.save_dir, args.alias + '-' + str(sum(EPOCHS)) + '.ckpt')
    tf.logging.info('Saving to "%s-%d"', checkpoint_path, epoch)
    saver.save(sess, checkpoint_path, write_meta_graph=True)

    print("Bye bye")
