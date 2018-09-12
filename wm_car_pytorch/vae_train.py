'''
    Train a Variational AutoEncoder using a specific dataset.
    It writes summaries for tensorboard and checkpoint to reuse the model.
'''

import numpy as np
import gym, pickle, os, argparse
from tqdm import trange
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from vae import VAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="""Size of the image batches.""")
    parser.add_argument('--latent_size', type=int, default=32, help="""Size of the latent vector.""")
    parser.add_argument('--save_interval', type=int, default=100, help="""How frequently to save a checkpoint.""")
    parser.add_argument('--dropout', type=float, default=0.9, help="""Dropout probability of dropout.""")
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
    parser.add_argument('--seed', type=int, default=42, help="""Randomization seed.""")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CUDA training')
    args, unparsed = parser.parse_known_args()
    # Check cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
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

    # NOTE: PyTorch accepts batches in the form [Batch_size, Channels, W, H], so we need to transpose
    dataset = np.transpose(dataset, [0, 3, 1, 2])
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    dataset = torch.from_numpy(dataset).float().to(device)

    # Create the VAE and the optimizer
    vae = VAE(arch=args.arch, latent_size=args.latent_size, beta=args.beta, dropout_proba=args.dropout).to(device)
    writer = SummaryWriter('logs/' + args.alias)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    # TODO: checkpoint loading

    # Training
    print("Start training...")
    vae.train()
    # Setting index of the learning_rate/epochs
    lr_index = 0
    for epoch in trange(args.start_epoch, sum(EPOCHS)):
        # Update index and select current learning rate
        if epoch > sum(EPOCHS[:lr_index+1]):
            lr_index += 1
        lr = LEARNING_RATES[lr_index]
        optimizer = optim.Adam(vae.parameters(), lr=lr)
        reco_losses, norm_losses, total_losses = [], [], []
        for bindex in range(0, len(dataset), args.batch_size):
            batch = dataset[bindex:bindex+args.batch_size]
            _reco, _norm, _total = vae.optimize(batch, optimizer)
            reco_losses.append(_reco.item())
            norm_losses.append(_norm.item())
            total_losses.append(_total.item())
        # Write losses to tensorboard
        writer.add_scalar('data/reconstruction_loss', np.mean(reco_losses), epoch)
        writer.add_scalar('data/normalization_loss', np.mean(norm_losses), epoch)
        writer.add_scalar('data/complete_loss', np.mean(total_losses), epoch)
        writer.add_scalar('data/learning_rate', lr, epoch)
        # Check if we need to save
        if (epoch > 0) and (epoch % args.save_interval == 0):
            filepath = 'checkpoints/' + args.alias + '-' + str(epoch) + '.torch'
            torch.save(vae.state_dict(), filepath)

    # End training
    writer.close()
    # Save last checkpoint
    filepath = 'checkpoints/' + args.alias + '-' + str(epoch) + '[final].torch'
    torch.save(vae.state_dict(), filepath)
