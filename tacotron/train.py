# Base Imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from logzero import logger, logfile
import multiprocessing as mp
import argparse
import os
import time

# Internal Imports
from network import Tacotron
import hyperparams as hp
from data import get_dataset, DataLoader, collate_fn, get_param_size

# Imports
from torch import optim
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

try: # older versions of torch don't recognise mps
    use_mps = torch.backends.mps.is_available() & torch.backends.mps.is_built()
except:
    use_mps = False

def main(args):

    # Get dataset
    dataset = get_dataset()

    # Construct model
    if use_mps:
        logger.info('\nRunning on M1')
        device = torch.device('mps')
        model = nn.DataParallel(Tacotron().to(device=device))
    elif use_cuda:
        logger.info('Using CUDA')
        device = torch.device('cuda')
        model = nn.DataParallel(Tacotron().cuda())
    else:
        logger.info('No Acceleration is being used')
        model = Tacotron()

    # Make optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar'% args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("\n--------model restored at step %d--------\n" % args.restore_step)

    except:
        logger.info("\n--------Start New Training--------\n")

    # Set training mode
    model = model.train()

    # Make checkpoint directory if not exists
    if not os.path.exists(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)

    # Decide loss function
    if use_mps:
        criterion = nn.L1Loss().to(device=device)
        logger.info('Criterion on M1')
    elif use_cuda:
        criterion = nn.L1Loss().cuda()
        logger.info('Criterion on cuda')
    else:
        criterion = nn.L1Loss()

    # Loss for frequency of human register
    n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    for epoch in range(hp.epochs):

        logger.info('\n-----Running Epoch {}-----'.format(epoch))

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn, drop_last=True, 
                                pin_memory=True, num_workers=args.num_workers)

        for i, data in enumerate(dataloader):

            logger.info('\n-----Running Epoch-Batch {}:{}-----'.format(epoch, i))

            current_step = i + args.restore_step + epoch * len(dataloader) + 1
            optimizer.zero_grad()

            # Make decoder input by concatenating [GO] Frame
            try:
                mel_input = np.concatenate((np.zeros([args.batch_size, hp.num_mels, 1], dtype=np.float32),data[2][:,:,1:]), axis=2)
                print(mel_input)
            except:
                raise TypeError("not same dimension")

            if use_mps:
                characters = Variable(torch.from_numpy(data[0]).type(torch.LongTensor), requires_grad=False).to(device)
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.FloatTensor), requires_grad=False).to(device)
                mel_spectrogram = Variable(torch.from_numpy(data[2]).type(torch.FloatTensor), requires_grad=False).to(device)
                linear_spectrogram = Variable(torch.from_numpy(data[1]).type(torch.FloatTensor), requires_grad=False).to(device)

            elif use_cuda:
                characters = Variable(torch.from_numpy(data[0]).type(torch.cuda.LongTensor), requires_grad=False).cuda()
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
                mel_spectrogram = Variable(torch.from_numpy(data[2]).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
                linear_spectrogram = Variable(torch.from_numpy(data[1]).type(torch.cuda.FloatTensor), requires_grad=False).cuda()

            else:
                characters = Variable(torch.from_numpy(data[0]).type(torch.LongTensor), requires_grad=False)
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.FloatTensor), requires_grad=False)
                mel_spectrogram = Variable(torch.from_numpy(data[2]).type(torch.FloatTensor), requires_grad=False)
                linear_spectrogram = Variable(torch.from_numpy(data[1]).type(torch.FloatTensor), requires_grad=False)

            # Forward
            mel_output, linear_output = model.forward(characters, mel_input)

            # Calculate loss
            logger.info('Calculating Loss')
            mel_loss = criterion(mel_output, mel_spectrogram)
            linear_loss = torch.abs(linear_output-linear_spectrogram)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:,:n_priority_freq,:])
            loss = mel_loss + linear_loss
            logger.info('Loss: {}'.format(loss))
            loss = loss.to(device)

            start_time = time.time()

            # Calculate gradients
            loss.backward()

            # clipping gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            time_per_step = time.time() - start_time

            if current_step % hp.log_step == 0:
                logger.info("time per step: %.2f sec" % time_per_step)
                logger.info("At timestep %d" % current_step)
                logger.info("linear loss: %.4f" % linear_loss.item())
                logger.info("mel loss: %.4f" % mel_loss.item())
                logger.info("total loss: %.4f" % loss.item())

            if current_step % hp.save_step == 0:
                save_checkpoint({'model':model.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar' % current_step))
                logger.info("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer

if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logfile('logs/training.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=0)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--num_workers', type=int, help='Num Workers', default=6)
    args = parser.parse_args()
    logger.info('Batch Size: {}'.format(args.batch_size))
    logger.info('Num Workers: {}'.format(args.num_workers))

    try: # for logging on artemis
        logger.info('Configuration:\n\tGPU:\t{}x {}\n\tCPU Count:\t{}'.format(
            torch.cuda.device_count(), torch.cuda.get_device_name(0), mp.cpu_count()))
    except:
        logger.warning('Training not completed on CUDA compatible device.')
        
    main(args)

