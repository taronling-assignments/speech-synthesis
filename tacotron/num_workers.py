import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Internal Imports
import hyperparams as hp
from data import get_dataset, DataLoader, collate_fn, get_param_size

# Testing 
import os
from logzero import logger, logfile
from time import time
import multiprocessing as mp

import torch


def main():

    dataset = get_dataset()
    batch_size = 32
    logger.info(f'Batch Size: {batch_size}')

    # Loss for frequency of human register
    n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    for num_workers in range(2, 16, 2):  
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=collate_fn, drop_last=True, 
                                pin_memory=True, num_workers=num_workers)
        start = time()
        for epoch in range(1, 2):
            print(f'epoch {epoch}')
            for i, data in enumerate(train_loader, 0):
                print(i)
                if i > 4: 
                    break
                pass
        end = time()
        logger.info("Finish with:{} second, num_workers={}, batch_size={}".format(end - start, num_workers, batch_size))


if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logfile('logs/num_workers.txt')
    logger.info('Configuration:\n\tGPU:\t{}x {}\n\tCPU Count:\t{}'.format(
        torch.cuda.device_count(), torch.cuda.get_device_name(0), '?'))
    main()

