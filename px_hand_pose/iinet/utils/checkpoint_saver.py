#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import glob
import os

import torch


def save_checkpoint(epoch, model, optimizer, prev_best, checkpoint_saver, best):
    """
    Save current state of training
    """

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_pred': prev_best
    }
    if best:
        checkpoint_saver.save_checkpoint(checkpoint, 'model.pth.tar', write_best=False)
    else:
        checkpoint_saver.save_checkpoint(checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.logdir, args.dataset, args.expname)
        self.summary_dir = os.path.join(self.directory, 'summary')
        self.model_dir = os.path.join(self.directory, 'models')

        os.makedirs(self.summary_dir,exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.save_experiment_config()

    def save_checkpoint(self, state, filename='model.pth.tar', write_best=True):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.model_dir, filename)
        torch.save(state, filename)

        best_pred = state['best_pred']
        if write_best:
            with open(os.path.join(self.model_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

    def save_experiment_config(self):
        with open(os.path.join(self.directory, 'parameters.txt'), 'w') as file:
            config_dict = vars(self.args)
            for k in vars(self.args):
                file.write(f"{k}={config_dict[k]} \n")
