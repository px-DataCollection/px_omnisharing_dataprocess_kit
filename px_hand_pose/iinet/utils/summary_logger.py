#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import logging
import os
import json

from tensorboardX import SummaryWriter

from utils.utils import DictAverageMeter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        self.flogger = self.config_logger()
        self.avgMeter = DictAverageMeter()

    def output_json(self, filename, epoch):
        """
        Outputs metrics to a json file.
        Args:
            filepath: file path where we should save the file.
            print_running_metrics: should we print running metrics or the
                final average?
        """
        filepath = os.path.join(self.directory, filename)


        if epoch == 0:
            epoch_contents = {}
        elif not os.path.exists(filepath):
            epoch_contents = {}
        else:
            with open(filepath, "r") as jsonf:
                epoch_contents = json.load(jsonf)
        epoch_contents[f"epoch{epoch}"]={}
        for key, value in self.avgMeter.avg_data.items():
            epoch_contents[f"epoch{epoch}"][key] = value
        data = json.dumps(epoch_contents, indent=1)
        with open(filepath, "w", newline='\n') as jfile:
            jfile.write(data)

    def config_logger(self):
        # create logger with 'spam_application'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(self.directory, 'eval.log'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger
