# --------------------------------------------------------
# [Logger] Log information to both console and file
# Copyright (c) 2021 Moffett.AI
# Licensed under Moffett.AI
# Written by CW
# --------------------------------------------------------

import os
import logging

from termcolor import colored
from functools import lru_cache


class Logger(logging.Logger):
    """
        Logger module for recording logs to both console & file.
        Use 'lru_cache' for avoiding duplicated instantiate.
        This module support distributed nodes, each process will log its information to
        its unique file.
    """
    @lru_cache
    def __init__(self, output_dir, dist_rank=0, name='', level=logging.DEBUG):
        super().__init__(name, level=level)

        # Create console handler for master process
        if not dist_rank:
            # Colored console outputs
            color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
            console_formatter = logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S')

            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)

            self.addHandler(console_handler)
        
        fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
        file_formatter = logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')

        os.makedirs(output_dir, exist_ok=True)
        # Each process own its unique log file
        self.log_file = os.path.join(output_dir, f'rank{dist_rank}.log')

        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)

        self.addHandler(file_handler)
