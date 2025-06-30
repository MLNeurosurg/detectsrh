# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Learning rate scheduler for model optimization during training"""

import math
import logging

from torch import optim
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer: optim.Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float,
                                    last_epoch: int = -1):

    def lr_func(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    logging.info(f"num_warmup_steps: {num_warmup_steps}")
    logging.info(f"num_training_steps: {num_training_steps}")
    logging.info(f"num_cycles: {num_cycles}")

    if len(optimizer.param_groups) > 1:
        lr_funcs = [lr_func for _ in optimizer.param_groups]
    else:
        lr_funcs = lr_func

    return LambdaLR(optimizer, lr_lambda=lr_funcs, last_epoch=last_epoch)
