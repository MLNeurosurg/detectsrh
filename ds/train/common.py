# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Helper functions for training run configuration."""

import os
import time
import logging
import time
import random
from typing import Dict, Any, Tuple, List
from functools import partial
import pynvml as nvml
import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import (StepLR, CosineAnnealingWarmRestarts)

import pytorch_lightning as pl

from ds.optim.cosine_schedule_warmup import get_cosine_schedule_with_warmup


def get_optimizer_func(opt_str):
    if opt_str == "sgd":
        opt_func = optim.SGD
    elif opt_str == "adam":
        opt_func = optim.Adam
    elif opt_str in {"adamw", "adamw_ld"}:
        opt_func = optim.AdamW
    else:
        raise ValueError(
            "Optimizer must be one of [sgd, adam, adamw, adamw_ld]")

    return opt_func


def get_scheduler(optimizer, cf, num_it_per_ep):
    if "scheduler" not in cf["training"]:
        return None

    sch_str = cf["training"]["scheduler"]["which"]
    sch_params = cf["training"]["scheduler"]["params"]
    required_params = {
        "step_lr": {"step_size", "step_unit", "gamma"},
        "cos_warm_restart": {"t0", "t0_unit", "t_mult", "eta_min"},
        "cos_linear_warmup": {"num_warmup_steps", "num_cycles"}
    }
    assert sch_params.keys() == required_params[sch_str]
    if sch_str == "step_lr":
        step_size = convert_epoch_to_iter(sch_params['step_unit'],
                                          sch_params['step_size'],
                                          num_it_per_ep)
        logging.info("step lr scheduler step size {}".format(step_size))
        scheduler = StepLR(optimizer,
                           step_size=step_size,
                           gamma=sch_params["gamma"])
    elif sch_str == "cos_linear_warmup":
        num_epochs = cf['training']['num_epochs']
        if sch_params['num_warmup_steps'] < 1:
            sch_params['num_warmup_steps'] = int(
                sch_params['num_warmup_steps'] * num_epochs * num_it_per_ep)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, sch_params['num_warmup_steps'],
            num_epochs * num_it_per_ep, sch_params['num_cycles'])
    elif sch_str == "cos_warm_restart":
        t0 = convert_epoch_to_iter(sch_params['t0_unit'], sch_params['t0'],
                                   num_it_per_ep)
        logging.info("cos_warm_restart lr scheduler t0 {}".format(t0))
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=t0,
                                                T_mult=sch_params["t_mult"],
                                                eta_min=sch_params["eta_min"])
    else:
        raise NotImplementedError(
            "Scheduler must be one of [step_lr, cos_warm_restart]")
    return scheduler


def convert_epoch_to_iter(unit, steps, num_it_per_ep):
    if unit == "epoch":
        return num_it_per_ep * steps  # per epoch
    elif unit == "iter":
        return steps
    else:
        NotImplementedError("unit must be one of [epoch, iter]")


def log_gpu_worker(writer, sleep_sec: int = 5) -> None:
    try:
        nvml.nvmlInit()
        ng = nvml.nvmlDeviceGetCount()
    except nvml.NVMLError as error:
        logging.error("GPU logging error")
        logging.error(error)
        return

    while True:
        start = time.time()
        try:
            temp = {}
            mem = {}
            util = {}

            for i in range(ng):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)

                temp[str(i)] = nvml.nvmlDeviceGetTemperature(
                    handle, nvml.NVML_TEMPERATURE_GPU)
                mem[str(i)] = nvml.nvmlDeviceGetMemoryInfo(
                    handle).used / 1e9  # in GB
                util[str(i)] = nvml.nvmlDeviceGetUtilizationRates(handle).gpu

            writer.add_scalars("gpu/temp".format(i), temp)
            writer.add_scalars("gpu/mem".format(i), mem)
            writer.add_scalars("gpu/util".format(i), util)

        except nvml.NVMLError as error:
            logging.error("GPU logging error")
            logging.error(error)

        duration = time.time() - start
        time.sleep(max(sleep_sec - duration, 0))


def setup_checkpoints(
    cf: Dict[str, Any], model_dir: str, num_it_per_ep: int
) -> Tuple[List[pl.callbacks.ModelCheckpoint], pl.Trainer]:

    ckpt_callable = partial(pl.callbacks.ModelCheckpoint,
                            dirpath=model_dir,
                            save_top_k=-1,
                            auto_insert_metric_name=False)
    ckpt_fmt = "ckpt-epoch{epoch}-step{step}-loss{val/sum_loss_manualepoch:.2f}"
    endofepoch_ckpt_fmt = "ckpt-endofepoch{epoch}-step{step}-loss{val/sum_loss_manualepoch:.2f}"

    if cf["valid"]["freq"]["unit"] == "epoch":
        if cf["valid"]["freq"]["interval"] < 1:
            iter_freq = cf["valid"]["freq"]["interval"] * num_it_per_ep
            if not iter_freq.is_integer():
                logging.warning("epoch not evenly divisible")
            ckpts = [
                ckpt_callable(every_n_train_steps=int(iter_freq),
                              filename=ckpt_fmt),
                ckpt_callable(every_n_epochs=1,
                              filename=endofepoch_ckpt_fmt,
                              save_on_train_epoch_end=True)
            ]
            trainer_ckpt_params = {
                "val_check_interval": int(iter_freq),
                "check_val_every_n_epoch": 1
            }
        else:
            ckpts = [
                ckpt_callable(every_n_epochs=cf["valid"]["freq"]["interval"],
                              filename=ckpt_fmt)
            ]
            trainer_ckpt_params = {
                "check_val_every_n_epoch": cf["valid"]["freq"]["interval"]
            }

    elif cf["valid"]["freq"]["unit"] == "iter":
        assert float(cf["valid"]["freq"]["interval"]).is_integer()
        ckpts = [
            ckpt_callable(every_n_train_steps=cf["valid"]["freq"]["interval"],
                          filename=ckpt_fmt)
        ]
        trainer_ckpt_params = {
            "val_check_interval": cf["valid"]["freq"]["interval"],
            "check_val_every_n_epoch": None
        }
    else:
        raise ValueError()

    return ckpts, trainer_ckpt_params


def get_num_worker():
    """Estimate number of cpu workers."""
    try:
        num_worker = len(os.sched_getaffinity(0))
    except Exception:
        num_worker = os.cpu_count()

    if num_worker > 1:
        return num_worker - 1
    else:
        return torch.cuda.device_count() * 4


def get_num_it_per_train_ep(train_len: int, cf: Dict) -> int:
    """Calcualtes the number of iteration in each epoch.

    Args:
        train_len: length of the training set
        cf: global config

    Returns:
        num_it_per_ep: number of iteration in each epoch
    """
    world_size = 1

    effective_batch_size = (
        cf["loader"]["direct_params"]["train"]["batch_size"] * world_size *
        cf["training"].get("accumulate_grad_batches", 1))

    num_it_per_ep = train_len // effective_batch_size

    if not cf["loader"]["direct_params"]["train"]["drop_last"]:
        num_it_per_ep += ((train_len % effective_batch_size) > 0)

    return {
        "num_it_per_ep": num_it_per_ep,
        "effective_batch_size": effective_batch_size
    }


def get_seed_worker_and_generator(seed=torch.initial_seed() % 2**32):

    def seed_worker(worker_id):  #TODO: can we remove worker_id and worker_seed
        worker_seed = seed
        np.random.seed(seed)
        random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return seed_worker, g
