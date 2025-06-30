# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Helper functions for model training infrastructure setup."""

import os
import json
import random
import logging
import argparse
import operator
import subprocess
from shutil import copy2
from functools import partial, reduce
from typing import Tuple, Dict, TextIO, Optional
import yaml

import numpy as np
import torch

import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


def modify_tune_cf(cf):
    taskid = cf["tune"]["taskid"]
    params = sorted(list(cf["tune"]["params"].keys()))
    options = [cf["tune"]["params"][p] for p in params]

    if cf["tune"]["diagonal_items"]:
        params_to_update = {p: o[taskid] for p, o in zip(params, options)}
    else:
        lengths = [len(cf["tune"]["params"][p]) for p in params]
        inds = np.unravel_index(taskid, lengths)
        params_to_update = {p: o[i] for p, o, i in zip(params, options, inds)}

    # https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
    for k in params_to_update:
        keys = k.split("/")
        parent_key = keys[:-1]
        final_key = keys[-1]
        val = params_to_update[k]
        reduce(operator.getitem, parent_key, cf)[final_key] = val

    return cf, f"tune{taskid}"


def read_cf(cf_fd: TextIO) -> Dict:
    if cf_fd.name.endswith(".json"):
        return json.load(cf_fd)
    elif cf_fd.name.endswith(".yaml") or cf_fd.name.endswith(".yml"):
        return yaml.load(cf_fd, Loader=yaml.FullLoader)


@pl.utilities.rank_zero.rank_zero_only
def config_loggers(exp_root):
    """Config logger for the experiments
    Sets string format and where to save.
    """

    logging_format_str = "[%(levelname)-s|%(asctime)s|%(name)s|" + \
        "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=logging_format_str,
                        datefmt="%H:%M:%S",
                        handlers=[
                            logging.FileHandler(
                                os.path.join(exp_root, 'train.log')),
                            logging.StreamHandler()
                        ],
                        force=True)
    logging.info("Exp root {}".format(exp_root))

    formatter = logging.Formatter(logging_format_str, datefmt="%H:%M:%S")
    logger = logging.getLogger("pytorch_lightning.core")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(exp_root, 'train.log')))
    for h in logger.handlers:
        h.setFormatter(formatter)


def setup_infra_light(cf_fd: TextIO, get_exp_name: callable):
    cf = read_cf(cf_fd)
    env_var = dict(os.environ)

    if "tune" in cf:
        if "SLURM_ARRAY_TASK_ID" in env_var:
            cf["tune"]["taskid"] = int(env_var["SLURM_ARRAY_TASK_ID"])
            no_tune_id = False
        else:
            no_tune_id = True
            cf["tune"]["taskid"] = 0

        cf, cmt_append = modify_tune_cf(cf)
    else:
        cmt_append = ""

    pl.seed_everything(cf["infra"]["seed"], workers=True)
    exp_root, model_dir, config_dir, code_dir, artifact_dir = \
        setup_output_dirs(cf, get_exp_name, cmt_append)

    # logging
    config_loggers(exp_root)
    if "tune" in cf:
        if no_tune_id:
            logging.warning("Tune status: Tune index None. Using default 0")
        else:
            logging.info(f"Tune status: Tune index {cf['tune']['taskid']}")
    else:
        logging.info("Tune status: Not a tune job")

    # config and environment variable
    env_var = dict(os.environ)
    with open(os.path.join(config_dir, "env.json"), "w") as fd:
        json.dump(env_var, fd, sort_keys=True, indent=4)
    with open(os.path.join(config_dir, "parsed_config.json"), "w") as fd:
        json.dump(cf, fd, sort_keys=True, indent=4)

    cp_config = partial(copy2, dst=config_dir)
    cp_config(cf_fd.name)

    copy_code_diff(code_dir)
    copy_slurm_script(env_var, cp_config)

    torch.cuda.empty_cache()

    return cf, cp_config, artifact_dir, model_dir, exp_root


@pl.utilities.rank_zero.rank_zero_only
def copy_slurm_script(env_var, cp_config):
    # copy slurm script if exists
    if 'SLURM_CONF' in env_var and 'SLURM_JOB_ID' in env_var:
        slurm_script_path = env_var['SLURM_CONF'].replace(
            'conf-cache/slurm.conf',
            'job%s/slurm_script' % env_var['SLURM_JOB_ID'])
        logging.debug(f"slurm script path {slurm_script_path}")
        if os.path.exists(slurm_script_path):
            try:
                cp_config(slurm_script_path)
            except:
                logging.error("Cannot save slurm script.")
        else:
            logging.error("Cannot find slurm script.")


def copy_code_diff(code_dir: str) -> None:
    with open(os.path.join(code_dir, 'head.txt'), 'w') as fd:
        subprocess.call(['git', 'rev-parse', 'HEAD'], stdout=fd, stderr=fd)

    with open(os.path.join(code_dir, 'git-status.txt'), 'w') as fd:
        subprocess.call(['git', 'status'], stdout=fd, stderr=fd)

    with open(os.path.join(code_dir, 'git-diff.txt'), 'w') as fd:
        subprocess.call(['git', 'diff'], stdout=fd, stderr=fd)

    with open(os.path.join(code_dir, 'pip-list.txt'), 'w') as fd:
        subprocess.call(['pip', 'list'], stdout=fd, stderr=fd)


# https://github.com/Lightning-AI/lightning/blob/7a1e0e801eac755b2da7b84ef3504a3e47b166c8/src/lightning_lite/utilities/rank_zero.py#L36
def get_rank() -> Optional[int]:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


def setup_output_dirs(cf: Dict, get_exp_name: callable, cmt_append: str):
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]

    if get_rank():
        exp_name = os.path.join(exp_name, "high_rank")

    instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, instance_name)

    model_dir = os.path.join(exp_root, 'models')
    config_dir = os.path.join(exp_root, 'config')
    code_dir = os.path.join(exp_root, 'code')
    artifact_dir = os.path.join(exp_root, 'artifacts')

    for dir_name in [model_dir, config_dir, code_dir, artifact_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    return exp_root, model_dir, config_dir, code_dir, artifact_dir

