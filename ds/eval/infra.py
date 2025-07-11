# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Helper functions for model evaluation infrastructure setup"""

import os
import json
import logging
from shutil import copy2
from functools import partial
from typing import Tuple, Dict, TextIO
import argparse
import gzip
from os.path import join as opjoin

import torch
import pytorch_lightning as pl

from ds.train.infra import (read_cf, modify_tune_cf, copy_code_diff)


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    if "ckpt_path" in cf["eval"]:
        instance_name = cf["eval"]["ckpt_path"].split("/")[0]

    eval_instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, instance_name, "evals",
                            eval_instance_name)

    # generate needed folders, evals will be embedded in experiment folders
    pred_dir = os.path.join(exp_root, 'predictions')
    config_dir = os.path.join(exp_root, 'config')
    code_dir = os.path.join(exp_root, 'code')
    artifact_dir = os.path.join(exp_root, 'artifacts')
    results_dir = os.path.join(exp_root, 'results')
    for dir_name in [
            pred_dir, config_dir, code_dir, artifact_dir, results_dir
    ]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # if there is a previously generated prediction, also return the
    # prediction filename so we don't have to predict again
    if cf["eval"].get("eval_predictions", None):
        other_eval_instance_name = cf["eval"]["eval_predictions"]
        pred_dir = os.path.join(log_root, exp_name, instance_name, "evals",
                                other_eval_instance_name, "predictions")
        pred_fname = {
            "train": os.path.join(pred_dir, "train_predictions.pt.gz"),
            "val": os.path.join(pred_dir, "val_predictions.pt.gz"),
            "xmplr": os.path.join(pred_dir, "xmplr_predictions.pt.gz")
        }
    else:
        pred_fname = None

    return (exp_root, config_dir, pred_dir, code_dir, artifact_dir,
            results_dir, partial(copy2, dst=config_dir), pred_fname)


def setup_infra(
        cf_fd: TextIO,
        get_exp_name: callable) -> Tuple[Dict, torch.device, callable, str]:

    cf = read_cf(cf_fd)
    env_var = dict(os.environ)

    if "tune" in cf:
        if "SLURM_ARRAY_TASK_ID" in env_var:
            cf["tune"]["taskid"] = int(env_var["SLURM_ARRAY_TASK_ID"])
        else:
            cf["tune"]["taskid"] = 0

        cf, cmt_append = modify_tune_cf(cf)
    else:
        cmt_append = ""

    (exp_root, config_dir, pred_dir, code_dir, artifact_dir, results_dir,
     cp_config, pred_fname) = setup_eval_paths(cf, get_exp_name, cmt_append)

    # logging config
    logging.basicConfig(
        level=logging.INFO,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(exp_root, 'eval.log')),
            logging.StreamHandler()
        ])
    logging.info("Exp root {}".format(exp_root))

    if "tune" in cf:
        if cf["tune"].get("taskid", None) is not None:
            logging.info("Tune index {}".format(cf["tune"]["taskid"]))
        else:
            logging.warning("Tune index None. Using default 0")

    pl.seed_everything(cf["infra"]["seed"], workers=True)

    # config + code
    with open(os.path.join(config_dir, "env.json"), "w") as fd:
        json.dump(env_var, fd, sort_keys=True, indent=4)
    with open(os.path.join(config_dir, "parsed_config.json"), "w") as fd:
        json.dump(cf, fd, sort_keys=True, indent=4)

    cp_config(cf_fd.name)
    copy_code_diff(code_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    logging.info("Using {}".format(device))

    return (cf, device, artifact_dir, results_dir, cp_config, exp_root,
            pred_dir, pred_fname)


def setup_eval_module_standalone_infra(get_train_pred=True,
                                       get_xmplr_pred=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    parser.add_argument('-e',
                        '--exp_name',
                        type=str,
                        required=False,
                        help='experiment name')
    parser.add_argument('-t',
                        '--train',
                        type=str,
                        required=False,
                        help='train instance')
    parser.add_argument('-v',
                        '--eval',
                        type=str,
                        required=False,
                        help='eval instance')

    args = parser.parse_args()

    config = read_cf(args.config)
    if "tune" in config:
        env_var = dict(os.environ)
        if "SLURM_ARRAY_TASK_ID" in env_var:
            config["tune"]["taskid"] = int(env_var["SLURM_ARRAY_TASK_ID"])
        else:
            config["tune"]["taskid"] = 0

        config, _ = modify_tune_cf(config)

    logging.basicConfig(
        level=logging.INFO,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Standalone eval module")

    pl.seed_everything(config["infra"]["seed"])

    if args.exp_name:
        config["infra"]["exp_name"] = args.exp_name

    if args.train:
        config["eval"]["ckpt_path"] = args.train

    if args.eval:
        config["eval"]["eval_predictions"] = args.eval

    logging.info("Loading train predictions")
    if get_train_pred:
        with gzip.open(
                opjoin(config["infra"]["log_dir"], config["infra"]["exp_name"],
                       config["eval"]["ckpt_path"].split("/")[0], "evals",
                       config["eval"]["eval_predictions"], "predictions",
                       "train_predictions.pt.gz")) as fd:
            train_pred = torch.load(fd)
        logging.info("Loading train predictions - done")
    else:
        train_pred = None
        logging.info("Loading train predictions - skipped")

    logging.info("Loading val predictions")
    with gzip.open(
            opjoin(config["infra"]["log_dir"], config["infra"]["exp_name"],
                   config["eval"]["ckpt_path"].split("/")[0], "evals",
                   config["eval"]["eval_predictions"], "predictions",
                   "val_predictions.pt.gz")) as fd:
        val_pred = torch.load(fd)
    logging.info("Loading val predictions - done")

    logging.info("Loading xmplr predictions")
    if get_xmplr_pred:
        with gzip.open(
                opjoin(config["infra"]["log_dir"], config["infra"]["exp_name"],
                       config["eval"]["ckpt_path"].split("/")[0], "evals",
                       config["eval"]["eval_predictions"], "predictions",
                       "xmplr_predictions.pt.gz")) as fd:
            xmplr_pred = torch.load(fd)
        logging.info("Loading xmplr predictions - done")
    else:
        xmplr_pred = None
        logging.info("Loading xmplr predictions - skipped")

    preds = {"train": train_pred, "val": val_pred, "xmplr": xmplr_pred}

    out_dir = config["eval"]["ckpt_path"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.info("Standalone eval infra - done")
    return config, out_dir, preds
