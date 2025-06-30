# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Script for running Mask R-CNN model evaluation"""

import os
from os.path import join as opj
from datetime import datetime
import uuid
import itertools

import pandas as pd

import torch
import pytorch_lightning as pl

from ds.lightning_modules.mrcnn_system import MRCNNSystem
from ds.eval.infra import setup_infra
from ds.eval.common import (compute_metrics, save_images, get_targets,
                            get_loaders)
from ds.train.infra import parse_args
from ds.eval.common import score_threshold_with_matrix_nms


def get_ims(loader):
    loader.dataset.transform.transforms[0].subtracted_base = 5000
    images = []
    for batch in loader:
        images.append(batch["image"])
    return torch.cat(images)


def get_predictions(cf, loader, celltypes):
    pl_module = MRCNNSystem
    ckpt_path = cf["infra"]["log_dir"] + cf["infra"]["exp_name"] + "/" + cf[
        "eval"]["ckpt_path"]
    params = {"cf": cf, "num_it_per_ep": 0}
    model = pl_module.load_from_checkpoint(ckpt_path, **params)
    trainer = pl.Trainer(accelerator="auto",
                         devices=1,
                         logger=False,
                         inference_mode=True,
                         deterministic=False)

    predictions = trainer.predict(model, dataloaders=loader)

    preds_per_class = {
        i:
        list(
            itertools.chain(*[[{
                "boxes": (instance_pred["boxes"][instance_pred["labels"] ==
                                                 celltypes[i], :]),
                "masks":
                instance_pred["masks"][instance_pred["labels"] == celltypes[i],
                                       ...],
                "labels":
                instance_pred["labels"][instance_pred["labels"] ==
                                        celltypes[i]],
                "scores":
                instance_pred["scores"][instance_pred["labels"] ==
                                        celltypes[i]]
            } for instance_pred in batch["thresh_pred"]]
                              for batch in predictions]))
        for i in celltypes.keys()
    }

    images = []
    for batch in predictions:
        images.append(batch["image"])

    all_tumors = []
    for batch in predictions:
        all_tumors += batch["tumor"]

    all_paths = []
    for batch in predictions:
        all_paths += batch["path"]

    all_images = torch.cat(images)

    return all_images, preds_per_class, all_tumors, all_paths


def main():

    def get_exp_name(cf):
        time = datetime.now().strftime("%b%d-%H-%M-%S")
        return "_".join([uuid.uuid4().hex[:8], time, cf["infra"]["comment"]])

    (cf, _, _, results_dir, _, _, pred_dir,
     _) = setup_infra(parse_args(), get_exp_name)

    train_loader, val_loader = get_loaders(cf)

    celltypes = train_loader.dataset.sc_label_map
    del celltypes["n/a"]

    do_train = False # perform evaluation on training set

    if do_train:
        _, train_preds, train_tumors, _ = get_predictions(cf,
                                                          train_loader,
                                                          celltypes)
        torch.save(train_preds, opj(pred_dir, "train_nuclei_pred.pt"))
        train_targets = get_targets(train_loader, celltypes)
        torch.save(train_targets, opj(pred_dir, "train_targets.pt"))

    _, val_preds, val_tumors, _ = get_predictions(cf, val_loader, celltypes)
    val_images = get_ims(val_loader)
    torch.save(val_preds, opj(pred_dir, "val_nuclei_pred.pt"))
    val_targets = get_targets(val_loader, celltypes)
    torch.save(val_targets, opj(pred_dir, "val_targets.pt"))

    if do_train:
        metrics = compute_metrics(train_preds, train_targets, train_tumors,
                                  val_preds, val_targets, val_tumors,
                                  celltypes)
    else:
        metrics = compute_metrics(None, None, None, val_preds, val_targets,
                                  val_tumors, celltypes)
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(metrics)

    # performing confidence thresholding and nms
    for k in val_preds:
        for i in range(len(val_preds[k])):
            val_preds[k][i] = score_threshold_with_matrix_nms(
                val_preds[k][i], confidence_threshold=.5)
    save_images(pred_dir, val_images, val_preds, val_targets, celltypes)


if __name__ == "__main__":
    main()
