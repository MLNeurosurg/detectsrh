# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Script to run Mask R-CNN model training"""

from datetime import datetime
import uuid
import logging
import pytorch_lightning as pl

from ds.train.common import setup_checkpoints, get_num_it_per_train_ep
from ds.train.infra import parse_args, setup_infra_light
from ds.datasets.utils import SCDSU
from ds.lightning_modules.mrcnn_system import MRCNNSystem


def main():
    # infrastructure
    def get_exp_name(cf):
        all_cmt = "sd{}_{}".format(cf["infra"]["seed"], cf["infra"]["comment"])
        time = datetime.now().strftime("%b%d-%H-%M-%S")
        return "_".join([uuid.uuid4().hex[:8], time, all_cmt])

    cf, cp_cf, artifact_dir, model_dir, exp_root = setup_infra_light(
        parse_args(), get_exp_name)

    # setup data
    dataloaders = SCDSU.setup_srh_data(cf, cp_cf, artifact_dir)
    training_params = get_num_it_per_train_ep(
        len(dataloaders["train"].dataset), cf)
    logging.info(f"actual num_it_per_ep {training_params}")

    if cf["model"]["name"] == "mrcnn":
        lm = MRCNNSystem(cf=cf, training_params=training_params)
    else:
        raise ValueError("model name should be in [mrcnn]")

    print(lm)

    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv"),
    ]

    # config callbacks
    ckpts, ckpt_params = setup_checkpoints(cf, model_dir,
                                           training_params["num_it_per_ep"])
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",
                                                  log_momentum=False)

    # create trainer
    trainer = pl.Trainer(accelerator="auto",
                         devices=1,
                         sync_batchnorm=True,
                         enable_progress_bar=True,
                         max_epochs=cf["training"]["num_epochs"],
                         callbacks=ckpts + [lr_monitor],
                         default_root_dir=exp_root,
                         logger=logger,
                         log_every_n_steps=min(
                             training_params["num_it_per_ep"], 10),
                         gradient_clip_val=0.5,
                         inference_mode=False,
                         **ckpt_params)
    trainer.fit(lm,
                train_dataloaders=dataloaders["train"],
                val_dataloaders=dataloaders["valid"])


if __name__ == "__main__":
    main()
