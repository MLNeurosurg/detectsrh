# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Helper functions for SRH single cell datasets / data loaders."""

import logging
from typing import Dict, Callable, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from ds.train.common import get_num_worker, get_seed_worker_and_generator
from ds.datasets.srh_single_cell import SRHSingleCell
from ds.datasets.db_improc import get_transformations_v2
from torch.utils.data import Dataset, RandomSampler, DataLoader
import pytorch_lightning as pl


class SCDatasetUtils():

    @staticmethod
    def copy_slide_files(cf: Dict[str, Any],
                         cp_cf: Optional[Callable]) -> None:
        if cp_cf:
            # copy slide files
            direct_params = cf["data"]["direct_params"]
            train_slides = direct_params.get("train", {}).get("slides_file")
            val_slides = direct_params.get("val", {}).get("slides_file")
            xmplr_slides = direct_params.get("xmplr", {}).get("slides_file")

            if train_slides: cp_cf(train_slides)
            if val_slides: cp_cf(val_slides)
            if xmplr_slides: cp_cf(xmplr_slides)

    @staticmethod
    def sc_collate_fn(batch: Dict):
        return {
            "image": torch.stack([i["image"] for i in batch]),
            "paths": [i["paths"] for i in batch],
            "target": [i["target"] for i in batch],
            "tumor": [i["tumor"] for i in batch]
        }

    @staticmethod
    def setup_srh_data(cf: Dict[str, Dict],
                       cp_cf: Optional[Callable] = None,
                       artifact_dir: Optional[str] = None,
                       eval_mode: bool = False,
                       eval_get_train: bool = False) -> Dict[str, DataLoader]:

        SCDSU.copy_slide_files(cf, cp_cf)

        train_xform, valid_xform = get_transformations_v2(
            **cf["data"]["augmentations"])

        train_params = {"transform": train_xform}
        valid_params = {"transform": valid_xform}

        dset_func = {"SRHSingleCell": SRHSingleCell}[cf["data"]["which"]]

        if (not eval_mode) or eval_get_train:
            train_dset = dset_func(**cf["data"]["direct_params"]["common"],
                                   **cf["data"]["direct_params"]["train"],
                                   **train_params)
        else:
            train_dset = None

        valid_dset = dset_func(**cf["data"]["direct_params"]["common"],
                               **cf["data"]["direct_params"]["val"],
                               **valid_params)


        if not eval_mode:
            loaders = SCDSU.get_dataloaders_(cf, train_dset, valid_dset)
        else:
            loaders = SCDSU.get_eval_dataloaders_(cf, train_dset, valid_dset)

        if cf["data"]["which"] == "SRHSingleCell":  # this is a hack
            for k in loaders:
                if loaders[k] is not None:
                    loaders[k].collate_fn = SCDSU.sc_collate_fn

        return loaders

    @staticmethod
    def get_dataloaders_(config: Dict, train_dset: Dataset,
                         valid_dset: Dataset) -> Dict[str, Dataset]:
        """Creates dataloader from datasets and config files."""

        sw, g = get_seed_worker_and_generator(seed=config["infra"]["seed"])

        train_loader_params = config["loader"]["direct_params"]["train"]
        val_loader_params = config["loader"]["direct_params"]["val"]

        config["loader"]["direct_params"]["common"].update({
            "worker_init_fn": sw,
            "generator": g
        })
        train_loader_params.update(config["loader"]["direct_params"]["common"])
        val_loader_params.update(config["loader"]["direct_params"]["common"])

        if (("val_sampler" in config["loader"])
                and (config["loader"]["val_sampler"]["num_samples"] > 0)):
            config["loader"]["direct_params"]["val"].update({
                "sampler":
                RandomSampler(valid_dset, **config["loader"]["val_sampler"])
            })

        if train_loader_params.get("num_workers") == "auto":
            train_loader_params["num_workers"] = get_num_worker()
        if val_loader_params.get("num_workers") == "auto":
            val_loader_params["num_workers"] = get_num_worker()

        logging.info(train_loader_params)
        logging.info(val_loader_params)

        return {
            "train": DataLoader(dataset=train_dset, **train_loader_params),
            "valid": DataLoader(dataset=valid_dset, **val_loader_params)
        }

    @staticmethod
    def get_eval_dataloaders_(config: Dict, train_dset: Dataset,
                              valid_dset: Dataset) -> Dict[str, DataLoader]:

        sw, g = get_seed_worker_and_generator(seed=config["infra"]["seed"])

        assert "train" not in config["loader"]["direct_params"]
        assert "val" not in config["loader"]["direct_params"]
        assert "valid" not in config["loader"]["direct_params"]

        dl_params = config["loader"]["direct_params"]["common"]
        dl_params.update({"worker_init_fn": sw, "generator": g})

        if dl_params.get("num_workers") == "auto":
            dl_params["num_workers"] = get_num_worker()

        if train_dset:
            train_loader = DataLoader(train_dset, **dl_params)
        else:
            train_loader = None  #TODO: Linter says this is unreachable. Can't you pass None for train_dset?

        valid_loader = DataLoader(valid_dset, **dl_params)
        logging.info(dl_params)

        return {"train": train_loader, "valid": valid_loader}


SCDSU = SCDatasetUtils
