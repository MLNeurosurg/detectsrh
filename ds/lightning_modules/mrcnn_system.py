# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""PyTorch Lightning module for interfacing with Mask R-CNN model."""

import copy
import logging
from typing import Dict, Any, Optional

import torch
import torchvision
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ds.models.mask_rcnn import MaskRCNNPredictor
from ds.models.faster_rcnn import FastRCNNPredictor
from ds.utils.box_ops import box_cxcywh_to_xyxy
from ds.train.common import (get_optimizer_func, get_scheduler)
from ds.datasets.label_maps import (short_sc_label_reverse_map_mrcnn
                                             as short_sc_label_reverse_map,
                                             short_sc_labels_mrcnn as
                                             short_sc_labels,
                                             sc_label_map_mrcnn as
                                             sc_label_map)


def box_cxcywh_to_xyxy_safe(x):
    if len(x):
        return box_cxcywh_to_xyxy(x)
    return x


def get_model_instance_segmentation(num_classes, pretrained=True):

    # load a model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


class MRCNNSystem(pl.LightningModule):

    def __init__(self,
                 cf: Dict[str, Any],
                 training_params: Optional[Dict] = None):
        super().__init__()
        self.cf_ = cf
        self.training_params_ = training_params
        self.sc_label_map = {
            label: idx
            for (label, idx) in sc_label_map.items() if label not in cf["data"]
            ["direct_params"]["common"]["removed_labels"]
        }
        num_classes = len(self.sc_label_map)

        self.model = get_model_instance_segmentation(num_classes=num_classes)

        self.bbox_metric = MeanAveragePrecision(box_format='xyxy',
                                                iou_type="bbox",
                                                class_metrics=True)
        self.mask_metric = MeanAveragePrecision(box_format='xyxy',
                                                iou_type="segm",
                                                class_metrics=True)

    def forward(self, x, y=None):
        return self.model(x, y)

    @staticmethod
    def data_processor(batch):
        x = batch["image"]
        y = [{
            "masks":
            b["masks"],
            "labels":
            b["labels"],
            "boxes":
            box_cxcywh_to_xyxy_safe(b["boxes"].to(torch.float32) *
                                    (b["masks"].shape[-1] - 1))
        } for b in batch["target"]]

        return x, y

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        # get data
        x, y = self.data_processor(batch)

        # forward pass
        loss_dict = self.model(x, y)

        losses_reduced = sum(loss for loss in loss_dict.values())

        bs = x.shape[0]
        self.log("train/loss",
                 losses_reduced.detach().cpu(),
                 on_step=True,
                 batch_size=bs)

        return losses_reduced

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        # get data
        x, y = self.data_processor(batch)

        # forward pass
        # list of dicts {boxes, labels, scores, masks}
        outputs = self.model(x)

        filtered_outputs = []
        for output in outputs:
            # compute binary mask using confidence
            output = score_threshold(output)
            filtered_outputs.append(output)

        self.bbox_metric.update(filtered_outputs, y)
        self.mask_metric.update(filtered_outputs, y)

        if batch_idx % 20 == 0:
            self.log_images_tb(batch, filtered_outputs, y, "val")

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        # compute metrics
        bbox_metric = self.bbox_metric.compute()
        mask_metric = self.mask_metric.compute()

        for i, label_idx in enumerate(bbox_metric["classes"]):
            label_idx = label_idx.item()
            label_bbox_metric = bbox_metric["map_per_class"][i]
            label_mask_metric = mask_metric["map_per_class"][i]
            # log metrics
            self.log(f"val/bbox_map_{short_sc_label_reverse_map[label_idx]}",
                     label_bbox_metric,
                     on_epoch=True,
                     sync_dist=True)
            self.log(f"val/mask_map_{short_sc_label_reverse_map[label_idx]}",
                     label_mask_metric,
                     on_epoch=True,
                     sync_dist=True)

        # reset metrics
        self.bbox_metric.reset()
        self.mask_metric.reset()

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        # get data
        x, y = self.data_processor(batch)

        # forward pass
        outputs = self.model(x)
        for output in outputs:
            for k in output:
                output[k] = output[k].detach().cpu()

        orig_output = copy.deepcopy(outputs)
        filtered_outputs = []
        for output in outputs:
            # filter output by nms + confidence
            output = score_threshold(output)
            filtered_outputs.append(output)

        return {
            "image": batch["image"].squeeze(1, 2), 
            "path": batch["paths"],
            "tumor": batch["tumor"],
            "target": y,
            "pred": orig_output,
            "thresh_pred": filtered_outputs
        }

    def configure_optimizers(self):
        # if not training, no optimizer
        if "training" not in self.cf_:
            return None

        opt, sch = get_mrcnn_optimizer_scheduler(
            self.cf_,
            self.model,
            num_it_per_ep=self.training_params_["num_it_per_ep"],
            effective_bs=self.training_params_["effective_batch_size"])

        # get learn rate scheduler
        lr_scheduler_config = {
            "scheduler": sch,
            "interval": "step",
            "frequency": 1,
            "name": "lr"
        }

        return [opt], lr_scheduler_config

    @torch.inference_mode()
    @pl.utilities.rank_zero.rank_zero_only
    def log_images_tb(self, batch, outputs, gts, tv, idx=0):
        im = batch["image"][idx, ...].squeeze()
        im = im - im.min()
        im = im / im.max()
        im = (im * 255).to(torch.uint8)
        h, w = im.shape[-2], im.shape[-1]
        gt_labels = [short_sc_labels[x.item()] for x in gts[idx]["labels"]]
        gt_bbox = gts[idx]["boxes"].to(gts[idx]["boxes"].device)
        im_gt = torchvision.utils.draw_bounding_boxes(torch.clone(im), gt_bbox,
                                                      gt_labels)

        def generate_mask(masks):
            from functools import reduce
            if masks.numel() == 0:
                return torch.zeros(
                    (3, masks.shape[-2], masks.shape[-1])).to(self.device)
            else:
                combined_mask = reduce(torch.logical_or, masks)
                rgb_combined_mask = torch.stack([combined_mask] * 3) * 255
                return rgb_combined_mask

        curr_pred = outputs[idx]
        if len(curr_pred["boxes"]) == 0:
            im_pred = torch.clone(im)
            im_dict = {"im": im, "pred_box": im_pred, "gt_box": im_gt}

        else:
            pred_labels = curr_pred["labels"]  # gt class idx
            pred_labels = [short_sc_labels[x.item()]
                           for x in pred_labels]  # text lbls
            pred_bbox = curr_pred["boxes"].to(gt_bbox.device)
            im_pred = torchvision.utils.draw_bounding_boxes(
                torch.clone(im), pred_bbox, pred_labels)

            im_dict = {"im": im, "pred_box": im_pred, "gt_box": im_gt}

            for class_idx in [1, 2, 3]:  # nuclei, cyto, rbc
                class_label = short_sc_label_reverse_map[class_idx]

                gt_keep = torch.tensor([l for l in gts[idx]["labels"]
                                        ]) == class_idx

                im_dict[f"gt_{class_label}"] = generate_mask(
                    gts[idx]["masks"][gt_keep, ...])

                pred_keep = pred_labels == class_label
                im_dict[f"pred_{class_label}"] = generate_mask(
                    curr_pred["masks"][pred_keep, ...])

        tb_logger = self.trainer.loggers[0]
        for k in im_dict.keys():
            tb_logger.experiment.add_image(f"im/{tv}/{k}",
                                           im_dict[k],
                                           global_step=self.global_step)


def get_mrcnn_optimizer_scheduler(cf: Dict, model, num_it_per_ep: int,
                                  effective_bs: int):
    opt_str = cf["training"]["optimizer"]["which"]
    opt_params = cf["training"]["optimizer"]["params"]

    if cf['training']['optimizer'].get("scale_lr", False):
        assert "lr" in opt_params

        logging.info(f"scaling learn rate, was {opt_params['lr']}")
        opt_params["lr"] = opt_params["lr"] * effective_bs / 256
        logging.info(f"With effective batch size: {effective_bs}, " +
                     f"learn rate now {opt_params['lr']}")

    optimizer = get_optimizer_func(opt_str)(filter(lambda p: p.requires_grad,
                                                   model.parameters()),
                                            **opt_params)
    return optimizer, get_scheduler(optimizer, cf, num_it_per_ep)


def score_threshold(output, mask_confidence=0.50):
    output["masks"] = (output["masks"]
                       > mask_confidence).to(torch.uint8).squeeze(1)

    return output
