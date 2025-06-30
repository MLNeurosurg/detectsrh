# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Helper functions for running model evaluation."""

import os
import numpy as np
import torch
import itertools
from tqdm import tqdm

import pandas as pd
from PIL import Image

from skimage.measure import find_contours
import cv2

from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms.functional import adjust_contrast, adjust_brightness

import einops

from ds.utils.box_ops import box_cxcywh_to_xyxy
from ds.datasets.utils import SCDSU
from ds.datasets.srh_single_cell import SRHSingleCell
from ds.datasets.db_improc import get_transformations_v2


def get_loaders(cf):

    _, valid_xform = get_transformations_v2(**cf["data"]["augmentations"])

    if "train" in cf["data"]["direct_params"]:
        train_dset = SRHSingleCell(transform=valid_xform,
                                   **cf["data"]["direct_params"]["common"],
                                   **cf["data"]["direct_params"]["train"])

        train_loader = torch.utils.data.DataLoader(
            train_dset,
            collate_fn=SCDSU.sc_collate_fn,
            **cf["loader"]["direct_params"]["common"])
    else:
        train_loader = None

    valid_dset = SRHSingleCell(transform=valid_xform,
                               **cf["data"]["direct_params"]["common"],
                               **cf["data"]["direct_params"]["val"])
    val_loader = torch.utils.data.DataLoader(
        valid_dset,
        collate_fn=SCDSU.sc_collate_fn,
        **cf["loader"]["direct_params"]["common"])

    return train_loader, val_loader


def get_targets(loader, celltypes):
    return list(
        itertools.chain.from_iterable([[{
            i: {
                "boxes":
                box_cxcywh_to_xyxy(instance_tgt["boxes"][
                    instance_tgt["labels"] == celltypes[i], :]) * 299,
                "masks":
                instance_tgt["masks"][instance_tgt["labels"] == celltypes[i],
                                      ...],
                "labels":
                instance_tgt["labels"][instance_tgt["labels"] == celltypes[i]]
            }
            for i in celltypes.keys()
        } for instance_tgt in batch["target"]] for batch in loader]))


def compute_metric(pred, targets, iou_type="bbox"):
    if pred is not None:
        metric = MeanAveragePrecision(box_format="xyxy", iou_type=iou_type)
        metric.update(pred, targets)
        metric_result = metric.compute()
        metric_filtered_result = {
            k: metric_result[k].tolist()
            for k in ["map", "map_50", "map_75"]
        }
    else:
        metric_filtered_result = {k: 0 for k in ["map", "map_50", "map_75"]}
    metric_filtered_result = pd.DataFrame.from_dict(metric_filtered_result,
                                                    orient="index").T

    return metric_filtered_result


def compute_metrics(train_preds, train_targets, train_tumors, val_preds,
                    val_targets, val_tumors, celltypes):

    # List of tumor types
    tumor_types = [
        "schwannoma", "metastasis", "normal", "pituitary", "lgg", "meningioma",
        "hgg"
    ]

    # Define column headers
    columns = ["set", "tumor", "label", "iou_type", "map", "map_50", "map_75"]

    # Initialize DataFrames with column headers
    if train_preds is not None:
        train_results = pd.DataFrame(columns=columns)

    val_results = pd.DataFrame(columns=columns)

    # Loop through all labels and iou_types
    for label in tqdm(celltypes, desc="Metrics - celltypes"):  # cell type
        for iou_type in ["bbox", "segm"]:

            if train_preds is not None:
                train_result = compute_metric(
                    train_preds[label], [tl[label] for tl in train_targets],
                    iou_type)
                train_results.loc[len(train_results)] = [
                    "train", "general", label, iou_type
                ] + train_result.iloc[0].values.tolist()

            # Compute general metrics for train and val sets
            val_result = compute_metric(val_preds[label],
                                        [vl[label] for vl in val_targets],
                                        iou_type)

            # Add "general" label to train and val row titles
            val_results.loc[len(val_results)] = [
                "val", "general", label, iou_type
            ] + val_result.iloc[0].values.tolist()

            # Compute tumor-specific metrics for each tumor
            for tumor in tumor_types:
                # Create masks to filter based on the tumor type for train and val sets
                if train_preds is not None:
                    train_tumor_mask = [t == tumor for t in train_tumors]
                    train_preds_tumor = [
                        p for p, mask in zip(train_preds[label],
                                             train_tumor_mask) if mask
                    ]
                    train_targets_tumor = [
                        t[label]
                        for t, mask in zip(train_targets, train_tumor_mask)
                        if mask
                    ]
                    train_tumor_result = compute_metric(
                        train_preds_tumor, train_targets_tumor, iou_type)
                    train_results.loc[len(train_results)] = [
                        "train", tumor, label, iou_type
                    ] + train_tumor_result.iloc[0].values.tolist()

                val_tumor_mask = [t == tumor for t in val_tumors]
                val_preds_tumor = [
                    p for p, mask in zip(val_preds[label], val_tumor_mask)
                    if mask
                ]
                val_targets_tumor = [
                    v[label] for v, mask in zip(val_targets, val_tumor_mask)
                    if mask
                ]
                val_tumor_result = compute_metric(val_preds_tumor,
                                                  val_targets_tumor, iou_type)

                val_results.loc[len(val_results)] = [
                    "val", tumor, label, iou_type
                ] + val_tumor_result.iloc[0].values.tolist()

    if train_preds is not None:
        # Concatenate train and val results
        return pd.concat([train_results, val_results], ignore_index=True)
    else:
        return val_results


def matrix_nms(seg_masks,
               cate_labels,
               cate_scores,
               kernel='gaussian',
               sigma=2.0,
               sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) -
                                  inter_matrix)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = iou_matrix.max(0)
    compensate_iou = compensate_iou.expand(n_samples,
                                           n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix  #* label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def score_threshold_with_matrix_nms(output, confidence_threshold=0.50):

    if len(output["scores"]) == 0:
        for k in output:
            output[k] = output[k].detach().to("cpu")
        return output

    output["scores"] = matrix_nms(output["masks"].squeeze(1), output["labels"],
                                  output["scores"])

    confidence_boxes = torch.argwhere(output['scores'] > confidence_threshold)
    confidence_boxes = torch.squeeze(confidence_boxes).detach().to("cpu")

    for k in output:
        output[k] = output[k].detach().to("cpu")

    # merge nms + confidence indices
    nms_confidence_boxes = torch.from_numpy(
        np.intersect1d(confidence_boxes,
                       torch.where(output["labels"] > 0)[0]))

    for k in output.keys():
        output[k] = torch.index_select(output[k], 0, nms_confidence_boxes)

    return output


def viz_mask_v2(image, masks, color=(255, 255, 0)):
    image = einops.rearrange(image, "c h w -> h w c")
    im_ = np.array(image.contiguous(), dtype=np.uint8)
    masks = masks.squeeze(1)

    for m in masks:
        contours = find_contours(m.numpy(), 0.5)
        for verts in contours:
            im_ = cv2.polylines(im_, [verts[:, ::-1].astype(np.int32)], True,
                                color, 1)

    return torch.tensor(einops.rearrange(im_, "h w c -> c h w"))


def output_mask_to_images(base_img, mask, bbox, color=(255, 255, 0)):

    if len(bbox):
        mask_img = viz_mask_v2(base_img, mask, color=color)
        box_img = draw_bounding_boxes(base_img,
                                      bbox,
                                      colors="#ffff00",
                                      width=1)
    else:
        box_img = base_img.clone()
        mask_img = base_img.clone()

    return mask_img, box_img


def output_mask_to_images_duo(mask_img,
                              box_img,
                              mask,
                              bbox,
                              color=(255, 255, 0)):

    if len(bbox):
        mask_img = viz_mask_v2(mask_img, mask, color=color)
        box_img = draw_bounding_boxes(box_img, bbox, colors="#ffff00", width=1)
    return mask_img, box_img


def output_box_to_images(base_img, pred):
    box_img = base_img.clone()
    bbox = pred["boxes"]
    if len(bbox):
        box_img = draw_bounding_boxes(box_img, bbox, colors="#ffff00", width=1)

    return box_img


def save_images(pred_dir, val_images, val_preds, val_targets, celltypes):
    ann_color_map = {
        "cell_nuclei": (255, 255, 0),
        "cell_cytoplasm": (255, 0, 255),
        "red_blood_cell": (255, 0, 0),
        "macrophage": (255, 127, 0)
    }

    for label in tqdm(celltypes.keys(), desc="Visualization - celltypes"):

        os.makedirs(f"{pred_dir}/{label}", exist_ok=True)
        for i, (val_image, pred, target) in enumerate(
                zip(val_images, val_preds[label], val_targets)):
            base_img = (val_image * 255).to(torch.uint8)
            base_img = adjust_brightness(adjust_contrast(base_img, 2), 2)

            target_mask_img, target_box_img = output_mask_to_images(
                base_img, target[label]["masks"], target[label]["boxes"])

            pred_mask_img, pred_box_img = output_mask_to_images(
                base_img, pred["masks"], pred["boxes"])

            imgs = torch.cat([
                base_img, target_box_img, pred_box_img, target_mask_img,
                pred_mask_img
            ],
                             dim=2)
            imgs = imgs.permute(1, 2, 0)

            Image.fromarray(
                imgs.cpu().numpy()).save(f"{pred_dir}/{label}/out_{i}.png")

    val_preds = [dict(zip(val_preds, t)) for t in zip(*val_preds.values())]

    os.makedirs(f"{pred_dir}/all", exist_ok=True)
    for i, (val_image, pred,
            target) in tqdm(enumerate(zip(val_images, val_preds,
                                          val_targets))):
        base_img = (val_image * 255).to(torch.uint8)
        base_img = adjust_brightness(adjust_contrast(base_img, 2), 2)

        target_mask_img = base_img.clone()
        target_box_img = base_img.clone()
        pred_mask_img = base_img.clone()
        pred_box_img = base_img.clone()

        for label in celltypes.keys():
            target_mask_img, target_box_img = output_mask_to_images_duo(
                target_mask_img,
                target_box_img,
                target[label]["masks"],
                target[label]["boxes"],
                color=ann_color_map.get(label, (255, 255, 0)))

            pred_mask_img, pred_box_img = output_mask_to_images_duo(
                pred_mask_img,
                pred_box_img,
                pred[label]["masks"],
                pred[label]["boxes"],
                color=ann_color_map.get(label, (255, 255, 0)))

        imgs = torch.cat([
            base_img, target_box_img, pred_box_img, target_mask_img,
            pred_mask_img
        ],
                         dim=2)
        imgs = imgs.permute(1, 2, 0)

        Image.fromarray(imgs.cpu().numpy()).save(f"{pred_dir}/all/out_{i}.png")
