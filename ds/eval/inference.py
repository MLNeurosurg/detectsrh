# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Script for running Mask R-CNN model inference on data subset"""

import os
from os.path import join as opj
from glob import glob

import json
import uuid
import numpy as np
from pathlib import Path
from image_labelling_tool import labelling_tool

import torch

from ds.lightning_modules.mrcnn_system import get_model_instance_segmentation
from ds.datasets.db_improc import get_transformations_v2
from ds.datasets.db_improc import process_read_srh
from ds.train.infra import read_cf, parse_args
from ds.eval.common import score_threshold_with_matrix_nms
from ds.datasets.label_maps import sc_labels_mrcnn


def get_model(ckpt_path, num_classes=5, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model = get_model_instance_segmentation(num_classes=num_classes)
    model.load_state_dict({
        k.removeprefix("model."): ckpt["state_dict"][k]
        for k in ckpt["state_dict"]
    })
    model.to(device)
    model.eval()
    return model


def get_xform(augmentation_cf=None):
    if augmentation_cf is None:
        augmentation_cf = {
            'base_aug_cf': {
                'get_third_channel_params': {
                    'mode': 'three_channels',
                    'subtracted_base': 5000
                }
            },
            'train_strong_aug_cf': {
                'p': '0,',
                'augs': []
            },
            'val_strong_aug_cf': 'same'
        }
    augs, _ = get_transformations_v2(inference_mode=True, **augmentation_cf)
    return augs


def masks_to_json(target, label_list, mask_threshold=0.80):
    """
    Converts masks and associated labels to a json dict for 
    one image instance in django-labeller format. 
    
    :param target: dict containing `masks` and `labels` 
    :param label_list: list with str label for each corresponding index
    :param mask_threshold: threshold above which to generate binary mask
    :return: json dict in django-labeller format
    
    """
    label_objects = []

    masks_fun = target['masks'].cpu().numpy()
    masks_fun = masks_fun.reshape(masks_fun.shape[0], 300, 300)
    masks_fun = np.where(masks_fun > mask_threshold, 1,
                         0)  # threshold defined here

    for label, mask in zip(target['labels'], masks_fun):
        regions = labelling_tool.PolygonLabel.mask_image_to_regions_cv(
            mask, sort_decreasing_area=True)

        # if no regions can be generated, skip
        if len(regions) == 0:
            continue

        label_object = labelling_tool.PolygonLabel(
            regions=[regions[0].tolist()],
            object_id=str(uuid.uuid4()),
            classification=label_list[label],
            source='auto:maskrcnn',
            anno_data=None)

        label_objects.append(label_object.to_json())

    return label_objects


def save_masks_to_json(target,
                       label_list,
                       filename=None,
                       mask_threshold=0.80,
                       save_path='.'):
    """
    Converts masks and associated labels to a json dict for 
    one image instance and saves in json file comptabile with 
    django-labeller tool 
    
    :param target: dict containing `masks` and `labels` 
    :param label_list: list with str label for each corresponding index
    :param filename: 'str' name of original SRH image, will assume .tif extension
    :param mask_threshold: threshold used to generate binary mask
    :param save_path: str or Path object pointing to save directory
    :return: no return object, saves json to file in `save_path` directory 
    
    """
    if filename is None:
        raise RuntimeError('Filename is required')

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    image_annotation = masks_to_json(target, label_list, mask_threshold)

    with open(save_path / f'{filename}__labels.json', 'w') as outfile:
        json.dump(image_annotation, outfile, indent=2)


def main():
    cf = read_cf(parse_args())

    # Get model
    model = get_model(cf["ckpt_path"], len(cf["classes"]), cf["device"])

    # get augmentation for model inference
    aug = get_xform(cf["augmentation"])

    # Get image list
    image_list = glob(opj(cf["inference_dir"], "*.tif"))

    # Read and process image
    raw_ims = [process_read_srh(i) for i in image_list]
    ims = [aug(i, {})[0] for i in raw_ims]

    # Inference on image
    results = []
    with torch.inference_mode():
        for im_b in torch.split(torch.stack(ims), cf["inference_batch_size"]):
            results_i = model(im_b.to(cf["device"]))
            results.extend([{
                k: j[k].detach().to("cpu")
                for k in j
            } for j in results_i])

    results = [
        score_threshold_with_matrix_nms(
            r, confidence_threshold=cf["confidence"])
        for r in results
    ]
    print(f"Finished inference on {len(results)} images")

    # Save results
    os.makedirs(cf["out_dir"], exist_ok=True)
    torch.save(results, opj(cf["out_dir"], "results.pt"))

    annotations_dir_path = opj(cf["out_dir"], "labels")
    os.makedirs(annotations_dir_path, exist_ok=True)
    for r, im_path in zip(results, image_list):
        filename = im_path.split("/")[-1].split(".")[0]
        save_masks_to_json(r,
                           sc_labels_mrcnn,
                           filename=filename,
                           mask_threshold=0.5,
                           save_path=annotations_dir_path)


if __name__ == "__main__":
    main()
