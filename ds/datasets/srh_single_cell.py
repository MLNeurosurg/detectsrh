# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""SRH single cell dataset class used for object detection and segmentation."""

import numpy as np
import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from image_labelling_tool import labelled_image

from ds.datasets.db_improc import process_read_srh
from ds.datasets.label_maps import sc_label_map_mrcnn


class SRHSingleCell(Dataset):
    """
    SRH RGB images annotated using the Django Labeller tool
    for segmented cell nuclei. Reads directory containing images
    and corresponding json labels containing Polygon label type.
    Only works with .tif images.
    """

    def __init__(self,
                 data_root,
                 slides_file,
                 folds=[],
                 rgb=False,
                 transform=None,
                 which_label_map="mrcnn",
                 removed_labels=[]):
        self.root = Path(data_root)
        self.transform = transform
        self.rgb = rgb

        labelled_images = labelled_image.LabelledImage.for_directory(
            self.root / 'images',
            labels_dir=(self.root / 'labels'),
            image_filename_patterns=['*.tif'])

        self.df = pd.read_csv(slides_file)
        if len(folds) != 0:
            self.df = self.df[self.df['fold'].isin(folds)]
        data_subset = set(self.df["image"])

        self.df = self.df.set_index("image")
        self.labelled_images = []
        for limg in labelled_images:
            if limg.image_source.local_path.name in data_subset:
                self.labelled_images.append(limg)

        if which_label_map == "mrcnn":
            self.sc_label_map = sc_label_map_mrcnn
        else:
            raise ValueError()

        assert all(label in self.sc_label_map.keys()
                   for label in removed_labels)
        labels = [
            l for (l, _) in self.sc_label_map.items()
            if l not in removed_labels
        ]

        self.sc_label_map = {label: idx for (idx, label) in enumerate(labels)}

    def __getitem__(self, idx):

        # load source image
        img = process_read_srh(
            self.labelled_images[idx].image_source.local_path)
        # filename
        filename = self.labelled_images[idx].image_source.local_path.name

        # load labels
        img_labels = self.labelled_images[idx].labels

        # load only segmentation masks and labels for each mask
        masks, mask_cls = img_labels.render_label_instances(
            label_classes=self.sc_label_map,
            image_shape=(300, 300),
            multichannel_mask=True)

        # convert (H, W, instance) to (instance, H, W)
        masks = masks.transpose(2, 0, 1)

        # get bounding box coordinates for each mask
        num_objs = len(mask_cls)
        boxes = []
        remove_indices = []  # keep track of invalid masks/boxes
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # check if valid mask/box
            if (xmax - xmin > 0) and (ymax - ymin > 0):
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                remove_indices.append(i)

        # remove invalid indices
        if len(remove_indices) > 0:
            mask_cls = np.delete(mask_cls, remove_indices)
            masks = np.delete(masks, remove_indices, axis=0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if len(boxes) != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.empty(0, 4)
            area = torch.empty(0, 1)

        # convert labels
        labels = torch.tensor(mask_cls, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        tumor_class = self.df.loc[
            self.labelled_images[idx].image_source.local_path.name]["class"]

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        if self.transform is not None:
            img, target = self.transform(img, target)

        return {
            "image": img,
            "target": target,
            "paths": [filename],
            "tumor": tumor_class
        }

    def __len__(self):
        return len(self.labelled_images)
