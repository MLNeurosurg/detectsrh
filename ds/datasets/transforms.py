# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Transformations that can be applied during SRH single cell dataset pre-processing"""

from typing import List, Tuple, Dict, Optional
from collections.abc import Sequence
import math
import random
import warnings

import torch
import torchvision
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import _interpolation_modes_from_int
from torchvision.transforms.transforms import _setup_size


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class GaussianNoise(torch.nn.Module):
    """object to add guassian noise to images."""

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def forward(self, tensor):
        var = random.uniform(self.min_var, self.max_var)
        noisy = tensor + torch.randn(tensor.size()) * var
        noisy = torch.clamp(noisy, min=0., max=1.)
        return noisy


class RandomResizedCropWithBoxes(torch.nn.Module):

    def __init__(self,
                 size,
                 p=0.5,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=torchvision.transforms.functional.
                 InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(
            size,
            error_msg="Please provide only two dimensions (h, w) for size.")
        self.p = p
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum.")
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float],
                   ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0],
                                                         scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1, )).item()
                j = torch.randint(0, width - w + 1, size=(1, )).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, target):
        if torch.rand(1).item() < self.p:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)

            img = F.resized_crop(img,
                                 i,
                                 j,
                                 h,
                                 w,
                                 self.size,
                                 self.interpolation,
                                 antialias=True)

            new_boxes = []
            img_width, img_height = F.get_image_size(img)
            for box in target["boxes"]:
                cx, cy, bw, bh = box

                # Convert to absolute coordinates
                cx_abs = cx * img_width
                cy_abs = cy * img_height
                bw_abs = bw * img_width
                bh_abs = bh * img_height

                # Adjust based on the crop
                cx_abs = (cx_abs - j) / w * self.size[1]
                cy_abs = (cy_abs - i) / h * self.size[0]
                bw_abs = bw_abs / w * self.size[1]
                bh_abs = bh_abs / h * self.size[0]

                # Convert back to normalized coordinates
                cx = cx_abs / self.size[1]
                cy = cy_abs / self.size[0]
                bw = bw_abs / self.size[1]
                bh = bh_abs / self.size[0]

                new_boxes.append([cx, cy, bw, bh])

            target["boxes"] = torch.tensor(new_boxes)

            xyxy = torchvision.ops.box_convert(target["boxes"], "cxcywh",
                                               "xyxy")
            xyxy[xyxy < 0] = 0
            xyxy[xyxy > 1] = 1
            target["boxes"] = torchvision.ops.box_convert(
                xyxy, "xyxy", "cxcywh")

            filt = ((xyxy[:, 2] - xyxy[:, 0]) *
                    (xyxy[:, 3] - xyxy[:, 1])) > 1.0e-6
            target["masks"] = target["masks"][filt, ...]
            target["area"] = target["area"][filt]
            target["boxes"] = target["boxes"][filt, ...]
            target["labels"] = target["labels"][filt]

        return img, target




class RandomHorizontalFlipWithMasks(T.RandomHorizontalFlip):

    def forward(self, img, target):
        if torch.rand(1).item() < self.p:
            img = F.hflip(img)

            bboxes = target["boxes"]
            bboxes[:, 0] = 0.9999 - bboxes[:, 0]
            target["boxes"] = bboxes
            target['masks'] = F.hflip(target['masks'])

        return img, target


class RandomVerticalFlipWithMasks(T.RandomHorizontalFlip):

    def forward(self, img, target):
        if torch.rand(1).item() < self.p:
            img = F.vflip(img)

            bboxes = target["boxes"]
            bboxes[:, 1] = 0.9999 - bboxes[:, 1]
            target["boxes"] = bboxes
            target['masks'] = F.vflip(target['masks'])

        return img, target


class RandomTransposeWithMasks(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, target):
        if torch.rand(1).item() < self.p:
            img = torch.transpose(img, -2, -1)

            target["boxes"] = target["boxes"][:, [1, 0, 3, 2]]
            target['masks'] = torch.transpose(target['masks'], -2, -1)

        return img, target


class RandomColorJitterWithMasks(T.ColorJitter):

    def __init__(self,
                 brightness: float = 0.4,
                 contrast: float = 0.4,
                 saturation: float = 0.4,
                 hue: float = 0.1,
                 p=0.5):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p

    def forward(self, img, target):
        if torch.rand(1).item() < self.p:
            img = super().forward(img)

        return img, target


class RandomGaussianBlurWithMasks(T.GaussianBlur):

    def __init__(self, kernel_size: 5, sigma: 1, p=0.5):
        super().__init__(kernel_size, sigma)
        self.p = p

    def forward(self, img, target):
        if torch.rand(1).item() < self.p:
            img = super().forward(img)

        return img, target


class RandomGaussianNoiseWithMasks(GaussianNoise):

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1, p=0.5):
        super().__init__(min_var, max_var)
        self.p = p

    def forward(self, img, target):
        if torch.rand(1).item() < self.p:
            img = super().forward(img)

        return img, target

