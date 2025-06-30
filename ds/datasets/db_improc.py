# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

"""Helper functions for SRH single cell dataset pre-processing."""

import logging
from functools import partial
from typing import List, Tuple, Dict, Optional

import numpy as np
from tifffile import imread

import torch
from torch import Tensor
from torchvision.transforms import Normalize

from ds.utils.box_ops import box_xyxy_to_cxcywh
from ds.datasets.transforms import (
    Compose, RandomResizedCropWithBoxes, RandomHorizontalFlipWithMasks,
    RandomVerticalFlipWithMasks, RandomTransposeWithMasks,
    RandomColorJitterWithMasks, RandomGaussianBlurWithMasks,
    RandomGaussianNoiseWithMasks)


# Base augmentation modules
class GetThirdChannelWithMask(torch.nn.Module):
    """computes the third channel of srh image

    compute the third channel of srh images by subtracting ch3 and ch2. the
    channel difference is added to the subtracted_base.

    """

    def __init__(self,
                 mode: str = "three_channels",
                 subtracted_base: float = 0):  # 5000 / 65536.0):
        super().__init__()

        self.subtracted_base = subtracted_base
        aug_func_dict = {
            "three_channels": self.get_third_channel_,
        }
        if mode in aug_func_dict:
            self.aug_func = aug_func_dict[mode]
        else:
            raise ValueError("base_augmentation must be in " +
                             f"{aug_func_dict.keys()}")

    def get_third_channel_(self, im2: Tensor) -> Tensor:
        ch2 = im2[0, :, :]
        ch3 = im2[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base
        return torch.stack((ch1, ch2, ch3), dim=0)

    def forward(
        self,
        two_channel_image: Tensor,
        target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        args:
            two_channel_image: a 2 channel np array in the shape h * w * 2
            subtracted_base: an integer to be added to (ch3 - ch2)

        returns:
            a 1 or 3 channel tensor in the shape 1xhxw or 3xhxw
        """
        return self.aug_func(two_channel_image), target


class MinMaxChop(torch.nn.Module):

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_ = min_val
        self.max_ = max_val

    def forward(self, image: Tensor) -> Tensor:
        return image.clamp(self.min_, self.max_)


class MinMaxChopWithMask(MinMaxChop):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        return (super().forward(image), target)


class NormalizeWithMinMaxChopWithMask(Normalize):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mmc = MinMaxChop()

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        im = self.mmc(super().forward(image))
        return (im, target)


class BoxXYXYToCXCYWH(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if len(target["boxes"]) == 0:
            return image, target

        w, h = image.shape[-2] - 1, image.shape[-1] - 1
        target["boxes"] = box_xyxy_to_cxcywh(target["boxes"]) / torch.tensor(
            [w, h, w, h], dtype=torch.float32)
        return image, target


def get_base_aug_v2(get_third_channel_params,
                    inference_mode=False) -> List:

    u16_min, u16_max = (0, 0, 0), (65536, 65536, 65536)  # 2^16

    xform_list = [
        GetThirdChannelWithMask(**get_third_channel_params),
        NormalizeWithMinMaxChopWithMask(mean=u16_min, std=u16_max)
    ]

    if not inference_mode:
        xform_list.append(BoxXYXYToCXCYWH())

    return xform_list


def get_strong_aug(augs, p) -> List:
    callable_dict = {
        "random_resized_crop": partial(RandomResizedCropWithBoxes, p=p),
        "random_horizontal_flip": partial(RandomHorizontalFlipWithMasks, p=p),
        "random_vertical_flip": partial(RandomVerticalFlipWithMasks, p=p),
        "random_transpose": partial(RandomTransposeWithMasks, p=p),
        "random_color_jitter": partial(RandomColorJitterWithMasks, p=p),
        "random_gaussian_blur": partial(RandomGaussianBlurWithMasks, p=p),
        "random_gaussian_noise": partial(RandomGaussianNoiseWithMasks, p=p),
    }

    return [callable_dict[a["which"]](**a["params"]) for a in augs]


def get_transformations_v2(base_aug_cf,
                           train_strong_aug_cf,
                           val_strong_aug_cf,
                           inference_mode=False):
    if val_strong_aug_cf == "same":
        val_strong_aug_cf = train_strong_aug_cf

    train_xform = Compose(
        get_base_aug_v2(**base_aug_cf, inference_mode=inference_mode) +
        get_strong_aug(**train_strong_aug_cf))

    valid_xform = Compose(
        get_base_aug_v2(**base_aug_cf, inference_mode=inference_mode) +
        get_strong_aug(**val_strong_aug_cf))

    logging.info(f"train_xform\n{train_xform.transforms}")
    logging.info(f"valid_xform\n{valid_xform.transforms}")
    return train_xform, valid_xform


def process_read_srh(imp: str) -> Tensor:
    """Read in two channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A 2 channel torch Tensor in the shape 2 * H * W
    """

    # reference: https://github.com/pytorch/vision/blob/49468279d9070a5631b6e0198ee562c00ecedb10/torchvision/transforms/functional.py#L133
    return torch.from_numpy(imread(imp).astype(np.float32)).contiguous()

