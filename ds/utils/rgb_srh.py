# Copyright (c) 2025 University of Michigan. All rights reserved.
# Licensed under the MIT License. See LICENSE for license information.

import numpy as np
from typing import List
import torch
from torchvision.transforms.functional import adjust_contrast, adjust_brightness


class SRHRGBToolbox():

    @staticmethod
    def percentile_rescaling_(array: np.ndarray,
                              percentile_clip: int = 3) -> np.ndarray:
        """Rescale one channel of an image based on a percentile clipping
        NOTE: percentile clip applies to the UPPER percentile. The lower
        percentile is fixed at 3 percentile to avoid overly dark images.
        """
        p_low, p_high = np.percentile(array, (3, 100 - percentile_clip))
        return (array.clip(min=p_low, max=p_high) - p_low) / (p_high - p_low)

    @staticmethod
    def histogram_equalization_(im: np.ndarray, bits=8):
        """Rescale one channel of image using histogram equalization
        Bits is desired number of bits in output (number of bins in the
        histogram). It can be a decimal, and the number of bins is floored
        """
        # adapted from https://ianwitham.wordpress.com/tag/histogram-equalization/
        num_bins = np.floor(2**bits).astype(np.uint16)
        hist, bins = np.histogram(im.flatten(), num_bins)
        hist[0] = 0

        cdf = hist.cumsum()
        cdf = (num_bins - 1) * cdf / cdf[-1]  # normalize

        im2 = np.interp(im.flatten(), bins[:-1], cdf)
        return np.array(im2).reshape(im.shape)

    @staticmethod
    def viz_rescale_perct(im: torch.Tensor,
                          scale: List[float] = [0.4, 1, 1]) -> np.ndarray:
        im_numpy = im.transpose(0, -1).numpy()
        stacked = np.dstack([
            SRHRGBToolbox.percentile_rescaling_(im_numpy[..., 0]).T * scale[0],
            SRHRGBToolbox.percentile_rescaling_(im_numpy[..., 1]).T * scale[1],
            SRHRGBToolbox.percentile_rescaling_(im_numpy[..., 2]).T * scale[2],
        ])
        return (255 * stacked).astype(np.uint8)

    @staticmethod
    def viz_rescale_hist(im: torch.Tensor,
                         bits: List[float] = [7.5, 8, 8]) -> np.ndarray:
        im_numpy = im.transpose(0, -1).numpy()
        stacked = np.dstack([
            SRHRGBToolbox.histogram_equalization_(im_numpy[..., 0], bits[0]).T,
            SRHRGBToolbox.histogram_equalization_(im_numpy[..., 1], bits[1]).T,
            SRHRGBToolbox.histogram_equalization_(im_numpy[..., 2], bits[2]).T,
        ])
        return stacked.astype(np.uint8)

    @staticmethod
    def viz_rescale_hardcode(im: torch.Tensor,
                             contrast_factor=2,
                             brightness_factor=3) -> np.ndarray:
        """Rescale SRH images using hard coded contrast and brightness adjust"""
        adjusted = adjust_brightness(adjust_contrast(im, contrast_factor),
                                     brightness_factor)
        return adjusted.numpy().swapaxes(0, -1)
