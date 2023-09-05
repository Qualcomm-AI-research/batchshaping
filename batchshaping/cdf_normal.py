# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Differentiable CDF for the Normal/Gaussian distribution"""

from typing import Union

import numpy as np
import torch


def get_normal_cdf(
    x_in: torch.Tensor,
    mean: Union[float, torch.Tensor],
    std: Union[float, torch.Tensor],
) -> torch.Tensor:
    """CDF of the Normal distribution

    :param x_in: Input Tensor
    :param mean: Mean of the distribution
    :param std: Variance of the distribution
    """
    norm_x = (x_in - mean) / (np.sqrt(2) * std)
    pcdf = 0.5 * (1 + torch.special.erf(norm_x))
    return pcdf
