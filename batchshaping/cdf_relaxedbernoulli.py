# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Differentiable CDF for the Relaxed Bernoulli distribution (aka Binary Concrete distribution)"""

from typing import Union

import torch
from scipy.special import logit as logit_scipy


def get_relaxed_bernoulli_cdf(
    x_in: torch.Tensor,
    mean: Union[float, torch.Tensor],
    temperature: Union[float, torch.Tensor],
    eps: float = 1e-3,
) -> torch.Tensor:
    """CDF of the relaxed Bernoulli distribution (aka Binary Concrete Distribution)

    :param x_in: Input Tensor
    :param mean: Mean of the distribution
    :param temperature: Temperature of the distribution
    :param eps: Epsilon value for clipping
    """
    x_in = torch.clip(x_in, eps, 1 - eps)
    loga = torch.special.xlog1py(temperature, -x_in)
    loga -= torch.special.xlogy(temperature, x_in)
    if isinstance(mean, float):
        logp = logit_scipy(mean)
    else:
        logp = torch.special.logit(mean)
    pcdf = torch.special.expit(-loga - logp)
    return pcdf


def get_relaxed_bernoulli_pdf(
    x_in: torch.Tensor,
    mean: Union[float, torch.Tensor],
    temperature: Union[float, torch.Tensor],
    eps: float = 1e-3,
) -> torch.Tensor:
    """PDF of the relaxed Bernoulli distribution (aka Binary Concrete Distribution)

    :param x_in: Input Tensor
    :param mean: Mean of the distribution
    :param temperature: Temperature of the distribution
    :param eps: Epsilon value for clipping
    """
    x_in = torch.clip(x_in, eps, 1 - eps)
    numerator = temperature * mean * (1 - mean) * torch.pow(x_in * (1 - x_in), -temperature - 1)
    denominator = mean * torch.pow(x_in, -temperature) + (1 - mean) * torch.pow(
        1 - x_in, -temperature
    )
    return numerator / (denominator**2 + eps)
