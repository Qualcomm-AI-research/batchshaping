# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Implements the Batch-Shaping loss for different prior distribution"""

from enum import Enum
from typing import Optional, Union

import torch

from batchshaping.cdf_beta import get_beta_cdf_factory
from batchshaping.cdf_normal import get_normal_cdf
from batchshaping.cdf_relaxedbernoulli import get_relaxed_bernoulli_cdf
from batchshaping.utils import validate_prior_param


class Prior(Enum):
    """Selects the family of priors for the Batch shaping loss"""

    SYMBETA = 0  # B(a, 1-a)
    BETA = 1  # B(a, b)
    RELAXED_BERNOULLI = 2  # RB(mu, temperature)
    NORMAL = 3  # N(mu, sigma)


def gbas_loss(  # pylint: disable=too-many-arguments, too-many-branches
    x_in: torch.Tensor,
    prior: Prior,
    prior_param1: Union[torch.Tensor, float, int],
    prior_param2: Optional[Union[torch.Tensor, float, int]] = None,
    trainable_prior_params: bool = False,
    dim: Optional[int] = None,
    handle_ties: bool = False,
) -> torch.Tensor:
    """(Generalized) Batch shaping loss

    :param x_in: Input tensor of shape (batch, ...).
    :param prior: Selected prior family
    :param prior_param1: First parameter of the prior. Either a float (constant),
      a trainable scalar Tensor (same across all dimensions) or a 1-dim Tensor (
      determine different prior parameter for each subarray along dim)
    :param prior_param2: Second parameter of the prior, optional. Can also be
      either a float, scalar Tensor or 1-dimensional tensor
    :param trainable_prior_params: If ``True``, gradients will also flow through
      the prior parameters. Assumes prior_param{1,2} are tensors.
    :param dim: If not None, the CDF diff loss will be applied along dim. If None,
      it will be applied on the flattened input array (i.e., original Batch Shaping)
    :param handle_ties: Whether to handle ties (equal values) when estimating the
      data's CDF.


    :return: The batch shaping loss
    """
    if trainable_prior_params:
        assert isinstance(prior_param1, torch.Tensor)
        assert prior_param2 is None or isinstance(prior_param2, torch.Tensor)

    # Flatten inputs to a two dimensional tensor where:
    # Dim 0: Index individual arrays to apply the loss on
    # Dim 1: CDF loss will be applied across this dimension for each subarray
    if dim is not None:
        x_in = x_in.transpose(dim, 0)
        x_in = x_in.reshape(x_in.shape[0], -1)
    else:
        x_in = x_in.flatten()[None, :]

    # Reshape prior_param to -> ((1 or x.shape[0]), 1)
    prior_param1 = validate_prior_param(prior_param1, x_in.shape[0], x_in.device)
    if prior_param2 is not None:
        prior_param2 = validate_prior_param(prior_param2, x_in.shape[0], x_in.device)

    # Sort input
    x_in, _ = torch.sort(x_in, dim=-1)

    # Compute Differentiable CDF along second dimension
    # B(a, 1 - a)
    if prior == Prior.SYMBETA:
        assert prior_param2 is None
        # BaS: Parameter a is not trainable
        if not trainable_prior_params:
            pcdf = torch.cat(
                [
                    get_beta_cdf_factory(trainable_prior_params, symmetric=True, beta_a=p).apply(
                        x_in
                    )
                    for p in prior_param1
                ],
                dim=0,
            )
        # GBaS: Parameter a is trainable and controlled by hyperprior
        else:
            pcdf = torch.cat(
                [
                    get_beta_cdf_factory(trainable_prior_params, symmetric=True, beta_a=None).apply(
                        x_in, p
                    )
                    for p in prior_param1
                ],
                dim=0,
            )

    # B(a, b)
    elif prior == Prior.BETA:
        assert prior_param2 is not None
        # Align param1 and param2 shapes for zipping
        num_params = max(prior_param1.shape[0], prior_param2.shape[0])
        if prior_param1.shape[0] < num_params:
            prior_param1 = prior_param1.expand(num_params, -1)
        if prior_param2.shape[0] < num_params:
            prior_param2 = prior_param2.expand(num_params, -1)

        if not trainable_prior_params:
            pcdf = torch.cat(
                [
                    get_beta_cdf_factory(
                        trainable_prior_params, symmetric=False, beta_a=p1, beta_b=p2
                    ).apply(x_in)
                    for (p1, p2) in zip(prior_param1, prior_param2)
                ],
                dim=0,
            )
        else:
            pcdf = torch.cat(
                [
                    get_beta_cdf_factory(
                        trainable_prior_params, symmetric=False, beta_a=None, beta_b=None
                    ).apply(x_in, p1, p2)
                    for (p1, p2) in zip(prior_param1, prior_param2)
                ],
                dim=0,
            )

    # RelaxedBernoulli(mean, temperature)
    elif prior == Prior.RELAXED_BERNOULLI:
        assert prior_param2 is not None
        pcdf = get_relaxed_bernoulli_cdf(x_in, mean=prior_param1, temperature=prior_param2)

    # Normal(mean, var)
    elif prior == Prior.NORMAL:
        assert prior_param2 is not None
        pcdf = get_normal_cdf(x_in, mean=prior_param1, std=prior_param2)

    else:
        raise NotImplementedError("Unknown prior", prior)

    # Compute empirical CDF
    if not handle_ties:
        ecdf = torch.arange(1, x_in.shape[1] + 1, device=x_in.device, dtype=torch.float) / (
            x_in.shape[1] + 1
        )
    # For the sake of completeness only: if there are equal data points $x_i = x_j$
    # then the previous empirical CDF estimation is not perfectly exact as
    # we need to handle the ties.
    # however this rarely happens in practice since $x$ is typically continuous
    # thus `handle_ties` defaults to False
    else:
        _, unique_idx, counts = torch.unique_consecutive(
            x_in, return_inverse=True, return_counts=True, dim=-1
        )
        # gather counts
        counts = torch.cumsum(counts, dim=-1) / (x_in.shape[1] + 1)
        ecdf = torch.gather(counts, dim=-1, index=unique_idx)

    # Return CDF difference
    cdf_diff = torch.sum((ecdf[None, :] - pcdf) ** 2, dim=-1) / x_in.shape[1]
    return torch.mean(cdf_diff)
