# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Minor utils functions"""

import os
import random
from typing import Callable, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixes the random seed for Pytorch, numpy and Python random module"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Helper function that turns the given tensor into a numpy array"""
    if hasattr(tensor, "is_cuda") and tensor.is_cuda:
        tensor = tensor.cpu()
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()

    return np.array(tensor)


def validate_prior_param(
    prior_param: Union[float, torch.Tensor], dim0: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Reshape prior parameters"""
    # Check dimensions
    if isinstance(prior_param, torch.Tensor) and prior_param.dim() == 1:
        assert dim0 in [1, prior_param.shape[0]]
    if isinstance(prior_param, (float, int)):
        prior_param = torch.ones(1, 1, device=device, dtype=torch.float32) * prior_param
    # Reshape
    if prior_param.dim() == 0:
        prior_param = prior_param[None, None]
    elif prior_param.dim() == 1:
        prior_param = prior_param[:, None]
    return prior_param


def warmup_factory(
    num_warmup_steps: int, learning_rate: float, min_learning_rate: float = 1e-4
) -> Callable[[int], float]:
    """Create a learning rate linear warmup scheduler"""

    def __warmup__(step: int) -> float:
        if step < num_warmup_steps:
            return max(min_learning_rate, learning_rate * step / num_warmup_steps)
        return learning_rate

    return __warmup__
