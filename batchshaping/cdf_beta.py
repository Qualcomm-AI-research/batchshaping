# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Differentiable CDF for the Beta distribution"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import beta as beta_scipy

from batchshaping.utils import to_numpy


def __get_bas_beta_cdf_factory__(
    symmetric: bool,
    beta_a: Union[float, torch.Tensor],
    beta_b: Optional[Union[float, torch.Tensor]] = None,
    # PDF clipping
    pdf_clip_min: float = 1e-3,
    pdf_clip_max: float = 100,
    # CDF gradient estimation wrt params
    **_kwargs: Any
) -> torch.autograd.Function:
    """Returns a differentiable Beta CDF wrt to its inputs only

    :param symmetric: If True, uses the B(a, 1-a) distribution and ignores b
    :param beta_a: First parameter
    :param beta_b: Second parameter (not used if symmetric)
    :param pdf_clip_min: Min bound for clipping the PDF values
    :param pdf_clip_max: Min bound for clipping the PDF values
    """
    del _kwargs
    # Classical Batch Shaping Loss: Non-trainable parameters for B(a, b) and B(a, 1 - a)
    if symmetric:
        assert beta_b is None
        beta_b = 1 - beta_a
    assert beta_b is not None

    def __validate_param__(param: Union[float, torch.Tensor]) -> float:
        assert param is not None
        if isinstance(param, torch.Tensor):
            assert param.squeeze().dim() == 0
            param = float(to_numpy(param.squeeze()))
        return param

    beta_a = __validate_param__(beta_a)
    beta_b = __validate_param__(beta_b)

    class BetaCDF(torch.autograd.Function):  # pylint: disable=abstract-method
        """Beta CDF prior differentiable wrt to the inputs"""

        @staticmethod
        def forward(ctx: Any, inp_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            """Note that input is expected to be a sorted array of two dimensions.
            The CDF will be computed independently across each array along dim 0"""
            device = inp_tensor.device
            inp = to_numpy(inp_tensor.clone())

            # PDF
            ppdf = beta_scipy.pdf(inp, a=beta_a, b=beta_b)
            ppdf = np.clip(ppdf, a_min=pdf_clip_min, a_max=pdf_clip_max)  # clip for gradient
            ppdf = torch.tensor(ppdf, dtype=torch.float32, device=device)
            ctx.save_for_backward(ppdf)

            # CDF
            pcdf = torch.tensor(
                beta_scipy.cdf(inp, a=beta_a, b=beta_b), dtype=torch.float32, device=device
            )
            return pcdf

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any, grad_loss: torch.Tensor
        ) -> Optional[torch.Tensor]:
            return ctx.saved_tensors[0] * grad_loss

    return BetaCDF()


def __get_gbas_beta_cdf_factory__(
    symmetric: bool,
    beta_a: Optional[Union[float, torch.Tensor]],
    beta_b: Optional[Union[float, torch.Tensor]] = None,
    # PDF clipping
    pdf_clip_min: float = 1e-3,
    pdf_clip_max: float = 100,
    # CDF gradient estimation wrt params
    num_pts: int = 5,
    eps: float = 0.05,
) -> torch.autograd.Function:
    """Returns a differentiable Beta CDF wrt to its inputs and parameters

    :param symmetric: If True, uses the B(a, 1-a) distribution and ignores b
    :param beta_a: First parameter
    :param beta_b: Second parameter (not used if symmetric)
    :param pdf_clip_min: Min bound for clipping the PDF values
    :param pdf_clip_max: Min bound for clipping the PDF values
    :param num_pts: Number of points to evaluate gradient of the Beta CDF wrt to
      its parameters
    :param eps: Epsilon value to avoid float errors
    """
    # Generalized Batch Shaping Loss: Trainable parameter for B(a, 1-a)
    # There is no simple closed-form solution for the derivatives of CDF(B(a, b))
    # with respect to a and b, so we estimate them with np.gradients instead
    if symmetric:
        assert beta_a is None
        assert beta_b is None

        class DiffSymBetaCDFWrtParams(torch.autograd.Function):  # pylint: disable=abstract-method
            """Beta(a, 1 - a) CDF prior differentiable wrt to the inputs and alpha parameter"""

            @staticmethod
            def forward(  # type: ignore[override]
                ctx: Any, inp_tensor: torch.Tensor, prior_param: torch.Tensor
            ) -> torch.Tensor:
                """Note that input is expected to be a sorted array of two dimensions.
                The CDF will be computed independently across each array along dim 0"""
                device = inp_tensor.device
                inp = to_numpy(inp_tensor.clone())
                alpha = float(to_numpy(prior_param))

                # PDF
                ppdf = beta_scipy.pdf(inp, a=alpha, b=1 - alpha)
                ppdf = np.clip(ppdf, a_min=pdf_clip_min, a_max=pdf_clip_max)  # clip for gradient
                ppdf = torch.tensor(ppdf, dtype=torch.float32, device=device)
                ctx.save_for_backward(ppdf)

                # CDF
                pcdf = torch.tensor(
                    beta_scipy.cdf(inp, a=alpha, b=1 - alpha), dtype=torch.float32, device=device
                )

                # Estimate CDF derivative wrt alpha
                alphas = np.linspace(
                    max(eps, alpha - eps), min(1 - eps, alpha + eps), num_pts * 2 + 1
                )
                f = np.stack([beta_scipy.cdf(inp, a=al, b=1 - al) for al in alphas], axis=0)
                param_grad = torch.tensor(
                    np.gradient(f, alphas, axis=0)[num_pts], dtype=torch.float32, device=device
                )
                ctx.save_for_backward(ppdf, param_grad)
                return pcdf

            @staticmethod
            def backward(  # type: ignore[override]
                ctx: Any, grad_loss: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                return ctx.saved_tensors[0] * grad_loss, ctx.saved_tensors[1] * grad_loss

        return DiffSymBetaCDFWrtParams()

    # Trainable parameters for general B(a, b)
    assert beta_a is None
    assert beta_b is None

    class DiffBetaCDFWrtParams(torch.autograd.Function):  # pylint: disable=abstract-method
        """Beta(a, b) CDF prior differentiable wrt to the inputs and alpha parameter"""

        @staticmethod
        def forward(  # type: ignore[override]
            ctx: Any,
            inp_tensor: torch.Tensor,
            prior_param1: torch.Tensor,
            prior_param2: torch.Tensor,
        ) -> torch.Tensor:
            """Note that input is expected to be a sorted array of two dimensions.
            The CDF will be computed independently across each array along dim 0"""
            device = inp_tensor.device
            inp = to_numpy(inp_tensor.clone())
            alpha = float(to_numpy(prior_param1))
            beta = float(to_numpy(prior_param2))

            # PDF
            ppdf = beta_scipy.pdf(inp, a=alpha, b=beta)
            ppdf = np.clip(ppdf, a_min=pdf_clip_min, a_max=pdf_clip_max)  # clip for gradient
            ppdf = torch.tensor(ppdf, dtype=torch.float32, device=device)
            ctx.save_for_backward(ppdf)

            # CDF
            pcdf = torch.tensor(
                beta_scipy.cdf(inp, a=alpha, b=beta), dtype=torch.float32, device=device
            )

            # Estimate CDF derivative wrt alpha
            alphas = np.linspace(max(eps, alpha - eps), alpha + eps, num_pts * 2 + 1)
            f = np.stack([beta_scipy.cdf(inp, a=al, b=beta) for al in alphas], axis=0)
            param_grad1 = torch.tensor(
                np.gradient(f, alphas, axis=0)[num_pts], dtype=torch.float32, device=device
            )

            # Estimate CDF derivative wrt beta
            betas = np.linspace(max(eps, beta - eps), beta + eps, num_pts * 2 + 1)
            f = np.stack([beta_scipy.cdf(inp, a=alpha, b=be) for be in betas], axis=0)
            param_grad2 = torch.tensor(
                np.gradient(f, betas, axis=0)[num_pts], dtype=torch.float32, device=device
            )
            ctx.save_for_backward(ppdf, param_grad1, param_grad2)
            return pcdf

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any, grad_loss: torch.Tensor
        ) -> Tuple[torch.Tensor, ...]:
            return (
                ctx.saved_tensors[0] * grad_loss,
                ctx.saved_tensors[1] * grad_loss,
                ctx.saved_tensors[2] * grad_loss,
            )

    return DiffBetaCDFWrtParams()


def get_beta_cdf_factory(
    trainable_params: bool, *args: Any, **kwargs: Any
) -> torch.autograd.Function:
    """Returns the correct differentiable beta CDF function
    :param trainable_params: Whether the parameters a and b of B(a, b) should be differentiable
    :param args: Arguments to feed to the CDF function
    :param kwargs: Keyword arguments to feed to the CDF function
    :return: Differentiable module computing the CDF of B(a, b) in the forward pass
    """
    if trainable_params:
        return __get_gbas_beta_cdf_factory__(*args, **kwargs)
    return __get_bas_beta_cdf_factory__(*args, **kwargs)
