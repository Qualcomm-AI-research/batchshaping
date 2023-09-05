# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Utils for notebook visualizations"""
from typing import List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import beta, norm

from batchshaping.batch_shaping_loss import Prior
from batchshaping.cdf_normal import get_normal_cdf
from batchshaping.cdf_relaxedbernoulli import (
    get_relaxed_bernoulli_cdf,
    get_relaxed_bernoulli_pdf,
)


def plot_gt_cdf(
    mpl_ax: plt.Axes,
    prior: Prior,
    prior_param1: float,
    prior_param2: Optional[float],
    num_pts: int = 100,
) -> None:
    """Plot target prior's CDF on the given subplot
    :param mpl_ax: Matplotlib axes to draw on
    :param prior: Choice of prior family
    :param prior_param1: Value of the first parameter of the prior
    :param prior_param2: Value of the second parameter of the prior
    :param num_pts: Number of points to draw
    """
    x_data = np.linspace(0.0, 1.0, num_pts)
    if prior == Prior.SYMBETA:
        y_data = beta.cdf(x_data, a=prior_param1, b=1 - prior_param1)
    elif prior == Prior.BETA:
        assert prior_param2 is not None
        y_data = beta.cdf(x_data, a=prior_param1, b=prior_param2)
    elif prior == Prior.RELAXED_BERNOULLI:
        assert prior_param2 is not None
        y_data = get_relaxed_bernoulli_cdf(torch.Tensor(x_data), prior_param1, prior_param2)
    elif prior == Prior.NORMAL:
        assert prior_param2 is not None
        y_data = get_normal_cdf(torch.Tensor(x_data), prior_param1, prior_param2)
    else:
        raise NotImplementedError("Unknown prior", prior)
    mpl_ax.plot(x_data, y_data, linewidth=2.5, color="orange", label="CDF (target)")


def plot_gt_pdf(  # pylint: disable=too-many-arguments
    mpl_ax: plt.Axes,
    prior: Prior,
    prior_param1: float,
    prior_param2: Optional[float],
    num_pts: int = 100,
    eps_clip_x: float = 0.005,
    eps_ylim: float = 0.25,
) -> None:
    """Plot target prior's PDF on the given subplot
    :param mpl_ax: Matplotlib axes to draw on
    :param prior: Choice of prior family
    :param prior_param1: Value of the first parameter of the prior
    :param prior_param2: Value of the second parameter of the prior
    :param num_pts: Number of points to draw
    :param eps_clip_x: Small clipping value to keep x in [0, 1] and avoid edge cases
      of {0, 1} where the prior can have very large magnitudes and is hard to plot
    :param eps_ylim: Small value to keep the axis's ylims slightly larger than the prior range
    """
    x_data = np.linspace(eps_clip_x, 1.0 - eps_clip_x, num_pts)
    if prior == Prior.SYMBETA:
        y_data = beta.pdf(x_data, prior_param1, 1 - prior_param1)
    elif prior == Prior.BETA:
        assert prior_param2 is not None
        y_data = beta.pdf(x_data, prior_param1, prior_param2)
    elif prior == Prior.RELAXED_BERNOULLI:
        assert prior_param2 is not None
        y_data = get_relaxed_bernoulli_pdf(torch.Tensor(x_data), prior_param1, prior_param2)
    elif prior == Prior.NORMAL:
        assert prior_param2 is not None
        y_data = norm.pdf(x_data, prior_param1, prior_param2)
    else:
        raise NotImplementedError("Unknown prior", prior)
    if isinstance(y_data, torch.Tensor):
        mpl_ax.set_ylim([-eps_ylim, torch.max(y_data).item() + eps_ylim])
    else:
        mpl_ax.set_ylim([-eps_ylim, np.amax(y_data) + eps_ylim])
    mpl_ax.plot(
        x_data,
        y_data,
        linewidth=2.5,
        color="orange",
        label="PDF (target)",
    )


def init_anim_plot(
    num_epochs: int, eps: float = 0.08
) -> Tuple[Figure, plt.Axes, plt.Axes, plt.Axes, PathCollection, Line2D, Line2D]:
    """Init plot for the BaS animation"""
    fig = plt.figure(figsize=(7, 8.5))
    grid = plt.GridSpec(3, 2, wspace=0.3, hspace=0.4)
    data_ax = plt.subplot(grid[:2, :])
    data_ax.set_xlim([-eps, 1 + eps])
    data_ax.set_ylim([-eps, 1 + eps])
    data_ax.set_xlabel("Data range", fontsize=16)
    data_plot = data_ax.scatter([], [], marker="o", alpha=0.15, label="data", color="orchid")
    (cdf_pred_plot,) = data_ax.plot(
        [], [], linewidth=2.5, linestyle="dashed", color="xkcd:turquoise", label="CDF (data)"
    )

    loss_ax = plt.subplot(grid[2:, 0])
    loss_ax.set_xlim([0, num_epochs])
    loss_ax.set_ylabel("Loss", fontsize=16)
    loss_ax.set_xlabel("Training step", fontsize=16)
    (loss_plot,) = loss_ax.plot([], [], marker=".", linewidth=2, color="k")

    pdf_ax = plt.subplot(grid[2:, 1])
    pdf_ax.set_xlim([0, 1.0])
    return fig, data_ax, loss_ax, pdf_ax, data_plot, cdf_pred_plot, loss_plot


def init_gbas_anim_plot(
    num_epochs: int, eps: float = 0.08
) -> Tuple[Figure, List[plt.Axes], plt.Axes, plt.Axes, plt.Axes, Line2D, Line2D]:
    """Init plot for GBaS animation; displays the CDF for the first six dimensions"""
    fig = plt.figure(figsize=(15, 8))
    grid = plt.GridSpec(3, 4, wspace=0.45, hspace=0.4)
    data_axes = []
    for i in range(6):
        col = i % 2
        row = i // 2
        data_ax = plt.subplot(grid[row, col])
        data_ax.set_xlim([-eps, 1 + eps])
        data_ax.set_ylim([-eps, 1 + eps])
        data_ax.set_xlabel("Data range", fontsize=16)
        data_axes.append(data_ax)

    # priors loss
    priors_loss_ax = plt.subplot(grid[0, 2:])
    priors_loss_ax.set_xlim([0, num_epochs])
    priors_loss_ax.set_ylabel("Priors loss (sum)", fontsize=18)
    priors_loss_ax.set_xlabel("Training step", fontsize=18)
    (priors_loss_plot,) = priors_loss_ax.plot([], [], marker=".", linewidth=2, color="k")

    # hyperprior loss
    hyperprior_loss_ax = plt.subplot(grid[1, 2:])
    hyperprior_loss_ax.set_xlim([0, num_epochs])
    hyperprior_loss_ax.set_ylabel("Hyperprior Loss", fontsize=18)
    hyperprior_loss_ax.set_xlabel("Training step", fontsize=18)
    (hyperprior_loss_plot,) = hyperprior_loss_ax.plot([], [], marker=".", linewidth=2, color="k")

    # Hyperprior PDF
    hyperprior_pdf_ax = plt.subplot(grid[2, 2:])
    hyperprior_pdf_ax.set_xlim([0, 1.0])
    return (
        fig,
        data_axes,
        priors_loss_ax,
        hyperprior_loss_ax,
        hyperprior_pdf_ax,
        priors_loss_plot,
        hyperprior_loss_plot,
    )
