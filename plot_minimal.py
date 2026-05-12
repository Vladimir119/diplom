"""Minimal matplotlib figure style: transparent canvas, no grid, no top/right spines."""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def apply_minimal_figure_style(
    fig: Figure,
    axes: Optional[Union[np.ndarray, object]] = None,
    *,
    transparent: bool = True,
    ax_patch_alpha: float = 0.0,
) -> None:
    """Apply transparent figure, hide top/right spines, disable grid on all axes."""
    if transparent:
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor("none")

    if axes is None:
        ax_list = fig.axes
    else:
        ax_list = np.atleast_1d(axes).ravel()

    for ax in ax_list:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if transparent:
            ax.patch.set_alpha(ax_patch_alpha)
            ax.patch.set_facecolor("none")


def minfig(*args, **kwargs):
    """plt.subplots wrapper: forward kwargs, then apply minimal style (call after!)."""
    return plt.subplots(*args, **kwargs)
