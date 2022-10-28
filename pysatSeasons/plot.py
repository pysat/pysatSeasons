#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022, pysat development team
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Support scatterplot production over seasons of interest."""

import matplotlib as mpl
import matplotlib.pyplot as plt
# Axes3D import needed to support 3D scatter plot below. Use not apparent
# but code required when projection='3d' set on an axis.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

import pysat


def scatterplot(const, labelx, labely, data_label, datalim, xlim=None,
                ylim=None):
    """Return 2D and 3D scatterplot of `data_label` over `label*` for a season.

    Parameters
    ----------
    const : pysat.Instrument or pysat.Constellation
        Instrument/Constellation to scatterplot.
    labelx : str
        Data product for x-axis.
    labely : str
        Data product for y-axis.
    data_label : str or array-like of str
        Data product(s) to be scatter plotted.
    datalim : numpy.array
        Plot limits for data_label.
    xlim, ylim : numpy.array or None.
        Array for limits along x or y axes. If None, limits
        are determined automatically. (default=None)

    Returns
    -------
    figs : list
        Scatter plots of `data_label` as a function of `labelx` and `labely`
        over the season delineated by `inst.bounds`.

    """

    if isinstance(const, pysat.Instrument):
        const = pysat.Constellation(instruments=[const])
    elif not isinstance(const, pysat.Constellation):
        raise ValueError("Parameter must be an Instrument or a Constellation.")

    # Get current plot settings. Alter for this function.
    if mpl.is_interactive():
        interactive_mode = True
        # Turn interactive plotting off
        plt.ioff()
    else:
        interactive_mode = False

    # Create figures for plotting
    figs = []
    axs = []

    # Check for list-like behaviour of data_label
    data_label = pysat.utils.listify(data_label)

    # Multiple data to be plotted
    for i in np.arange(len(data_label)):
        figs.append(plt.figure())
        ax1 = figs[i].add_subplot(211, projection='3d')
        ax2 = figs[i].add_subplot(212)
        axs.append((ax1, ax2))
        plt.suptitle(data_label[i])
        if xlim is not None:
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)
        if ylim is not None:
            ax1.set_ylim(ylim)
            ax2.set_ylim(ylim)

    # Norm method so that data may be scaled to colors appropriately
    norm = mpl.colors.Normalize(vmin=datalim[0], vmax=datalim[1])

    # Prep memory to store figures.
    figs3d = [i for i in np.arange(len(figs))]
    figs2d = [i for i in np.arange(len(figs))]

    # Perform plotting while iterating over data season.
    for linst in const.instruments:
        for inst in linst:
            if not inst.empty:
                for j, (fig, ax) in enumerate(zip(figs, axs)):
                    if (len(inst.data[labelx]) > 0
                            and (len(inst.data[labely]) > 0)
                            and (len(inst.data[data_label[j]]) > 0)):
                        figs3d[j] = ax[0].scatter(inst.data[labelx],
                                                  inst.data[labely],
                                                  inst.data[data_label[j]],
                                                  zdir='z',
                                                  c=inst.data[data_label[j]],
                                                  norm=norm,
                                                  linewidth=0, edgecolors=None)
                        figs2d[j] = ax[1].scatter(inst.data[labelx],
                                                  inst.data[labely],
                                                  c=inst.data[data_label[j]],
                                                  norm=norm, alpha=0.5,
                                                  edgecolor=None)

    for j, (fig, ax) in enumerate(zip(figs, axs)):
        try:
            plt.colorbar(figs3d[j], ax=ax[0], label='Amplitude (m/s)')
        except Exception:
            print('Tried colorbar but failed, thus no colorbar.')
        ax[0].elev = 30.

    if interactive_mode:
        # Turn interactive plotting back on
        plt.ion()

    return figs
