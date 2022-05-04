#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022, pysat development team
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Occurrence probability routines, daily or by orbit.

Routines calculate the occurrence of an event greater than a supplied gate
occurring at least once per day, or once per orbit. The probability is
calculated as the (number of times with at least one hit in bin) / (number
of times in the bin). The data used to determine the occurrence must be 1D.
If a property of a 2D or higher dataset is needed attach a custom function
that performs the check and returns a 1D Series.

Note
----
The included routines use the bounds attached to the supplied instrument
object as the season of interest.

"""

import numpy as np
import warnings

import pysat


def daily2D(const, bin1, label1, bin2, label2, data_label, gate,
            returnBins=None, return_bins=False):
    """2D Daily Occurrence Probability of `data_label` > `gate` over a season.

    .. deprecated:: 0.2.0
              `returnBins` will be removed in v0.3.0, it is replaced by
              `return_bins` because it has standards compliant case.

    If `data_label` is greater than `gate` at least once per day,
    then a 100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    const : pysat.Instrument or pysat.Constellation
        Instrument/Constellation to use for calculating occurrence probability.
    bin1, bin2 : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    label1, label2 : str
        Identifies data product for binX.
    data_label : list of str
        Identifies data product(s) to calculate occurrence probability
        e.g. inst[data_label].
    gate : list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins : bool or NoneType
        If True, also return arrays with values of bin edges, useful for pcolor.
        Deprecated in favor of `return_bins`. (default=None)
    return_bins : bool
        If True, also return arrays with values of bin edges, useful for pcolor.
        (default=False)


    Returns
    -------
    occur_prob : dict
        A dict of dicts indexed by `data_label`. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of days with any
        data; 'bin_x' and 'bin_y' are also returned if requested. Note that
        arrays are organized for direct plotting, y values along rows, x along
        columns.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    return _occurrence2D(const, bin1, label1, bin2, label2, data_label, gate,
                         by_orbit=False, returnBins=returnBins,
                         return_bins=return_bins)


def by_orbit2D(const, bin1, label1, bin2, label2, data_label, gate,
               returnBins=None, return_bins=False):
    """2D Occurrence Probability of `data_label` orbit-by-orbit over a season.

    .. deprecated:: 0.2.0
              `returnBins` will be removed in v0.3.0, it is replaced by
              `return_bins` because it has standards compliant case.

    If `data_label` is greater than `gate` at least once per orbit, then a
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    const : pysat.Instrument or pysat.Constellation
        Instrument/Constellation to use for calculating occurrence probability.
    bin1, bin2 : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    label1, label2 : str
        Identifies data product for binX.
    data_label : list of str
        Data product label(s) to calculate occurrence probability.
    gate : list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins : bool or NoneType
        If True, also return arrays with values of bin edges, useful for pcolor.
        Deprecated in favor of `return_bins`. (default=None)
    return_bins : bool
        If True, also return arrays with values of bin edges, useful for pcolor.
        (default=False)

    Returns
    -------
    occur_prob : dict
        A dict of dicts indexed by data_label. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of orbits with
        any data; 'bin_x' and 'bin_y' are also returned if requested. Note that
        arrays are organized for direct plotting, y values along rows, x along
        columns.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    return _occurrence2D(const, bin1, label1, bin2, label2, data_label, gate,
                         by_orbit=True, returnBins=returnBins,
                         return_bins=return_bins)


def _occurrence2D(const, bin1, label1, bin2, label2, data_label, gate,
                  by_orbit=False, returnBins=None, return_bins=False):
    """2D Occurrence Probability of `data_label` orbit-by-orbit over a season.

    .. deprecated:: 0.2.0
              `returnBins` will be removed in v0.3.0, it is replaced by
              `return_bins` because it has standards compliant case.

    If `data_label` is greater than `gate` at least once per orbit, then a
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    const : pysat.Instrument or pysat.Constellation
        Instrument to use for calculating occurrence probability.
    bin1, bin2 : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    label1, label2 : str
        Identifies data product for binX.
    data_label : list of str
        Data product label(s) to calculate occurrence probability.
    gate : list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    by_orbit : bool
        If True, then occurrence probability determined by orbit rather than
        by day. (default=False)
    returnBins : bool or NoneType
        If True, also return arrays with values of bin edges, useful for pcolor.
        Deprecated in favor of `return_bins`. (default=None)
    return_bins : bool
        If True, also return arrays with values of bin edges, useful for pcolor.
        (default=False)

    Returns
    -------
    occur_prob : dict
        A dict of dicts indexed by data_label. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of orbits with
        any data; 'bin_x' and 'bin_y' are also returned if requested. Note that
        arrays are organized for direct plotting, y values along rows, x along
        columns.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    if isinstance(const, pysat.Instrument):
        const = pysat.Constellation(instruments=[const])
    elif not isinstance(const, pysat.Constellation):
        raise ValueError("Parameter must be an Instrument or a Constellation.")

    if returnBins is not None:
        return_bins = returnBins
        warnings.warn(''.join(['"returnBins" has been deprecated in favor of ',
                               '"return_bins". Assigning input "returnBins" ',
                               'to "return_bins". ']), DeprecationWarning,
                      stacklevel=2)

    # Ensure inputs are list-like
    data_label = pysat.utils.listify(data_label)
    gate = pysat.utils.listify(gate)

    if len(gate) != len(data_label):
        raise ValueError('Must have a gate value for each data_label')

    # Create bins
    binx = np.linspace(bin1[0], bin1[1], bin1[2] + 1)
    biny = np.linspace(bin2[0], bin2[1], bin2[2] + 1)

    numx, numy, numz = len(binx) - 1, len(biny) - 1, len(data_label)
    arrx, arry, arrz = np.arange(numx), np.arange(numy), np.arange(numz)

    # Create arrays to store all values
    total = np.zeros((numz, numy, numx))
    hits = np.zeros((numz, numy, numx))

    # Support iterating by day or by orbit.
    iter_insts = []
    if by_orbit:
        for inst in const:
            iter_insts.append(inst.orbits)
    else:
        for inst in const:
            iter_insts.append(inst)

    # Iterate over instruments and calculate occurrence.
    for inst, cinst in zip(iter_insts, const.instruments):

        # Copy instrument to provide data source independent access
        loop_inst = cinst.copy()

        # Iterate over Instrument bounds by orbit or by day.
        for linst in inst:

            # Collect data in 2D distribution of bins
            if len(linst.data) != 0:
                xind = np.digitize(linst.data[label1], binx) - 1
                for xi in arrx:
                    xindex, = np.where(xind == xi)
                    if len(xindex) > 0:
                        loop_inst.data = linst[xindex]
                        yind = np.digitize(loop_inst[label2], biny) - 1
                        for yj in arry:
                            yindex, = np.where(yind == yj)
                            if len(yindex) > 0:

                                # Binning complete. Iterate over the different
                                # data_labels and record if above gate.
                                for zk in arrz:
                                    zdata = loop_inst[yindex, data_label[zk]]
                                    if np.any(np.isfinite(zdata)):
                                        # There is some data.
                                        total[zk, yj, xi] += 1.

                                        if np.any(zdata > gate[zk]):
                                            # There is data above the gate.
                                            hits[zk, yj, xi] += 1.

    # All of the loading and storing data is done. Calculate probability.
    prob = hits / total

    # Make nicer dictionary output
    output = {}
    for i, label in enumerate(data_label):
        output[label] = {'prob': prob[i, :, :], 'count': total[i, :, :]}
        if return_bins:
            output[label]['bin_x'] = binx
            output[label]['bin_y'] = biny

    # Clean up
    del iter_insts, loop_inst
    return output


def daily3D(const, bin1, label1, bin2, label2, bin3, label3,
            data_label, gate, returnBins=None, return_bins=False):
    """3D Daily Occurrence Probability of `data_label` > `gate` over a season.

    .. deprecated:: 0.2.0
              `returnBins` will be removed in v0.3.0, it is replaced by
              `return_bins` because it has standards compliant case.

    If `data_label` is greater than `gate` at least once per day,
    then a 100% occurrence probability results. Season delineated by
    the bounds attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    const : pysat.Instrument or pysat.Constellation
        Instrument/Constellation to use for calculating occurrence probability.
    bin1, bin2, bin3 : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    label1, label2, label3 : str
        Identifies data product for binX.
    data_label : list of str
        Identifies data product(s) to calculate occurrence probability
        e.g. inst[data_label].
    gate : list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins : bool or NoneType
        If True, also return arrays with values of bin edges, useful for pcolor.
        Deprecated in favor of `return_bins`. (default=None)
    return_bins : bool
        If True, also return arrays with values of bin edges, useful for pcolor.
        (default=False)

    Returns
    -------
    occur_prob : dict
        A dict of dicts indexed by `data_label`. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of days with any
        data; 'bin_x', 'bin_y', and 'bin_z' are also returned if requested.
        Note that arrays are organized for direct plotting, z, y, x.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    return _occurrence3D(const, bin1, label1, bin2, label2, bin3, label3,
                         data_label, gate, returnBins=returnBins,
                         return_bins=return_bins, by_orbit=False)


def by_orbit3D(const, bin1, label1, bin2, label2, bin3, label3,
               data_label, gate, returnBins=None, return_bins=False):
    """3D Occurrence Probability of `data_label` orbit-by-orbit over a season.

    .. deprecated:: 0.2.0
              `returnBins` will be removed in v0.3.0, it is replaced by
              `return_bins` because it has standards compliant case.

    If `data_label` is greater than `gate` at least once per orbit, then a
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Prob = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    const : pysat.Instrument or pysat.Constellation
        Instrument/Constellation to use for calculating occurrence probability.
    bin1, bin2, bin3 : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    label1, label2, label3 : str
        Identifies data product for binX.
    data_label : list of str
        Identifies data product(s) to calculate occurrence probability
        e.g. inst[data_label].
    gate : list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins : bool or NoneType
        If True, also return arrays with values of bin edges, useful for pcolor.
        Deprecated in favor of `return_bins`. (default=None)
    return_bins : bool
        If True, also return arrays with values of bin edges, useful for pcolor.
        (default=False)

    Returns
    -------
    occur_prob : dict
        A dict of dicts indexed by data_label. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of orbits with
        any data; 'bin_x', 'bin_y', and 'bin_z' are also returned if requested.
        Note that arrays are organized for direct plotting, z, y, x.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    return _occurrence3D(const, bin1, label1, bin2, label2, bin3, label3,
                         data_label, gate, returnBins=returnBins,
                         return_bins=return_bins, by_orbit=True)


def _occurrence3D(const, bin1, label1, bin2, label2, bin3, label3,
                  data_label, gate, returnBins=None, return_bins=False,
                  by_orbit=False):
    """3D Occurrence Probability of `data_label` orbit-by-orbit over a season.

    .. deprecated:: 0.2.0
              `returnBins` will be removed in v0.3.0, it is replaced by
              `return_bins` because it has standards compliant case.

    If `data_label` is greater than `gate` at least once per iteration, then a
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    const : pysat.Instrument or pysat.Constellation
        Instrument/Constellation to use for calculating occurrence probability.
    bin1, bin2, bin3 : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    label1, label2, label3: str
        Identifies data product for binX.
    data_label: list of str
        Data product label(s) to calculate occurrence probability.
    gate: list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins : bool or NoneType
        If True, also return arrays with values of bin edges, useful for pcolor.
        Deprecated in favor of `return_bins`. (default=None)
    return_bins : bool
        If True, also return arrays with values of bin edges, useful for pcolor.
        (default=False)
    by_orbit : bool
        If True, then occurrence probability determined by orbit rather than
        by day. (default=False)

    Returns
    -------
    occur_prob : dict
        A dict of dicts indexed by data_label. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of orbits with
        any data; 'bin_x', 'bin_y', and 'bin_z' are also returned if requested.
        Note that arrays are organized for direct plotting, z, y, x.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    if isinstance(const, pysat.Instrument):
        const = pysat.Constellation(instruments=[const])
    elif not isinstance(const, pysat.Constellation):
        raise ValueError("Parameter must be an Instrument or a Constellation.")

    if returnBins is not None:
        return_bins = returnBins
        warnings.warn(''.join(['"returnBins" has been deprecated in favor of ',
                               '"return_bins". Assigning input "returnBins" ',
                               'to "return_bins". ']), DeprecationWarning,
                      stacklevel=2)

    # Ensure inputs are list-like
    data_label = pysat.utils.listify(data_label)
    gate = pysat.utils.listify(gate)

    if len(gate) != len(data_label):
        raise ValueError('Must have a gate value for each data_label')

    # Create bins
    binx = np.linspace(bin1[0], bin1[1], bin1[2] + 1)
    biny = np.linspace(bin2[0], bin2[1], bin2[2] + 1)
    binz = np.linspace(bin3[0], bin3[1], bin3[2] + 1)

    numx, numy, numz, numd = (len(binx) - 1, len(biny) - 1, len(binz) - 1,
                              len(data_label))

    # Create array to store all values before taking median
    xarr, yarr, zarr, darr = (np.arange(numx), np.arange(numy), np.arange(numz),
                              np.arange(numd))

    total = np.zeros((numd, numz, numy, numx))
    hits = np.zeros((numd, numz, numy, numx))

    # Support iterating by day or by orbit.
    iter_insts = []
    if by_orbit:
        for inst in const:
            iter_insts.append(inst.orbits)
    else:
        for inst in const:
            iter_insts.append(inst)

    # Iterate over instruments and calculate occurrence.
    for inst, cinst in zip(iter_insts, const.instruments):

        # Copy instrument to provide data source independent access
        loop_sat_y = cinst.copy()
        loop_sat_z = cinst.copy()

        # Iterate over given season
        for sat in inst:

            # Bin data over 3D bins
            if not sat.empty:
                xind = np.digitize(sat.data[label1], binx) - 1
                for xi in xarr:
                    xindex, = np.where(xind == xi)
                    if len(xindex) > 0:
                        loop_sat_y.data = sat[xindex]
                        yind = np.digitize(loop_sat_y[label2], biny) - 1
                        for yj in yarr:
                            yindex, = np.where(yind == yj)
                            if len(yindex) > 0:
                                loop_sat_z.data = loop_sat_y[yindex]
                                zind = np.digitize(loop_sat_z[label3], binz) - 1
                                for zk in zarr:
                                    zindex, = np.where(zind == zk)
                                    if len(zindex) > 0:

                                        # Binning complete. Calculate data
                                        # larger than gate.
                                        for di in darr:
                                            ddata = loop_sat_z[zindex,
                                                               data_label[di]]
                                            idx, = np.where(np.isfinite(ddata))
                                            if len(idx) > 0:
                                                # There is data
                                                total[di, zk, yj, xi] += 1
                                                idx, = np.where(ddata
                                                                > gate[di])
                                                if len(idx) > 0:
                                                    # Data above gate
                                                    hits[di, zk, yj, xi] += 1

    # All of the loading and storing data is done
    prob = hits / total

    # Make nicer dictionary output
    output = {}
    for i, label in enumerate(data_label):
        output[label] = {'prob': prob[i, :, :, :], 'count': total[i, :, :, :]}
        if return_bins:
            output[label]['bin_x'] = binx
            output[label]['bin_y'] = biny
            output[label]['bin_z'] = binz

    # Clean up
    del iter_insts, loop_sat_y, loop_sat_z
    return output
