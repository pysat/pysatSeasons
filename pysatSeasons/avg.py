#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022, pysat development team
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Instrument independent seasonal averaging routine.

Supports bin averaging N-dimensional data over 1D and 2D bin distributions.

"""

import numpy as np
import pandas as pds
import warnings

import pysat
import pysatSeasons as pyseas


def median1D(const, bin1, label1, data_label, auto_bin=True, returnData=None,
             return_data=False):
    """Cacluate a 1D median of nD `data_label` over a season and `label1`.

    Parameters
    ----------
    const: Constellation or Instrument
        Constellation or Instrument object.
    bin1: array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    label1: str
        Identifies data product for bin1.
    data_label: list-like
        Strings identifying data product(s) to be averaged.
    auto_bin: bool
        If True, function will create bins from the min, max and
        number of bins. If false, bin edges must be manually entered in `bin1`.
        (default=True)
    returnData : bool or NoneType
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
        Deprecated in favor of `return_data`.
        (default=None)
    return_data : bool
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
        (default=False)

    Returns
    -------
    median : dict
        1D median accessed by `data_label` as a function of `label1`
        over the season delineated by bounds of passed instrument objects.
        Also includes 'count' and 'avg_abs_dev' as well as the values of
        the bin edges in 'bin_x'. If returnData True, then binned data
        stored under 'data' under `data_label`.

    Note
    ----
    The range of dates to be loaded, and the cadence used to load data over
    that range, is controlled by the `const.bounds` attribute.

    """

    # `const` is either an Instrument or a Constellation, and we want to
    #  iterate over it. If it's a Constellation, then we can do that as is,
    #  but if it's an Instrument, we just have to put that Instrument
    #  into a Constellation.
    if isinstance(const, pysat.Instrument):
        const = pysat.Constellation(instruments=[const])
    elif not isinstance(const, pysat.Constellation):
        raise ValueError("Parameter must be an Instrument or a Constellation.")

    # Create the boundaries used for sorting into bins
    if auto_bin:
        binx = np.linspace(bin1[0], bin1[1], bin1[2] + 1)
    else:
        binx = np.array(bin1)

    # How many bins are used
    numx = len(binx) - 1

    # How many different data products
    numz = len(data_label)

    # Create array to store all values before taking median.
    # The indices of the bins used for looping.
    xarr = np.arange(numx)
    zarr = np.arange(numz)

    # 3D array:  stores the data that is sorted into each bin - in a list.
    ans = [[[] for i in xarr] for k in zarr]

    for inst1 in const.instruments:
        # Iterate over instrument season, loading successive
        # data between start and end bounds.
        for inst in inst1:
            # Collect data in bins for averaging
            if len(inst.data) != 0:
                # Sort the data into bins (x) based on `label1`
                # (stores bin indexes in xind)
                xind = np.digitize(inst[label1], binx) - 1

                # For each possible x index
                for xi in xarr:
                    # Get the indices of those pieces of data in that bin
                    xindex, = np.where(xind == xi)

                    if len(xindex) > 0:
                        # For each data product label zk
                        for zk in zarr:
                            # Take the data (already filtered by x), select the
                            # data, put it in a list, and extend the list.
                            ans[zk][xi].extend(inst[xindex, data_label[zk]])

    # Calculate the 1D median
    return _calc_1d_median(ans, data_label, binx, xarr, zarr, numx, numz,
                           returnData, return_data)


def median2D(const, bin1, label1, bin2, label2, data_label,
             returnData=None, auto_bin=True, return_data=False):
    """Return a 2D average of nD `data_label` over season and `label1` `label2`.

    Parameters
    ----------
    const : pysat.Constellation or Instrument
    bin1, bin2 : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    label1, label2:  str
        Identifies data product for binning.
    data_label : list-like
        Strings identifying data product(s) to be averaged.
    returnData : bool or NoneType
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
        Deprecated in favor of `return_data`.
        (default=None)
    auto_bin : bool
        If True, function will create bins from the min, max and
        number of bins. If false, bin edges must be manually entered in `bin*`.
        (default=True)
    return_data : bool
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
        (default=False)

    Returns
    -------
    median : dict
        2D median accessed by data_label as a function of label1 and label2
        over the season delineated by bounds of passed instrument objects.
        Also includes 'count' and 'avg_abs_dev' as well as the values of
        the bin edges in 'bin_x' and 'bin_y'.

    Note
    ----
    The range of dates to be loaded, and the cadence used to load data over
    that range, is controlled by the `const.bounds` attribute.

    """

    # `const` is either an Instrument or a Constellation, and we want to
    #  iterate over it. If it's a Constellation, then we can do that as is,
    #  but if it's an Instrument, we just have to put that Instrument
    #  into a Constellation.
    if isinstance(const, pysat.Instrument):
        const = pysat.Constellation(instruments=[const])
    elif not isinstance(const, pysat.Constellation):
        raise ValueError("Parameter must be an Instrument or a Constellation.")

    # Create the boundaries used for sorting into bins
    if auto_bin:
        binx = np.linspace(bin1[0], bin1[1], bin1[2] + 1)
        biny = np.linspace(bin2[0], bin2[1], bin2[2] + 1)
    else:
        binx = np.array(bin1)
        biny = np.array(bin2)

    # How many bins are used
    numx = len(binx) - 1
    numy = len(biny) - 1

    # How many different data products
    numz = len(data_label)

    # Create array to store all values before taking median.
    # The indices of the bins/data products. Used for looping.
    yarr = np.arange(numy)
    xarr = np.arange(numx)
    zarr = np.arange(numz)

    # 3D array:  stores the data that is sorted into each bin - in a list.
    ans = [[[[] for i in xarr] for j in yarr] for k in zarr]

    # Iterate over Instruments
    for inst1 in const.instruments:
        # Copy instrument to provide data source independent access
        yinst = inst1.copy()

        # Iterate over instrument season.
        for inst in inst1:
            # Collect data in bins for averaging
            if not inst.empty:
                # Sort the data into bins (x) based on label 1
                # (stores bin indexes in xind)
                xind = np.digitize(inst[label1], binx) - 1

                # For each possible x index
                for xi in xarr:
                    # Get the indices of those pieces of data in that bin.
                    xindex, = np.where(xind == xi)

                    if len(xindex) > 0:
                        # Look up the data along y (label2) at that set of
                        # indices (a given x).
                        yinst.data = inst[xindex]

                        # Digitize that, to sort data into bins along y
                        # (label2) (get bin indexes)
                        yind = np.digitize(yinst[label2], biny) - 1

                        # For each possible y index
                        for yj in yarr:
                            # Select data with this y index (and we already
                            # filtered for this x index)
                            yindex, = np.where(yind == yj)

                            if len(yindex) > 0:
                                # For each data product label zk
                                for zk in zarr:
                                    # Take the data (already filtered by x),
                                    # filter it by y, select the data product,
                                    # put it in a list, and extend the deque.
                                    ans[zk][yj][xi].extend(
                                        yinst[yindex, data_label[zk]])

    return _calc_2d_median(ans, data_label, binx, biny, xarr, yarr, zarr,
                           numx, numy, numz, returnData, return_data)


def _calc_2d_median(ans, data_label, binx, biny, xarr, yarr, zarr, numx,
                    numy, numz, returnData=None, return_data=False):
    """Return a 2D average of nD `data_label` over season and `label1` `label2`.

    Parameters
    ----------
    ans : list of lists
        List of lists containing binned data. Provided by `median2D`.
    data_label : str
        Label for data to be binned and averaged.
    binx, biny : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where * = 1, 2.
    xarr, yarr, zarr : list-like
        Indexing array along bin directions x, y, and data dimension z.
    numx, numy, numz : int
        Number of elements along xarr, yarr, zarr.
    returnData : bool or NoneType
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
        Deprecated in favor of `return_data`.
        (default=None)
    return_data : bool
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
        (default=False)

    Returns
    -------
    median : dict
        2D median accessed by `data_label` as a function of `label1` and
        `label2` over the season delineated by bounds of passed Instrument
        objects. Also includes 'count' and 'avg_abs_dev' as well as the
        values of the bin edges in 'bin_x' and 'bin_y'.

    """

    if returnData is not None:
        return_data = returnData
        warnings.warn(''.join(['"returnData" has been deprecated in favor of ',
                               '"return_data". Assigning input "returnData" ',
                               'to "return_data". ']), DeprecationWarning,
                      stacklevel=2)

    # set up output arrays
    median_ans = [[[[] for i in xarr] for j in yarr] for k in zarr]
    count_ans = [[[[] for i in xarr] for j in yarr] for k in zarr]
    dev_ans = [[[[] for i in xarr] for j in yarr] for k in zarr]

    # All of the loading and storing data is done, though the data
    # could be of different types. Make all of them xarray datasets.
    dim = 'pysat_binning'
    for zk in zarr:
        scalar_avg = True
        for yj in yarr:
            for xi in xarr:

                count_ans[zk][yj][xi] = len(ans[zk][yj][xi])

                if len(ans[zk][yj][xi]) > 0:
                    data = pyseas.to_xarray_dataset(ans[zk][yj][xi])

                    # Higher order data has the 'pysat_binning' dim
                    if dim in data.dims:
                        if len(data.dims) > 1:
                            scalar_avg = False
                        # All data is prepped. Perform calculations.
                        median_ans[zk][yj][xi] = data.median(dim=dim)

                        dev_ans[zk][yj][xi] = data - median_ans[zk][yj][xi]
                        dev_ans[zk][yj][xi] = dev_ans[zk][yj][xi].map(np.abs)
                        dev_ans[zk][yj][xi] = dev_ans[zk][yj][xi].median(
                            dim=dim)
                    else:
                        median_ans[zk][yj][xi] = data.median()

                        dev_ans[zk][yj][xi] = data - median_ans[zk][yj][xi]
                        dev_ans[zk][yj][xi] = dev_ans[zk][yj][xi].map(np.abs)
                        dev_ans[zk][yj][xi] = dev_ans[zk][yj][xi].median()

        if scalar_avg:
            # Store current structure
            temp_median = median_ans[zk]
            temp_count = count_ans[zk]
            temp_dev = dev_ans[zk]

            # Create 2D numpy arrays for new storage
            median_ans[zk] = np.full((numy, numx), np.nan)
            count_ans[zk] = np.full((numy, numx), np.nan)
            dev_ans[zk] = np.full((numy, numx), np.nan)

            # Store data
            for yj in yarr:
                for xi in xarr:
                    if len(temp_median[yj][xi]) > 0:
                        key = [name for name in temp_median[yj][xi].data_vars]
                        median_ans[zk][yj, xi] = temp_median[yj][xi][key[0]]
                        count_ans[zk][yj, xi] = temp_count[yj][xi]
                        dev_ans[zk][yj, xi] = temp_dev[yj][xi][key[0]]

    # Prepare output
    output = {}
    for i, label in enumerate(data_label):
        output[label] = {'median': median_ans[i],
                         'count': count_ans[i],
                         'avg_abs_dev': dev_ans[i],
                         'bin_x': binx,
                         'bin_y': biny}

        if return_data:
            output[label]['data'] = ans[i]

    return output


def mean_by_day(inst, data_label):
    """Mean of `data_label` by day over `Instrument.bounds`.

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object to perform mean upon.
    data_label : str
        Data product label to be averaged.

    Returns
    -------
    mean : pandas.Series
        Mean of `data_label` indexed by day.

    Note
    ----
    The range of dates to be loaded, and the cadence used to load data over
    that range, is controlled by the `inst.bounds` attribute.

    """
    return _core_mean(inst, data_label, by_day=True)


def mean_by_orbit(inst, data_label):
    """Mean of `data_label` by orbit over Instrument.bounds.

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object to perform mean upon.
    data_label : str
        Data product label to be averaged.

    Returns
    -------
    mean : pandas.Series
        Mean of `data_label` indexed by start of each orbit.

    Note
    ----
    The range of dates to be loaded, and the cadence used to load data over
    that range, is controlled by the `inst.bounds` attribute.

    """
    return _core_mean(inst, data_label, by_orbit=True)


def mean_by_file(inst, data_label):
    """Mean of `data_label` by orbit over Instrument.bounds.

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object to perform mean upon.
    data_label : str
        Data product label to be averaged.

    Returns
    -------
    mean : pandas.Series
        Mean of `data_label` indexed by start of each file.

    Note
    ----
    The range of dates to be loaded, and the cadence used to load data over
    that range, is controlled by the `inst.bounds` attribute.

    """
    return _core_mean(inst, data_label, by_file=True)


def _core_mean(inst, data_label, by_orbit=False, by_day=False, by_file=False):
    """Mean of `data_label` by different iterations over `inst.bounds`.

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object to perform mean upon.
    data_label : str
        Data product label to be averaged.
    by_orbit : bool
        If True, iterate by orbit. (default=False)
    by_day : bool
        If True, iterate by day. (default=False)
    by_file : bool
        If True, iterate by file. (default=False)

    Returns
    -------
    mean : pandas.Series
        Mean of `data_label` indexed by start of each file.

    Note
    ----
    The range of dates to be loaded, and the cadence used to load data over
    that range, is controlled by the `inst.bounds` attribute.

    """

    if by_orbit:
        iterator = inst.orbits
    elif by_day or by_file:
        iterator = inst
    else:
        raise ValueError('A choice must be made, by day, file, or orbit')

    # Create empty series to hold result
    mean_val = pds.Series(dtype=np.float64)

    # Iterate over season, calculate the mean
    for linst in iterator:
        if not linst.empty:
            # Compute mean using xarray functions and store

            data = linst[data_label]
            data = pyseas.to_xarray_dataset(data)
            if 'time' in data.dims:
                epoch_dim = 'time'
            elif 'Epoch' in data.dims:
                epoch_dim = 'Epoch'
            data = data.dropna(dim=epoch_dim)

            if by_orbit or by_file:
                date = linst.index[0]
            else:
                date = linst.date

            # Perform average
            mean_val[date] = data.mean(dim=data[data_label].dims[0],
                                       skipna=True)[data_label].values

    del iterator
    return mean_val


def _calc_1d_median(ans, data_label, binx, xarr, zarr, numx, numz,
                    returnData=None, return_data=False):
    """Return a 1D average of nD `data_label` over season.

    Parameters
    ----------
    ans: list of lists
        List of lists containing binned data. Provided by `median1D`.
    data_label : str
        Label for data to be binned and averaged.
    binx: array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges.
    xarr, zarr: list-like
        Indexing array along bin direction x and data dimension z.
    numx, numz: int
        Number of elements along xarr, zarr.
    returnData : bool or NoneType
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
        Deprecated in favor of `return_data`.
        (default=None)
    return_data : bool
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
        (default=False)

    Returns
    -------
    median : dict
        1D median accessed by `data_label` as a function of `label1` and
        `label2` over the season delineated by bounds of passed Instrument
        objects. Also includes 'count' and 'avg_abs_dev' as well as the
        values of the bin edges in 'bin_x' and 'bin_y'.

    """

    if returnData is not None:
        return_data = returnData
        warnings.warn(''.join(['"returnData" has been deprecated in favor of ',
                               '"return_data". Assigning input "returnData" ',
                               'to "return_data". ']), DeprecationWarning,
                      stacklevel=2)

    # Set up output arrays
    median_ans = [[None for i in xarr] for k in zarr]
    count_ans = [[None for i in xarr] for k in zarr]
    dev_ans = [[None for i in xarr] for k in zarr]

    # All of the loading and storing data is done, though the data
    # could be of different types. Make all of them xarray datasets.
    dim = 'pysat_binning'
    for zk in zarr:
        scalar_avg = True
        for xi in xarr:
            if len(ans[zk][xi]) > 0:
                count_ans[zk][xi] = len(ans[zk][xi])

                data = pyseas.to_xarray_dataset(ans[zk][xi])

                # Higher order data has the 'pysat_binning' dim
                if dim in data.dims:
                    if len(data.dims) > 1:
                        scalar_avg = False
                    # All data is prepped. Perform calculations.
                    median_ans[zk][xi] = data.median(dim=dim)

                    dev_ans[zk][xi] = data - median_ans[zk][xi]
                    dev_ans[zk][xi] = dev_ans[zk][xi].map(np.abs)
                    dev_ans[zk][xi] = dev_ans[zk][xi].median(dim=dim)
                else:
                    median_ans[zk][xi] = data.median()

                    dev_ans[zk][xi] = data - median_ans[zk][xi]
                    dev_ans[zk][xi] = dev_ans[zk][xi].map(np.abs)
                    dev_ans[zk][xi] = dev_ans[zk][xi].median()

        if scalar_avg:
            # Store current structure
            temp_median = median_ans[zk]
            temp_count = count_ans[zk]
            temp_dev = dev_ans[zk]

            # Create 1D numpy arrays for new storage
            median_ans[zk] = np.full((numx), np.nan)
            count_ans[zk] = np.full((numx), np.nan)
            dev_ans[zk] = np.full((numx), np.nan)

            # Store data
            for xi in xarr:
                if len(temp_median[xi]) > 0:
                    key = [name for name in temp_median[xi].data_vars]
                    median_ans[zk][xi] = temp_median[xi][key[0]]
                    count_ans[zk][xi] = temp_count[xi]
                    dev_ans[zk][xi] = temp_dev[xi][key[0]]

    # Prepare output
    output = {}
    for i, label in enumerate(data_label):
        output[label] = {'median': median_ans[i],
                         'count': count_ans[i],
                         'avg_abs_dev': dev_ans[i],
                         'bin_x': binx}

        if return_data:
            output[label]['data'] = ans[i]

    return output
