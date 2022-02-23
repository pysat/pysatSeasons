# -*- coding: utf-8 -*-
"""
Instrument independent seasonal averaging routine. Supports averaging
1D and 2D data.
"""

import numpy as np
import pandas as pds

import pysat
import pysatSeasons as ssnl


def median1D(const, bin1, label1, data_label, auto_bin=True, returnData=False):
    """Return a 1D median of nD `data_label` over a season and `label1`.

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
    returnData : bool
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.

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
    #  into something that will yield that Instrument, like a list.
    if isinstance(const, pysat.Instrument):
        const = [const]
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

    # 3d array:  stores the data that is sorted into each bin - in a list.
    ans = [[[] for i in xarr] for k in zarr]

    for inst1 in const:
        # Iterate over instrument season. Probably iterates by date but that
        # all depends on the configuration of that particular instrument.
        # Either way, it iterates over the instrument, loading successive
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
                           returnData)


def median2D(const, bin1, label1, bin2, label2, data_label,
             returnData=False, auto_bin=True):
    """Return a 2D average of nD `data_label` over season and `label1` `label2`.

    Parameters
    ----------
    const : pysat.Constellation or Instrument
    bin* : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where * = 1, 2.
    label*:  str
        Identifies data product for bin*, where * = 1, 2.
    data_label : list-like
        Strings identifying data product(s) to be averaged.
    returnData : bool
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.
    auto_bin : bool
        If True, function will create bins from the min, max and
        number of bins. If false, bin edges must be manually entered in `bin*`.

    Returns
    -------
    median : dictionary
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
    #  into something that will yield that Instrument, like a list.
    if isinstance(const, pysat.Instrument):
        const = [const]
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

    # Create array to store all values before taking median
    # The indices of the bins/data products. Used for looping.
    yarr = np.arange(numy)
    xarr = np.arange(numx)
    zarr = np.arange(numz)

    # 3d array:  stores the data that is sorted into each bin - in a list.
    ans = [[[[] for i in xarr] for j in yarr] for k in zarr]

    for inst1 in const:
        # Iterate over instrument season. Probably iterates by date but that
        # all depends on the configuration of that particular instrument.
        # Either way, it iterates over the instrument, loading successive
        # data between start and end bounds.
        for inst in inst1:
            # Collect data in bins for averaging
            if not inst.empty:
                # Sort the data into bins (x) based on label 1
                # (stores bin indexes in xind)
                xind = np.digitize(inst[label1], binx) - 1
                yinst = inst.copy()
                # For each possible x index
                for xi in xarr:
                    # Get the indices of those pieces of data in that bin.
                    xindex, = np.where(xind == xi)

                    if len(xindex) > 0:
                        # Look up the data along y (label2) at that set of
                        # indices (a given x).
                        yinst.data = inst[xindex]

                        # digitize that, to sort data into bins along y
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
                           numx, numy, numz, returnData)


def _calc_2d_median(ans, data_label, binx, biny, xarr, yarr, zarr, numx,
                    numy, numz, returnData=False):
    """Return a 2D average of nD `data_label` over season and `label1` `label2`.

    Parameters
    ----------
    ans : list of lists
        List of lists containing binned data. Provided by `median2D`.
    bin* : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where * = 1, 2.
    *arr : list-like
        Indexing array along bin directions x, y, and data dimension z.
    num* : int
        Number of elements along *arr.
    returnData : bool
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.

    Returns
    -------
    median : dictionary
        2D median accessed by `data_label` as a function of `label1` and
        `label2` over the season delineated by bounds of passed Instrument
        objects. Also includes 'count' and 'avg_abs_dev' as well as the
        values of the bin edges in 'bin_x' and 'bin_y'.

    """
    # set up output arrays
    medianAns = [[[[] for i in xarr] for j in yarr] for k in zarr]
    countAns = [[[[] for i in xarr] for j in yarr] for k in zarr]
    devAns = [[[[] for i in xarr] for j in yarr] for k in zarr]

    # All of the loading and storing data is done, though the data
    # could be of different types. Make all of them xarray datasets.
    dim = 'pysat_binning'
    for zk in zarr:
        scalar_avg = True
        for yj in yarr:
            for xi in xarr:

                countAns[zk][yj][xi] = len(ans[zk][yj][xi])

                if len(ans[zk][yj][xi]) > 0:
                    data = ssnl.to_xarray_dataset(ans[zk][yj][xi])

                    # Higher order data has the 'pysat_binning' dim
                    if dim in data.dims:
                        scalar_avg = False
                        # All data is prepped. Perform calculations.
                        medianAns[zk][yj][xi] = data.median(dim=dim)

                        devAns[zk][yj][xi] = data - medianAns[zk][yj][xi]
                        devAns[zk][yj][xi] = devAns[zk][yj][xi].map(np.abs)
                        devAns[zk][yj][xi] = devAns[zk][yj][xi].median(dim=dim)
                    else:
                        medianAns[zk][yj][xi] = data.median()

                        devAns[zk][yj][xi] = data - medianAns[zk][yj][xi]
                        devAns[zk][yj][xi] = devAns[zk][yj][xi].map(np.abs)
                        devAns[zk][yj][xi] = devAns[zk][yj][xi].median()

        if scalar_avg:
            # Store current structure
            temp_median = medianAns[zk]
            temp_count = countAns[zk]
            temp_dev = devAns[zk]

            # Create 2D numpy arrays for new storage
            medianAns[zk] = np.full((numy, numx), np.nan)
            countAns[zk] = np.full((numy, numx), np.nan)
            devAns[zk] = np.full((numy, numx), np.nan)

            # Store data
            for yj in yarr:
                for xi in xarr:
                    if len(temp_median[yj][xi]) > 0:
                        medianAns[zk][yj, xi] = temp_median[yj][xi]['data']
                        countAns[zk][yj, xi] = temp_count[yj][xi]
                        devAns[zk][yj, xi] = temp_dev[yj][xi]['data']

    # Prepare output
    output = {}
    for i, label in enumerate(data_label):
        output[label] = {'median': medianAns[i],
                         'count': countAns[i],
                         'avg_abs_dev': devAns[i],
                         'bin_x': binx,
                         'bin_y': biny}

        if returnData:
            output[label]['data'] = ans[i]

    return output


def mean_by_day(inst, data_label):
    """Mean of `data_label` by day over Instrument.bounds

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
        If True, iterate by orbit.
    by_day : bool
        If True, iterate by day.
    by_file : bool
        If True, iterate by fil.

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
            data = data.dropna()

            if by_orbit or by_file:
                date = linst.index[0]
            else:
                date = linst.date

            # Perform average
            data = ssnl.to_xarray_dataset(data)
            mean_val[date] = data.mean(dim=data[data_label].dims[0],
                                       skipna=True)[data_label].values

    del iterator
    return mean_val


def _calc_1d_median(ans, data_label, binx, xarr, zarr, numx, numz,
                    returnData=False):
    """Return a 1D average of nD `data_label` over season.

    Parameters
    ----------
    ans: list of lists
        List of lists containing binned data. Provided by `median1D`.
    binx: array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where * = 1, 2.
    *arr: list-like
        Indexing array along bin direction x and data dimension z.
    num*: int
        Number of elements along *arr.
    returnData : bool
        If True, also return binned data used to calculate the average in
        the output dictionary as 'data', in addition to the statistical outputs.

    Returns
    -------
    median : dictionary
        1D median accessed by `data_label` as a function of `label1` and
        `label2` over the season delineated by bounds of passed Instrument
        objects. Also includes 'count' and 'avg_abs_dev' as well as the
        values of the bin edges in 'bin_x' and 'bin_y'.

    """
    # Set up output arrays
    medianAns = [[None for i in xarr] for k in zarr]
    countAns = [[None for i in xarr] for k in zarr]
    devAns = [[None for i in xarr] for k in zarr]

    # All of the loading and storing data is done, though the data
    # could be of different types. Make all of them xarray datasets.
    dim = 'pysat_binning'
    for zk in zarr:
        scalar_avg = True
        for xi in xarr:
            if len(ans[zk][xi]) > 0:
                countAns[zk][xi] = len(ans[zk][xi])

                data = ssnl.to_xarray_dataset(ans[zk][xi])

                # Higher order data has the 'pysat_binning' dim
                if dim in data.dims:
                    scalar_avg = False
                    # All data is prepped. Perform calculations.
                    medianAns[zk][xi] = data.median(dim=dim)

                    devAns[zk][xi] = data - medianAns[zk][xi]
                    devAns[zk][xi] = devAns[zk][xi].map(np.abs)
                    devAns[zk][xi] = devAns[zk][xi].median(dim=dim)
                else:
                    medianAns[zk][xi] = data.median()

                    devAns[zk][xi] = data - medianAns[zk][xi]
                    devAns[zk][xi] = devAns[zk][xi].map(np.abs)
                    devAns[zk][xi] = devAns[zk][xi].median()

        if scalar_avg:
            # Store current structure
            temp_median = medianAns[zk]
            temp_count = countAns[zk]
            temp_dev = devAns[zk]

            # Create 1D numpy arrays for new storage
            medianAns[zk] = np.full((numx), np.nan)
            countAns[zk] = np.full((numx), np.nan)
            devAns[zk] = np.full((numx), np.nan)

            # Store data
            for xi in xarr:
                if len(temp_median[xi]) > 0:
                    medianAns[zk][xi] = temp_median[xi]['data']
                    countAns[zk][xi] = temp_count[xi]
                    devAns[zk][xi] = temp_dev[xi]['data']

    # Prepare output
    output = {}
    for i, label in enumerate(data_label):
        output[label] = {'median': medianAns[i],
                         'count': countAns[i],
                         'avg_abs_dev': devAns[i],
                         'bin_x': binx}

        if returnData:
            output[label]['data'] = ans[i]

    return output
