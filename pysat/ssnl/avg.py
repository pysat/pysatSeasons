# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
"""
Instrument independent seasonal averaging routine. Supports averaging
1D and 2D data.
"""

import pysat
import numpy as np
import pandas as pds
import collections

def median2D(inst, bin1, label1, bin2, label2, data_label, 
             returnData=False):
    """Return a 2D average of data_label over a season and label1, label2.

    Parameters
    ----------
        bin#: [min, max, number of bins]
        label#: string 
            identifies data product for bin#
        data_label: list-like 
            contains strings identifying data product(s) to be averaged

    Returns
    -------
    median : dictionary
        2D median accessed by data_label as a function of label1 and label2
        over the season delineated by bounds of passed instrument objects.
        Also includes 'count' and 'avg_abs_dev' as well as the values of
        the bin edges in 'bin_x' and 'bin_y'.
    
    """

    # create bins
    binx = np.linspace(bin1[0], bin1[1], bin1[2]+1)
    biny = np.linspace(bin2[0], bin2[1], bin2[2]+1)

    numx = len(binx)-1
    numy = len(biny)-1
    numz = len(data_label)

    # create array to store all values before taking median
    yarr = np.arange(numy)
    xarr = np.arange(numx)
    zarr = np.arange(numz)
    ans = [ [ [collections.deque() for i in xarr] for j in yarr] for k in zarr]

    # set up output arrays
    medianAns = [ [ [ None for i in xarr] for j in yarr] for k in zarr]
    countAns = [ [ [ None for i in xarr] for j in yarr] for k in zarr]
    devAns = [ [ [ None for i in xarr] for j in yarr] for k in zarr]    

    # do loop to iterate over instrument season
    for inst in inst:
        # collect data in bins for averaging
        if len(inst.data) != 0:
            xind = np.digitize(inst.data[label1], binx)-1
            for xi in xarr:
                xindex, = np.where(xind==xi)
                if len(xindex) > 0:
                    yData = inst.data.iloc[xindex]
                    yind = np.digitize(yData[label2], biny)-1
                    for yj in yarr:
                        yindex, = np.where(yind==yj)
                        if len(yindex) > 0:
                            for zk in zarr:
                                ans[zk][yj][xi].extend( yData.ix[yindex,data_label[zk]].tolist() )

    # all of the loading and storing data is done
    # determine what kind of data is stored
    # if just numbers, then use numpy arrays to store data
    # if the data is a more generalized object, use lists to store data
    # need to find first bin with data
    dataType = [None for i in np.arange(numz)]
    for zk in zarr:
        breakNow=False
        for yj in yarr:
            for xi in xarr:
                if len(ans[zk][yj][xi]) > 0:
                    dataType[zk] = type(ans[zk][yj][xi][0]) 
                    breakNow = True
                    break 
            if breakNow:
                break

    # determine if normal number objects are being used or if there
    # are more complicated objects
    objArray = [False]*len(zarr)
    for i,thing in enumerate(dataType):
         if thing == pds.core.series.Series:
            objArray[i] = 'S'
         elif thing == pds.core.frame.DataFrame:
            objArray[i] = 'F'
         else:
             # other, simple scalaRs
            objArray[i] = 'R'

    objArray = np.array(objArray)

    # if some pandas data series are returned in average, return a list
    objidx, = np.where(objArray == 'S')
    if len(objidx) > 0:
        for zk in zarr[objidx]:
            for yj in yarr:
                for xi in xarr:
                    if len(ans[zk][yj][xi]) > 0:
                        ans[zk][yj][xi] = list(ans[zk][yj][xi])
                        medianAns[zk][yj][xi] =  pds.DataFrame(ans[zk][yj][xi] ).median(axis=0)
                        countAns[zk][yj][xi] = len(ans[zk][yj][xi])
                        devAns[zk][yj][xi] = pds.DataFrame([abs(temp - medianAns[zk][yj][xi]) for temp in ans[zk][yj][xi] ] ).median(axis=0)
                                                                    
    # if some pandas DataFrames are returned in average, return a list
    objidx, = np.where(objArray == 'F')
    if len(objidx) > 0:
        for zk in zarr[objidx]:
            for yj in yarr:
                for xi in xarr:                    
                    if len(ans[zk][yj][xi]) > 0:
                        ans[zk][yj][xi] = list(ans[zk][yj][xi])
                        countAns[zk][yj][xi] = len(ans[zk][yj][xi])
                        test = pds.Panel.from_dict(dict([(i,temp) for i,temp in enumerate(ans[zk][yj][xi]) ]) )
                        medianAns[zk][yj][xi] = test.median(axis=0)
                        devAns[zk][yj][xi] = (test.subtract(medianAns[zk][yj][xi], axis=0)).abs().median(axis=0, skipna=True)
                                                                                                      
    objidx, = np.where(objArray == 'R')
    if len(objidx) > 0:
        for zk in zarr[objidx]:
            medianAns[zk] = np.zeros((numy, numx))*np.nan
            countAns[zk] = np.zeros((numy, numx))*np.nan
            devAns[zk] = np.zeros((numy, numx))*np.nan
            for yj in yarr:
                for xi in xarr:
                    # convert deque storing data into numpy array
                    ans[zk][yj][xi] = np.array(ans[zk][yj][xi])
                    # filter out an NaNs in the arrays
                    idx, = np.where(np.isfinite(ans[zk][yj][xi]))
                    ans[zk][yj][xi] = (ans[zk][yj][xi])[idx]
                    # perform median averaging
                    if len(idx) > 0:
                        medianAns[zk][yj,xi] = np.median(ans[zk][yj][xi])
                        countAns[zk][yj,xi] = len(ans[zk][yj][xi])
                        devAns[zk][yj,xi] = np.median(abs(ans[zk][yj][xi] - medianAns[zk][yj,xi]))

    # prepare output
    output = {}
    for i,label in enumerate(data_label):
        output[label] = {'median': medianAns[i], 
                        'count':countAns[i],
                        'avg_abs_dev':devAns[i],
                        'bin_x': binx,
                        'bin_y': biny}

        if returnData:
            output[label]['data'] = ans[i]

    return output

# simple averaging through multiple iterations

def mean_by_day(inst, data_label):
    """Mean of data_label by day over Instrument.bounds

    Parameters
    ----------
    data_label : string
        string identifying data product to be averaged

    Returns
    -------
    mean : pandas Series
        simple mean of data_label indexed by day

    """
    return _core_mean(inst, data_label, by_day=True)

def mean_by_orbit(inst, data_label):
    """Mean of data_label by orbit over Instrument.bounds

    Parameters
    ----------
    data_label : string
        string identifying data product to be averaged

    Returns
    -------
    mean : pandas Series
        simple mean of data_label indexed by start of each orbit

    """
    return _core_mean(inst, data_label, by_orbit=True)

def mean_by_file(inst, data_label):
    """Mean of data_label by orbit over Instrument.bounds

    Parameters
    ----------
    data_label : string
        string identifying data product to be averaged

    Returns
    -------
    mean : pandas Series
        simple mean of data_label indexed by start of each file

    """
    return _core_mean(inst, data_label, by_file=True)

def _core_mean(inst, data_label, by_orbit=False, by_day=False, by_file=False):

    if by_orbit:
        iterator = inst.orbits
    elif by_day or by_file:
        iterator = inst
    else:
        raise ValueError('A choice must be made, by day, file, or orbit')

    # create empty series to hold result
    mean_val = pds.Series()
    # iterate over season, calculate the mean
    for inst in iterator:
        if not inst.data.empty:
                # compute mean absolute using pandas functions and store
                # data could be an image, or lower dimension, account for 2D and lower
                data = inst[data_label]
                data.dropna(inplace=True)

                if by_orbit or by_file:
                   date = inst.data.index[0]
                else:
                   date = inst.date
                # perform average
                mean_val[date] = pysat.utils.computational_form(data).mean(axis=0, skipna=True)

    del iterator
    return mean_val
