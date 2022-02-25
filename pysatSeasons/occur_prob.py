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


def daily2D(inst, bin1, label1, bin2, label2, data_label, gate,
            returnBins=False):
    """2D Daily Occurrence Probability of `data_label` > `gate` over a season.

    If `data_label` is greater than `gate` at least once per day,
    then a 100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument to use for calculating occurrence probability.
    binX : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where X = 1, 2.
    labelX : str
        Identifies data product for binX, where X = 1, 2.
    data_label : list of str
        Identifies data product(s) to calculate occurrence probability
        e.g. inst[data_label].
    gate : list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins : bool
        If True, also return arrays with values of bin edges, useful for pcolor.
        (default=False)

    Returns
    -------
    occur_prob : dictionary
        A dict of dicts indexed by `data_label`. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of days with any
        data; 'bin_x' and 'bin_y' are also returned if requested. Note that
        arrays are organized for direct plotting, y values along rows, x along
        columns.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    return _occurrence2D(inst, bin1, label1, bin2, label2, data_label, gate,
                         by_orbit=False, returnBins=returnBins)


def by_orbit2D(inst, bin1, label1, bin2, label2, data_label, gate,
               returnBins=False):
    """2D Occurrence Probability of `data_label` orbit-by-orbit over a season.

    If `data_label` is greater than `gate` at least once per orbit, then a
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument to use for calculating occurrence probability.
    binX : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where X = 1, 2.
    labelX: str
        Identifies data product for binX, where X = 1, 2.
    data_label: list of str
        Data product label(s) to calculate occurrence probability.
    gate: list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins: bool
        If True, return arrays with values of bin edges, useful for pcolor.
        (default=False)

    Returns
    -------
    occur_prob : dictionary
        A dict of dicts indexed by data_label. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of orbits with
        any data; 'bin_x' and 'bin_y' are also returned if requested. Note that
        arrays are organized for direct plotting, y values along rows, x along
        columns.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    return _occurrence2D(inst, bin1, label1, bin2, label2, data_label, gate,
                         by_orbit=True, returnBins=returnBins)


def _occurrence2D(inst, bin1, label1, bin2, label2, data_label, gate,
                  by_orbit=False, returnBins=False):
    """2D Occurrence Probability of `data_label` orbit-by-orbit over a season.

    If `data_label` is greater than `gate` at least once per orbit, then a
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument to use for calculating occurrence probability.
    binX : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where X = 1, 2.
    labelX: str
        Identifies data product for binX, where X = 1, 2.
    data_label: list of str
        Data product label(s) to calculate occurrence probability.
    gate: list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    by_orbit : bool
        If True, then occurrence probability determined by orbit rather than
        by day. (default=False)
    returnBins: bool
        If True, return arrays with values of bin edges, useful for pcolor.
        (default=False)

    Returns
    -------
    occur_prob : dictionary
        A dict of dicts indexed by data_label. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of orbits with
        any data; 'bin_x' and 'bin_y' are also returned if requested. Note that
        arrays are organized for direct plotting, y values along rows, x along
        columns.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    if not hasattr(data_label, '__iter__'):
        raise ValueError(' '.join(('Data label must be list-like group of',
                                   'variable names.')))
    if not hasattr(gate, '__iter__'):
        raise ValueError(' '.join(('Gate levels must be list-like group of',
                                   'variable names.')))
    if len(gate) != len(data_label):
        raise ValueError('Must have a gate value for each data_label')

    # Create bins
    binx = np.linspace(bin1[0], bin1[1], bin1[2] + 1)
    biny = np.linspace(bin2[0], bin2[1], bin2[2] + 1)

    numx = len(binx) - 1
    numy = len(biny) - 1
    numz = len(data_label)
    arrx = np.arange(numx)
    arry = np.arange(numy)
    arrz = np.arange(numz)

    # Create arrays to store all values
    total = np.zeros((numz, numy, numx))
    hits = np.zeros((numz, numy, numx))
    if by_orbit:
        inst.load(date=inst.bounds[0][0])
        iterator = inst.orbits
    else:
        iterator = inst

    for i, inst in enumerate(iterator):
        loop_inst = inst.copy()
        if len(inst.data) != 0:
            xind = np.digitize(inst.data[label1], binx) - 1
            for xi in arrx:
                xindex, = np.where(xind == xi)
                if len(xindex) > 0:
                    loop_inst.data = inst[xindex]
                    yind = np.digitize(loop_inst[label2], biny) - 1
                    for yj in arry:
                        yindex, = np.where(yind == yj)
                        if len(yindex) > 0:
                            # Iterate over the different data_labels
                            for zk in arrz:
                                # indlab = yData.columns.get_loc(data_label[zk])
                                zdata = loop_inst[yindex, data_label[zk]]
                                if np.any(np.isfinite(zdata)):
                                    total[zk, yj, xi] += 1.
                                    if np.any(zdata > gate[zk]):
                                        hits[zk, yj, xi] += 1.

    # All of the loading and storing data is done. Calculate probability.
    prob = hits / total

    # Make nicer dictionary output
    output = {}
    for i, label in enumerate(data_label):
        output[label] = {'prob': prob[i, :, :], 'count': total[i, :, :]}
        if returnBins:
            output[label]['bin_x'] = binx
            output[label]['bin_y'] = biny

    # Clean up
    del iterator
    return output


def daily3D(inst, bin1, label1, bin2, label2, bin3, label3,
            data_label, gate, returnBins=False):
    """3D Daily Occurrence Probability of `data_label` > `gate` over a season.

    If `data_label` is greater than `gate` at least once per day,
    then a 100% occurrence probability results. Season delineated by
    the bounds attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument to use for calculating occurrence probability.
    binX : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where X = 1, 2.
    labelX : str
        Identifies data product for binX, where X = 1, 2.
    data_label : list of str
        Identifies data product(s) to calculate occurrence probability
        e.g. inst[data_label].
    gate : list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins : bool
        If True, also return arrays with values of bin edges, useful for pcolor.
        (default=False)

    Returns
    -------
    occur_prob : dictionary
        A dict of dicts indexed by `data_label`. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of days with any
        data; 'bin_x', 'bin_y', and 'bin_z' are also returned if requested.
        Note that arrays are organized for direct plotting, z, y, x.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.

    """

    return _occurrence3D(inst, bin1, label1, bin2, label2, bin3, label3,
                         data_label, gate, returnBins=returnBins,
                         by_orbit=False)


def by_orbit3D(inst, bin1, label1, bin2, label2, bin3, label3,
               data_label, gate, returnBins=False):
    """3D Occurrence Probability of `data_label` orbit-by-orbit over a season.

    If `data_label` is greater than `gate` at least once per orbit, then a
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Prob = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument to use for calculating occurrence probability.
    binX : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where X = 1, 2.
    labelX : str
        Identifies data product for binX, where X = 1, 2.
    data_label : list of str
        Identifies data product(s) to calculate occurrence probability
        e.g. inst[data_label].
    gate : list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins : bool
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

    return _occurrence3D(inst, bin1, label1, bin2, label2, bin3, label3,
                         data_label, gate, returnBins=returnBins,
                         by_orbit=True)


def _occurrence3D(inst, bin1, label1, bin2, label2, bin3, label3,
                  data_label, gate, returnBins=False, by_orbit=False):
    """3D Occurrence Probability of `data_label` orbit-by-orbit over a season.

    If `data_label` is greater than `gate` at least once per iteration, then a
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Probability = (# of times with at least one hit) / (# of times in bin)

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument to use for calculating occurrence probability.
    binX : array-like
        List holding [min, max, number of bins] or array-like containing
        bin edges, where X = 1, 2.
    labelX: str
        Identifies data product for binX, where X = 1, 2.
    data_label: list of str
        Data product label(s) to calculate occurrence probability.
    gate: list of values
        Values that `data_label` must achieve to be counted as an occurrence.
    returnBins: bool
        If True, return arrays with values of bin edges, useful for pcolor.
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

    if not hasattr(data_label, '__iter__'):
        raise ValueError(' '.join(('Data label must be list-like group of',
                         'variable names.')))
    if not hasattr(gate, '__iter__'):
        raise ValueError(' '.join(('Gate levels must be list-like group of',
                                   'variable names.')))
    if len(gate) != len(data_label):
        raise ValueError('Must have a gate value for each data_label')

    # Create bins
    binx = np.linspace(bin1[0], bin1[1], bin1[2] + 1)
    biny = np.linspace(bin2[0], bin2[1], bin2[2] + 1)
    binz = np.linspace(bin3[0], bin3[1], bin3[2] + 1)

    numx = len(binx) - 1
    numy = len(biny) - 1
    numz = len(binz) - 1
    numd = len(data_label)

    # Create array to store all values before taking median
    yarr = np.arange(numy)
    xarr = np.arange(numx)
    zarr = np.arange(numz)
    darr = np.arange(numd)

    total = np.zeros((numd, numz, numy, numx))
    hits = np.zeros((numd, numz, numy, numx))

    if by_orbit:
        iterator = inst.orbits
    else:
        iterator = inst

    # Iterate over given season
    for i, sat in enumerate(iterator):
        loop_sat_y = sat.copy()
        loop_sat_z = sat.copy()
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
                                    for di in darr:
                                        # indlab = zData.columns.get_loc(data_label[di])
                                        ddata = loop_sat_z[zindex,
                                                           data_label[di]]
                                        idx, = np.where(np.isfinite(ddata))
                                        if len(idx) > 0:
                                            total[di, zk, yj, xi] += 1
                                            idx, = np.where(ddata > gate[di])
                                            if len(idx) > 0:
                                                hits[di, zk, yj, xi] += 1

    # All of the loading and storing data is done
    prob = hits / total

    # Make nicer dictionary output
    output = {}
    for i, label in enumerate(data_label):
        output[label] = {'prob': prob[i, :, :, :], 'count': total[i, :, :, :]}
        if returnBins:
            output[label]['bin_x'] = binx
            output[label]['bin_y'] = biny
            output[label]['bin_z'] = binz

    # Clean up
    del iterator
    return output
