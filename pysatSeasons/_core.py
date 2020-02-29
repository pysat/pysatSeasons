import pandas as pds

def computational_form(data):
    """
    Repackages numbers, Series, or DataFrames

    Regardless of input format, mathematical operations may be performed on the
    output via the same pandas mechanisms.

    This method may be particularly useful in analysis methods that aim to be
    instrument independent. pysat.Instrument objects can package data in a
    variety of ways within a DataFrame, depending upon the scientific data
    source. Thus, a variety of data types will be encountered by instrument
    independent methods and computational_form method may reduce the effort
    required to support more generalized processing.

    Parameters
    ----------
    data : pandas.Series
        Series of numbers, Series, DataFrames

    Returns
    -------
    pandas.Series, DataFrame, or Panel
        repacked data, aligned by indices, ready for calculation
    """

    if isinstance(data.iloc[0], pds.DataFrame):
        dslice = pds.Panel.from_dict(dict([(i, data.iloc[i])
                                       for i in range(len(data))]))
    elif isinstance(data.iloc[0], pds.Series):
        dslice = pds.DataFrame(data.tolist())
        dslice.index = data.index
    else:
        dslice = data
    return dslice
