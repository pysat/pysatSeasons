import pandas as pds
import xarray as xr


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
    data : array-like
        Series of numbers, Series, or DataFrames

    Returns
    -------
    pandas.Series, DataFrame, or xarray.Dataset
        repacked data, aligned by indices, ready for calculation

    """

    if isinstance(data.iloc[0], pds.DataFrame):
        # Convert data to xarray
        info = [xr.Dataset.from_dataframe(temp)
                for temp in data]

        vars = info[0].data_vars.keys()
        dslice = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            dslice[var] = xr.concat([item[var] for item in info],
                                    'pysat_binning')
    elif isinstance(data.iloc[0], pds.Series):
        dslice = pds.DataFrame(data.tolist())
        dslice.index = data.index
    else:
        dslice = data

    return dslice
