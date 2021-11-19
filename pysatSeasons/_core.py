import pandas as pds
import xarray as xr


def computational_form(data):
    """
    Repackages input data into xarray.Dataset

    Regardless of input format, mathematical operations may be performed on the
    output via the same xarray mechanisms.

    This method may be particularly useful in analysis methods that aim to be
    instrument independent. pysat.Instrument objects can package data in a
    variety of ways within a DataFrame, depending upon the scientific data
    source. Thus, a variety of data types will be encountered by instrument
    independent methods and computational_form method may reduce the effort
    required to support more generalized processing.

    Parameters
    ----------
    data : pds.Series, pds.DataFrame, xr.DataArray, xr.DataSet, or list-like
        List-like of numbers, Series, DataFrames, or Datasets

    Returns
    -------
    xarray.Dataset
        repacked data, aligned by indices, ready for calculation

    """
    if isinstance(data, pds.DataFrame):
        dslice = data.to_xarray()
    elif isinstance(data, pds.Series):
        dslice = xr.Dataset()
        dslice[data.name] = data.to_xarray()
    elif isinstance(data, xr.Dataset):
        dslice = data
    elif isinstance(data, xr.DataArray):
        dslice = xr.Dataset()
        dslice[data.name] = data

    elif isinstance(data[0], xr.Dataset):
        # Combine multiple datasets into one
        vars = data[0].data_vars.keys()
        dslice = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            dslice[var] = xr.concat([item[var] for item in data],
                                    'pysat_binning')

    elif isinstance(data[0], xr.DataArray):
        # Combine multiple datasets into one
        vars = [data[0].name]
        dslice = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            dslice[var] = xr.concat([item[var] for item in data],
                                    'pysat_binning')

    elif isinstance(data[0], pds.DataFrame):
        # Convert data to xarray
        info = [xr.Dataset.from_dataframe(item) for item in data]

        vars = info[0].data_vars.keys()
        dslice = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            dslice[var] = xr.concat([item[var] for item in info],
                                    'pysat_binning')

    elif isinstance(data[0], pds.Series):
        # Combine multiple datasets into one
        vars = [data[0].name]
        dslice = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            dslice[var] = xr.concat([xr.DataArray.from_series(item)
                                     for item in data], 'pysat_binning')

    else:
        dslice = data

    return dslice
