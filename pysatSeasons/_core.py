import pandas as pds
import xarray as xr


def to_xarray_dataset(data):
    """
    Repackage input data into xarray.Dataset

    Regardless of input format, mathematical operations may be performed on the
    output via the same xarray mechanisms.

    This method may be particularly useful in analysis methods that aim to be
    instrument independent. pysat.Instrument objects can package data in a
    variety of ways within a DataFrame, depending upon the scientific data
    source. Thus, a variety of data types will be encountered by instrument
    independent methods and to_xarray_dataset method may reduce the effort
    required to support more generalized processing.

    Parameters
    ----------
    data : pds.Series, pds.DataFrame, xr.DataArray, xr.DataSet, or list-like
           of the same or numbers.
        List-like of numbers, Series, DataFrames, or Datasets to be combined
        into a single Dataset.

    Returns
    -------
    output : xr.Dataset
        Repacked data, aligned by indices. If data is a list of multidimensional
        objects then output will have a corresponding new dimension
        'pysat_binning' to reflect that organization. If data is a list of
        numbers, then output will have a single associated variable, `data`.
        Otherwise, variable names are retained from the input data.

    """

    if isinstance(data, pds.DataFrame):
        output = data.to_xarray()
    elif isinstance(data, pds.Series):
        output = xr.Dataset()
        output[data.name] = data.to_xarray()
    elif isinstance(data, xr.Dataset):
        output = data
    elif isinstance(data, xr.DataArray):
        output = xr.Dataset()
        output[data.name] = data
    elif isinstance(data[0], xr.Dataset):
        # Combine list-like of Datasets
        vars = data[0].data_vars.keys()
        output = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            output[var] = xr.concat([item[var] for item in data],
                                    'pysat_binning')

    elif isinstance(data[0], xr.DataArray):
        # Combine list-like of DataArrays
        vars = [data[0].name]
        output = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            output[var] = xr.concat(data, 'pysat_binning')

    elif isinstance(data[0], pds.DataFrame):
        # Convert list-like of DataFrames
        info = [xr.Dataset.from_dataframe(item) for item in data]

        vars = info[0].data_vars.keys()
        output = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            output[var] = xr.concat([item[var] for item in info],
                                    'pysat_binning')

    elif isinstance(data[0], pds.Series):
        # Convert list-like of pds.Series
        vars = [data[0].name]
        output = xr.Dataset()

        # Combine all info for each variable into a single data
        # array.
        for var in vars:
            output[var] = xr.concat([xr.DataArray.from_series(item)
                                     for item in data], 'pysat_binning')

    else:
        output = xr.Dataset()
        output['data'] = data

    return output
