"""Seasonal analysis demo using COSMIC RO profiles and IVM in situ data.

"""

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from packaging import version as pack_version
import pandas as pds
from scipy.stats import mode


import apexpy
import pysat
import pysatCDAAC
import pysatNASA
import pysatSeasons


if pack_version(pysatCDAAC.__version__) <= pack_version('0.0.2'):
    estr = ' '.join(['This demo is written expecting xarray support for ',
                     'COSMIC data. Unfortunately, this is not supported by ',
                     'the currently installed version. Please see the demo ',
                     'code in pysatSeasons v0.1.3 for COSMIC support when in ',
                     'pandas data format.'])
    raise(ValueError, estr)


def add_magnetic_coordinates(inst):
    """Add magnetic longitude and latitude for COSMIC density maximum location.

    Adds 'qd_lon' and 'qd_lat' to `inst`.

    Parameters
    ----------
    inst : pysat.Instrument
        'COSMIC' Instrument object

    """

    apex = apexpy.Apex(date=inst.date)

    # Convert geographic profile location to magnetic location
    lats, lons = apex.geo2qd(inst['edmaxlat'], inst['edmaxlon'],
                             inst['edmaxalt'])

    # Longitudes between 0 - 360.
    idx, = np.where(lons < 0)
    lons[idx] += 360.

    # Add data and metadata to Instrument object
    inst['edmax_qd_lat'] = lats
    notes_str = ''.join(['Obtained from apexpy by transforming ',
                         '`edmaxlat` and `edmaxlon` into quasi-dipole ',
                         'coordinates.'])
    meta_data = {inst.meta.labels.units: 'degrees',
                 inst.meta.labels.name: 'Quasi-Dipole Latitude',
                 inst.meta.labels.notes: notes_str,
                 inst.meta.labels.fill_val: np.nan,
                 inst.meta.labels.min_val: -90.,
                 inst.meta.labels.max_val: 90.}
    inst.meta['edmax_qd_lat'] = meta_data

    inst['edmax_qd_lon'] = lons
    meta_data = {inst.meta.labels.units: 'degrees',
                 inst.meta.labels.name: 'Quasi-Dipole Longitude',
                 inst.meta.labels.notes: notes_str,
                 inst.meta.labels.fill_val: np.nan,
                 inst.meta.labels.min_val: 0.,
                 inst.meta.labels.max_val: 360.}
    inst.meta['edmax_qd_lon'] = meta_data

    return


def restrict_abs_values(inst, label, max_val):
    """Restrict absolute qd_lat values.

    Parameters
    ----------
    inst : pysat.Instrument
        'COSMIC' Instrument object
    label : str
        Label for variable to restrict.
    max_val : float
        Absolute maximum value of `label`. Values greater
        than `max_val` are removed from `inst`.

    """

    inst.data = inst[np.abs(inst[label]) <= max_val]
    return


def filter_values(inst, label, val_range=(0., 15.)):
    """Filter values to those in `label` that are within `val_range` limits.

    Parameters
    ----------
    inst : pysat.Instrument
        'COSMIC' Instrument object
    label : str
        Label for variable to restrict.
    val_range : tuple/list of floats
        Minimum and maximum values allowed. Times where `label`
        outside `val_range` are removed from `inst`. (default=(0., 15.))

    """

    inst.data = inst[(inst[label].values >= val_range[0])
                     & (inst[label].values <= val_range[1])]
    return


def add_log_density(inst):
    """Add the log of COSMIC maximum density, 'edmax'.

    Parameters
    ----------
    inst : pysat.Instrument
        'COSMIC' 'GPS' object.

    """

    # Assign data
    inst['lognm'] = np.log10(inst['edmax'])

    # Assign metadata
    notes_str = ''.join(['Log base 10 of `edmax` variable.'])
    meta_data = {inst.meta.labels.units: 'Log(N/cc)',
                 inst.meta.labels.name: 'Log Density',
                 inst.meta.labels.notes: notes_str,
                 inst.meta.labels.fill_val: np.nan,
                 inst.meta.labels.min_val: -np.inf,
                 inst.meta.labels.max_val: np.inf}
    inst.meta['lognm'] = meta_data

    return


def add_scale_height(inst):
    """Calculate topside scale height using observed electron density profiles.

    Parameters
    ----------
    inst : pysat.Instrument
        'COSMIC' 'GPS' Instrument.

    """

    output = inst['edmaxlon'].copy()

    for i, profile in enumerate(inst['ELEC_dens']):
        profile = profile[(profile
                          >= (1. / np.e) * inst[i, 'edmax'])
                          & (profile.coords["MSL_alt"] >= inst[i, 'edmaxalt'])]

        # Want the first altitude where density drops below NmF2/e.
        if len(profile) > 10:
            i1 = profile.coords["MSL_alt"][1:]
            i2 = profile.coords["MSL_alt"][0:-1]
            modeDiff = mode(i1.values - i2.values)[0][0]

            # Ensure there are no gaps, if so, remove all data above gap
            idx, = np.where((i1.values - i2.values) > 2 * modeDiff)
            if len(idx) > 0:
                profile = profile[0:idx[0]]

            # Ensure there are no null values in the middle of profile.
            idx, = np.where(profile.isnull())
            if len(idx) > 0:
                profile = profile[0:idx[0]]

        if len(profile) > 10:
            # Make sure density at highest altitude is near Nm/e
            if profile[-1] / profile[0] < 0.4:
                alt = profile.coords["MSL_alt"]
                alt_diff = alt[-1] - alt["MSL_alt"][0]
                if alt_diff >= 500:
                    alt_diff = np.nan
                output[i] = alt_diff
            else:
                output[i] = np.nan
        else:
            output[i] = np.nan

    inst['thf2'] = output

    return


# Register all instruments in pysatCDAAC and pysatNASA. Only required once
# per install.
pysat.utils.registry.register_by_module(pysatCDAAC.instruments)
pysat.utils.registry.register_by_module(pysatNASA.instruments)

# Dates for demo
ssn_days = 67
startDate = dt.datetime(2009, 12, 21) - pds.DateOffset(days=ssn_days)
stopDate = dt.datetime(2009, 12, 21) + pds.DateOffset(days=ssn_days)

# Instantiate IVM Object
ivm = pysat.Instrument(platform='cnofs', name='ivm', tag='',
                       clean_level='clean')

# Restrict measurements to those near geomagnetic equator.
ivm.custom_attach(restrict_abs_values, args=['mlat', 25.])

# Perform seasonal average
ivm.bounds = (startDate, stopDate)
ivmResults = pysatSeasons.avg.median2D(ivm, [0, 360, 24], 'alon',
                                       [0, 24, 24], 'mlt',
                                       ['ionVelmeridional'])

# Create COSMIC instrument object. Engage supported keyword `altitude_bin`
# to bin all altitude profiles into 3 km increments.
cosmic = pysat.Instrument(platform='cosmic', name='gps', tag='ionprf',
                          clean_level='clean', altitude_bin=3)

# Apply custom functions to all data that is loaded through cosmic
cosmic.custom_attach(add_magnetic_coordinates)

# Select locations near the magnetic equator
cosmic.custom_attach(filter_values, args=['edmax_qd_lat', (-10., 10.)])

# Take the log of NmF2 and add to the dataframe
cosmic.custom_attach(add_log_density)

# Calculates the height above hmF2 to reach Ne < NmF2/e
cosmic.custom_attach(add_scale_height)

# Perform a bin average of multiple COSMIC data products, from startDate
# through stopDate. A mixture of 1D and 2D data is averaged.
cosmic.bounds = (startDate, stopDate)
cosmicResults = pysatSeasons.avg.median2D(cosmic, [0, 360, 24], 'edmax_qd_lon',
                                          [0, 24, 24], 'edmaxlct',
                                          ['ELEC_dens', 'edmaxalt',
                                           'lognm', 'thf2'])

# The work is done, plot the results!

# Make IVM and COSMIC plots
f, axarr = plt.subplots(4, sharex=True, sharey=True, figsize=(8.5, 11))
cax = []

# Meridional ion drift average
merDrifts = ivmResults['ionVelmeridional']['median']
x_arr = ivmResults['ionVelmeridional']['bin_x']
y_arr = ivmResults['ionVelmeridional']['bin_y']

# Mask out NaN values
masked = np.ma.array(merDrifts, mask=np.isnan(merDrifts))

# Plot, NaN values are white.
# Note how the data returned from the median function is in plot order.
cax.append(axarr[0].pcolor(x_arr, y_arr,
                           masked, vmax=30., vmin=-30.,
                           edgecolors='none'))
axarr[0].set_ylim(0, 24)
axarr[0].set_yticks([0, 6, 12, 18, 24])
axarr[0].set_xlim(0, 360)
axarr[0].set_xticks(np.arange(0, 420, 60))
axarr[0].set_ylabel('Magnetic Local Time')
axarr[0].set_title('IVM Meridional Ion Drifts')
cbar0 = f.colorbar(cax[0], ax=axarr[0])
cbar0.set_label('Ion Drift (m/s)')

maxDens = cosmicResults['lognm']['median']
cx_arr = cosmicResults['lognm']['bin_x']
cy_arr = cosmicResults['lognm']['bin_y']

# Mask out NaN values
masked = np.ma.array(maxDens, mask=np.isnan(maxDens))

# Plot, NaN values are white
cax.append(axarr[1].pcolor(cx_arr, cy_arr,
           masked, vmax=6.1, vmin=4.8,
           edgecolors='none'))
axarr[1].set_title('COSMIC Log Density Maximum')
axarr[1].set_ylabel('Solar Local Time')
cbar1 = f.colorbar(cax[1], ax=axarr[1])
cbar1.set_label('Log Density')

maxAlt = cosmicResults['edmaxalt']['median']

# Mask out NaN values
masked = np.ma.array(maxAlt, mask=np.isnan(maxAlt))

# Plot, NaN values are white
cax.append(axarr[2].pcolor(cx_arr, cy_arr,
           masked, vmax=375., vmin=200.,
           edgecolors='none'))
axarr[2].set_title('COSMIC Altitude Density Maximum')
axarr[2].set_ylabel('Solar Local Time')
cbar = f.colorbar(cax[2], ax=axarr[2])
cbar.set_label('Altitude (km)')


maxTh = cosmicResults['thf2']['median']

# Mask out NaN values
masked = np.ma.array(maxTh, mask=np.isnan(maxTh))

# Plot, NaN values are white
cax.append(axarr[3].pcolor(cx_arr, cy_arr, masked,
                           vmax=225., vmin=75., edgecolors='none'))
axarr[3].set_title('COSMIC Topside Scale Height')
axarr[3].set_ylabel('Solar Local Time')
cbar = f.colorbar(cax[3], ax=axarr[3])
cbar.set_label('Scale Height (km)')
axarr[3].set_xlabel('Apex Longitude')
f.tight_layout()
f.savefig('ssnl_median_ivm_cosmic_1d.png')


# Make COSMIC profile plots
# 6 pages of plots, 4 plots per page
for k in np.arange(6):
    f, axarr = plt.subplots(4, sharex=True, figsize=(8.5, 11))
    # Iterate over a group of four sectors at a time (4 plots per page)
    for (j, sector) in enumerate(list(zip(*cosmicResults['ELEC_dens']['median']))
                                 [k * 4:(k + 1) * 4]):
        # Iterate over all local times within longitude sector.
        # Data is returned from the median routine in plot order, [y, x]
        # instead of [x,y].
        for (i, ltview) in enumerate(sector):
            if np.isfinite(ltview):
                # Plot a given longitude/local time profile
                temp = pds.DataFrame(ltview['ELEC_dens'])

                # Produce a grid covering plot region
                # (y values determined by profile)
                xx, yy = np.meshgrid(np.array([i, i + 1]),
                                     np.arange(len(temp.index.values) + 1))
                filtered = ma.array(np.log10(temp.values),
                                    mask=pds.isnull(temp))
                graph = axarr[j].pcolormesh(xx, yy, filtered,
                                            vmin=3., vmax=6.5)

        cbar = f.colorbar(graph, ax=axarr[j])
        cbar.set_label('Log Density')
        axarr[j].set_xlim(0, 24)
        axarr[j].set_ylim(0., 300.)
        axarr[j].set_yticks([50., 100., 150., 200., 250.],
                            [150., 300., 450., 600., 750.])
        axarr[j].set_ylabel('Altitude (km)')
        axarr[j].set_title('Apex Longitudes %i-%i' %
                           (4 * k * 15 + j * 15, 4 * k * 15 + (j + 1) * 15))

    axarr[-1].set_xticks([0., 6., 12., 18., 24.])
    axarr[-1].set_xlabel('Solar Local Time of Profile Maximum Density')
    f.tight_layout()
    f.savefig('cosmic_part{}.png'.format(k))
