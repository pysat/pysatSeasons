import datetime as dt
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pds

import apexpy
import pysat
import pysatSeasons

# dates for demo
ssnDays = 67
startDate = dt.datetime(2009, 12, 21) - pds.DateOffset(days=ssnDays)
stopDate = dt.datetime(2009, 12, 21) + pds.DateOffset(days=ssnDays)


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
    lats, lons = apex(inst['edmaxlat'], inst['edmaxlon'])

    # Longitudes between 0 - 360.
    idx, = np.where(lons < 0)
    lons[idx] += 360.

    # Add data and metadata to Instrument object
    inst['qd_lat'] = lats
    notes_str = ''.join(['Obtained from apexpy by transforming ',
                         '`edmaxlat` and `edmaxlon` into quasi-dipole ',
                         'coordinates.'])
    meta_data = {inst.meta.labels.units: 'degrees',
                 inst.meta.labels.name: 'Quasi-Dipole Latitude',
                 inst.meta.labels.notes: notes_str,
                 inst.meta.labels.fill_val: np.nan,
                 inst.meta.labels.min_val: -90.,
                 inst.meta.labels.max_val: 90.}
    inst.meta['qd_lat'] = meta_data

    inst['qd_lon'] = lons
    meta_data = {inst.meta.labels.units: 'degrees',
                 inst.meta.labels.name: 'Quasi-Dipole Longitude',
                 inst.meta.labels.notes: notes_str,
                 inst.meta.labels.fill_val: np.nan,
                 inst.meta.labels.min_val: 0.,
                 inst.meta.labels.max_val: 360.}
    inst.meta['qd_lon'] = meta_data

    return


def restrictMLAT(inst, max_mlat=25.):
    """Restrict absolute MLAT values.

    Parameters
    ----------
    inst : pysat.Instrument
        'COSMIC' Instrument object
    max_mlat : float
        Absolute value of maximum magnetic latitude. Positions greater
        than `max_mlat` are removed from `inst`. (default=25.)

    """
    inst.data = inst.data[np.abs(inst['mlat']) <= max_mlat]
    return


def filterMLAT(inst, mlat_range=(0., 15.)):
    """Filter absolute MLAT values to those >=, or <= values in `mlat_range`.

    Parameters
    ----------
    inst : pysat.Instrument
        'COSMIC' Instrument object
    mlat_range : tuple/list of floats
        Absolute value of minimum and maximum magnetic latitudes. Positions
        outside `mlat_range` are removed from `inst`. (default=(0., 15.))

    """

    inst.data = inst.data[(np.abs(inst['mlat']) >= mlat_range[0])
                          & (np.abs(inst['mlat']) <= mlat_range[1])]
    return


def add_log_density(inst):
    """Add the log of COSMIC maximum density, 'edmax'.

    Parameters
    ----------
    inst : pysat.Instrument
        'COSMIC' 'GPS' object.

    """

    inst['lognm'] = np.log10(inst['edmax'])

    return


def add_scale_height(cosmic):
    from scipy.stats import mode

    output = cosmic['edmaxlon'].copy()
    output.name = 'thf2'

    for i, profile in enumerate(cosmic['profiles']):
        profile = profile[(profile['ELEC_dens']
                          >= (1. / np.e) * cosmic['edmax'].iloc[i])
                          & (profile.index >= cosmic['edmaxalt'].iloc[i])]
        # Want the first altitude where density drops below NmF2/e.
        # First, resample such that we know all altitudes in between samples
        # are there.
        if len(profile) > 10:
            i1 = profile.index[1:]
            i2 = profile.index[0:-1]
            modeDiff = mode(i1.values - i2.values)[0][0]
            profile = profile.reindex(np.arange(profile.index[0],
                                                profile.index[-1] + modeDiff,
                                                modeDiff))
            # Ensure there are no gaps, if so, remove all data above gap
            idx, = np.where(profile['ELEC_dens'].isnull())
            if len(idx) > 0:
                profile = profile.iloc[0:idx[0]]

        if len(profile) > 10:
            # Make sure density at highest altitude is near Nm/e
            if (profile['ELEC_dens'].iloc[-1] / profile['ELEC_dens'].iloc[0]
                    < 0.4):
                altDiff = profile.index.values[-1] - profile.index.values[0]
                if altDiff >= 500:
                    altDiff = np.nan
                output[i] = altDiff
            else:
                output[i] = np.nan
        else:
            output[i] = np.nan

    return output


# Instantiate IVM Object
ivm = pysat.Instrument(platform='cnofs', name='ivm', tag='',
                       clean_level='clean')

# Restrict measurements to those near geomagnetic equator.
ivm.custom_attach(restrictMLAT, kwargs={'max_mlat': 25.})

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
cosmic.custom_attach(filterMLAT, kwargs={'mlatRange': (0., 10.)})

# Take the log of NmF2 and add to the dataframe
cosmic.custom_attach(addlogNm)

# Calculates the height above hmF2 to reach Ne < NmF2/e
cosmic.custom_attach(addTopsideScaleHeight)

# Perform a bin average of multiple COSMIC data products, from startDate
# through stopDate. A mixture of 1D and 2D data is averaged.
cosmic.bounds = (startDate, stopDate)
cosmicResults = pysatSeasons.avg.median2D(cosmic, [0, 360, 24], 'apex_long',
                                          [0, 24, 24], 'edmaxlct',
                                          ['profiles', 'edmaxalt',
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
    for (j, sector) in enumerate(list(zip(*cosmicResults['profiles']['median']))
                                 [k * 4:(k + 1) * 4]):
        # Iterate over all local times within longitude sector.
        # Data is returned from the median routine in plot order, [y, x]
        # instead of [x,y].
        for (i, ltview) in enumerate(sector):
            if ltview is not None:
                # Plot a given longitude/local time profile
                temp = pds.DataFrame(ltview['ELEC_dens'])

                # Produce a grid covering plot region
                # (y values determined by profile)
                xx, yy = np.meshgrid(np.array([i, i + 1]), temp.index.values)
                filtered = ma.array(np.log10(temp.values),
                                    mask=pds.isnull(temp))
                graph = axarr[j].pcolormesh(xx, yy, filtered,
                                            vmin=3., vmax=6.5)

        cbar = f.colorbar(graph, ax=axarr[j])
        cbar.set_label('Log Density')
        axarr[j].set_xlim(0, 24)
        axarr[j].set_ylim(50., 700.)
        axarr[j].set_yticks([50., 200., 350., 500., 650.])
        axarr[j].set_ylabel('Altitude (km)')
        axarr[j].set_title('Apex Longitudes %i-%i' %
                           (4 * k * 15 + j * 15, 4 * k * 15 + (j + 1) * 15))

    axarr[-1].set_xticks([0., 6., 12., 18., 24.])
    axarr[-1].set_xlabel('Solar Local Time of Profile Maximum Density')
    f.tight_layout()
    f.savefig('cosmic_part{}.png'.format(k))
