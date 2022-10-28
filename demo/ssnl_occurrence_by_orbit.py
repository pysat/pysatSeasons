"""Seasonal occurence by orbit demo code.

Demonstrate iteration over an instrument data set by orbit and determining
the occurrence probability of an event occurring.

"""

import datetime as dt
import numpy as np
import os

import matplotlib.pyplot as plt

import pysat
import pysatNASA
import pysatSeasons

# Ensure all pysatNASA data plugins are registered with pysat. Only needs
# to be performed once per installation/upgrade.
pysat.utils.registry.register_by_module(pysatNASA.instruments)

# Set the directory where the plots will be saved. Setting nothing will put
# the plots in the current directory
results_dir = ''

# Select C/NOFS VEFI DC magnetometer data, use longitude to determine where
# there are changes in the orbit (local time info not in file)
orbit_info = {'index': 'longitude', 'kind': 'longitude'}
vefi = pysat.Instrument(platform='cnofs', name='vefi', tag='dc_b',
                        clean_level=None, orbit_info=orbit_info)


# Define function to remove flagged values
def filter_vefi(inst):
    """Filter all instrument data by flag criteria."""

    idx, = np.where(inst['B_flag'] == 0)
    inst.data = inst[idx]
    return


# Attach filtering function to `vefi` object.
vefi.custom_attach(filter_vefi)

# Set limits on dates analysis will cover, inclusive
start = dt.datetime(2010, 5, 9)
stop = dt.datetime(2010, 5, 15)

# Check if data already on system, if not, download.
if len(vefi.files[start:stop]) < (stop - start).days:
    vefi.download(start, stop)

# Specify the analysis time limits using `bounds`, otherwise all VEFI DC
# data will be processed.
vefi.bounds = (start, stop)

# Perform occurrence probability calculation.
# Any data added by custom functions is available within analysis below.
ans = pysatSeasons.occur_prob.by_orbit2D(vefi, [0, 360, 144], 'longitude',
                                         [-13, 13, 104], 'latitude',
                                         ['dB_mer'], [0.], return_bins=True)

# A dict indexed by data_label is returned.
ans = ans['dB_mer']

# Plot occurrence probability
f, axarr = plt.subplots(2, 1, sharex=True, sharey=True)

# Mask for locations not observed.
masked = np.ma.array(ans['prob'], mask=np.isnan(ans['prob']))

# Plot occurrence probability
im = axarr[0].pcolor(ans['bin_x'], ans['bin_y'], masked)
axarr[0].set_title('Occurrence Probability Delta-B Meridional > 0')
axarr[0].set_ylabel('Latitude')
axarr[0].set_yticks((-13, -10, -5, 0, 5, 10, 13))
axarr[0].set_ylim((ans['bin_y'][0], ans['bin_y'][-1]))
plt.colorbar(im, ax=axarr[0], label='Occurrence Probability')

# Plot number of orbits per bin.
im = axarr[1].pcolor(ans['bin_x'], ans['bin_y'], ans['count'])
axarr[1].set_title('Number of Orbits in Bin')
axarr[1].set_xlabel('Longitude')
axarr[1].set_xticks((0, 60, 120, 180, 240, 300, 360))
axarr[1].set_xlim((ans['bin_x'][0], ans['bin_x'][-1]))
axarr[1].set_ylabel('Latitude')
plt.colorbar(im, ax=axarr[1], label='Counts')

f.tight_layout()
plt.savefig(os.path.join(results_dir, 'ssnl_occurrence_by_orbit_demo'))
plt.close()
