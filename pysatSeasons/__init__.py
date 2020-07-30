"""
pysatSeasons is a pysat module that provides
the interface to perform seasonal analysis on
data managed by pysat.  These analysis methods
are independent of instrument type.

Main Features
-------------
- Seasonal averaging routine for 1D and 2D data.
- Occurrence probability routines, daily or by orbit.
- Scatterplot of data_label(s) as functions of labelx,y
    over a season.

"""

import os

# Import key modules and skip F401 testing in flake8
from pysatSeasons import occur_prob  # noqa: F401
from pysatSeasons import avg  # noqa: F401
from pysatSeasons import plot  # noqa: F401
from pysatSeasons._core import computational_form  # noqa: F401

# set version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'version.txt')) as version_file:
    __version__ = version_file.read().strip()
