#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022, pysat development team
# Full license can be found in License.md
# DOI:10.5281/zenodo.3475493
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""pysatSeasons.

pysatSeasons is a pysat module that provides the interface to perform seasonal
analysis on data managed by pysat.  These analysis methods are independent of
instrument type.

Main Features
-------------
- Seasonal averaging routine for 1D and 2D data.
- Occurrence probability routines, daily or by orbit.
- Scatterplot of data_label(s) as functions of labelx,y
    over a season.

"""

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

# Import key modules and skip F401 testing in flake8
from pysatSeasons._core import to_xarray_dataset  # noqa: F401
from pysatSeasons import avg  # noqa: F401
from pysatSeasons import occur_prob  # noqa: F401
from pysatSeasons import plot  # noqa: F401

# set version
__version__ = metadata.version('pysatSeasons')
