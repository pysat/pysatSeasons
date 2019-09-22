<div align="left">
        <img height="0" width="0px">
        <img width="20%" src="/poweredbypysat.png" alt="pysat" title="pysat"</img>
</div>

# pysatSeasons
[![Build Status](https://travis-ci.org/pysat/pysatSeasons.svg?branch=master)](https://travis-ci.org/pysat/pysatSeasons)
[![Coverage Status](https://coveralls.io/repos/github/pysat/pysatSeasons/badge.svg?branch=master)](https://coveralls.io/github/pysat/pysatSeasons?branch=master)

This code will handle the seasonal analysis routines for pysat.  It is currently a work in progress, and will eventually replace the pysat.ssnl module in pysat.

pysatSeasons allows users to run basic seasonal analysis over N-dimensional datasets managed through the pysat code.

Main Features
-------------
- Seasonal averaging routine for 1D and 2D data.
- Occurrence probability routines, daily or by orbit.
- Scatterplot of data_label(s) as functions of labelx,y
    over a season.


# Installation

First, checkout the repository:

```
  git clone https://github.com/pysat/pysatSeasons.git
```

Change directories into the repository folder and run the setup.py file.  For
a local install use the "--user" flag after "install".

```
  cd pysatSeasons/
  python setup.py install
```
