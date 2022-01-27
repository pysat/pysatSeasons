<div align="left">
        <img height="0" width="0px">
        <img width="20%" src="https://raw.githubusercontent.com/pysat/pysatSeasons/main/poweredbypysat.png" alt="pysat" title="pysat"</img>
</div>

# pysatSeasons
[![PyPI Package latest release](https://img.shields.io/pypi/v/pysatSeasons.svg)](https://pypi.python.org/pypi/pysatSeasons)
[![Build Status](https://github.com/pysat/pysatSeasons/actions/workflows/main.yml/badge.svg)](https://github.com/pysat/pysatSeasons/actions/workflows/main.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/pysat/pysatSeasons/badge.svg?branch=main)](https://coveralls.io/github/pysat/pysatSeasons?branch=main)
[![DOI](https://zenodo.org/badge/209365329.svg)](https://zenodo.org/badge/latestdoi/209365329)



This code will handle the seasonal analysis routines for pysat.  It is currently a work in progress, and will eventually replace the pysat.ssnl module in pysat.

pysatSeasons allows users to run basic seasonal analysis over N-dimensional datasets managed through the pysat code.

Main Features
-------------
- Seasonal averaging routine for 1D and 2D data.
- Occurrence probability routines, daily or by orbit.
- Scatterplot of data_label(s) as functions of labelx,y
    over a season.


# Installation

### Prerequisites

pysatSeasons uses common Python modules, as well as modules developed by
and for the Space Physics community.  

| Common modules | Community modules |
| -------------- | ----------------- |
| matplotlib     | pysat             |
| numpy          |                   |
| pandas         |                   |


## GitHub Installation

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
