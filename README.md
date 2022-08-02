<div align="left">
        <img height="0" width="0px">
        <img width="20%" src="https://raw.githubusercontent.com/pysat/pysatSeasons/main/docs/images/logo.png" alt="The pysatSeasons logo: A calendar page featuring a snake orbiting a blue planet" title="pysatSeasons"</img>
</div>

# pysatSeasons
[![Documentation Status](https://readthedocs.org/projects/pysatseasons/badge/?version=latest)](https://pysatseasons.readthedocs.io/en/latest/?badge=latest)
[![PyPI Package latest release](https://img.shields.io/pypi/v/pysatSeasons.svg)](https://pypi.python.org/pypi/pysatSeasons)
[![Build Status](https://github.com/pysat/pysatSeasons/actions/workflows/main.yml/badge.svg)](https://github.com/pysat/pysatSeasons/actions/workflows/main.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/pysat/pysatSeasons/badge.svg?branch=main)](https://coveralls.io/github/pysat/pysatSeasons?branch=main)
[![DOI](https://zenodo.org/badge/209365329.svg)](https://zenodo.org/badge/latestdoi/209365329)



pysatSeasons allows users to run seasonal data analyses over N-dimensional 
data sets managed through the pysat module.

Main Features
-------------
- Seasonal binning and averaging routines for 1D and 2D distributions of nD data.
- Occurrence probability routines, daily or by orbit.
- Scatterplot of data_label(s) over two dimensions over a season.


# Installation

### Prerequisites

pysatSeasons uses common Python modules, as well as modules developed by
and for the Space Physics community.  

| Common modules | Community modules |
| -------------- | ----------------- |
| matplotlib     | pysat             |
| numpy          |                   |
| pandas         |                   |
| xarray         |                   |

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
