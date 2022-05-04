# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [0.2.0] - 2022-05-06
- New Features
  - Added support for xarray data in the seasonal averaging functions in `pysatSeasons.avg`
  - Added support for xarray data in the occurrence probability functions in `pysatSeasons.occur_prob`
  - Added support for Constellations in `pysatSeasons.occur_prob`
  - Added support for Constellations in `pysatSeasons.plot`
  - Renamed `computational_form` to `to_xarray_dataset` and refocused.
- Deprecations
  - Deprecated `returnBins` keyword in favor of `return_bins` in `pysatSeasons.occur_prob`.
  - Deprecated `returnData` keyword in favor of `return_data` in `pysatSeasons.avg`.
- Documentation
  - Improved docstrings throughout.
  - Updated documentation examples.
  - Documentation now available on readthedocs.org.
- Bug Fix
- Maintenance
  - Removed deprecated `pandas.Panel` from functions.
  - Removed old `__future__` imports.
  - Removed use of `collections.deque` in `pysatSeasons.avg`.
  - Migrated to GitHub Workflows for CI testing.
  - Migrated from noses to pytest.
  - Adopted setup.cfg

## [0.1.3] - 2021-06-18
- Updates style to match pysat 3.0.0 release candidate
- Improves discussion of rationale for version caps on readme page
- Migrates CI tests to github actions

## [0.1.2] - 2020-07-29
- Updates demo codes to import objects from datetime and pandas for pysat 3.0.0 compatibility
- Fixed a bug where test routines used float where numpy 1.18 expects an int
- Import objects from datetime and pandas for pysat 3.0.0 compatibility
- Use conda to manage Travis CI
- Rename default branch as `main`
- Update to pysat documentation standards
- Add flake8 testing for code

## [0.1.1] - 2019-10-09
- Add demo code
- Added DOI badge to documentation page

## [0.1.0] - 2019-10-07
- Initial release
