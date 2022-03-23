#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022, pysat development team
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Test the pysat averaging code."""

import datetime as dt
import numpy as np
import pandas as pds
import pytest
import warnings

import pysat
from pysat.utils import testing
from pysatSeasons import avg


class TestBasics():
    """Test basic functions using pandas 1D data sources."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean')
        self.bounds1 = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 3))
        self.bounds2 = (dt.datetime(2009, 1, 1), dt.datetime(2009, 1, 2))

        self.long_bins = [0., 360., 24]
        self.mlt_bins = [0., 24., 24]
        self.auto_bin = True

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.bounds1, self.bounds2, self.long_bins
        del self.mlt_bins

        return

    def test_basic_seasonal_median2D(self):
        """Test the basic seasonal 2D median."""
        self.testInst.bounds = self.bounds1
        vars = ['dummy1', 'dummy2', 'dummy3']
        results = avg.median2D(self.testInst, self.long_bins, 'longitude',
                               self.mlt_bins, 'mlt', vars, returnData=True,
                               auto_bin=self.auto_bin)

        # Iterate over all y rows. Value should be equal to integer value of
        # mlt. No variation in the median, all values should be the same.
        for i, y in enumerate(results['dummy1']['bin_y'][:-1]):
            assert np.all(results['dummy1']['median'][i, :] == y.astype(int))
            assert np.all(results['dummy1']['avg_abs_dev'][i, :] == 0)

        # Iterate over x rows. Value should be the longitude / 15.
        for i, x in enumerate(results['dummy1']['bin_x'][:-1]):
            assert np.all(results['dummy2']['median'][:, i] == x / 15.0)
            assert np.all(results['dummy2']['avg_abs_dev'][:, i] == 0)

        # Iterate over x rows. Value should be the longitude / 15 * 1000.
        for i, x in enumerate(results['dummy1']['bin_x'][:-1]):
            assert np.all(results['dummy3']['median'][:, i] == x / 15.0 * 1000.0
                          + results['dummy1']['bin_y'][:-1])
            assert np.all(results['dummy3']['avg_abs_dev'][:, i] == 0)

        # Holds here because there are 32 days, no data is discarded,
        # and each day holds same amount of data.
        assert np.all(self.testInst.data['dummy1'].size * 3
                      == sum([sum(i) for i in results['dummy1']['count']]))

        # Ensure all outputs are numpy arrays
        for var in vars:
            assert isinstance(results[var]['median'], type(np.array([])))

        # Ensure binned data returned
        for var in vars:
            assert 'data' in results[var].keys()

        return

    def test_basic_seasonal_median1D(self):
        """Test the basic seasonal 1D median."""
        self.testInst.bounds = self.bounds1
        vars = ['dummy1', 'dummy2', 'dummy3']
        results = avg.median1D(self.testInst, self.long_bins, 'longitude',
                               vars, returnData=True, auto_bin=self.auto_bin)

        # Iterate over x rows. Value should be the longitude / 15.
        for i, x in enumerate(results['dummy1']['bin_x'][:-1]):
            assert np.all(results['dummy2']['median'][i] == x / 15.0)
            assert np.all(results['dummy2']['avg_abs_dev'][i] == 0)

        # Iterate over x rows. Value should be the longitude / 15 * 1000.
        # except for the variation in value with 'mlt'.
        for i, x in enumerate(results['dummy1']['bin_x'][:-1]):
            assert np.all(results['dummy3']['median'][i] // 100 * 100 == x
                          / 15.0 * 1000.0)
            assert np.all(results['dummy3']['avg_abs_dev'][i] > 0)

        # Holds here because there are 32 days, no data is discarded,
        # and each day holds same amount of data.
        assert np.all(self.testInst.data['dummy1'].size * 3
                      == sum(results['dummy1']['count']))

        # Ensure all outputs are numpy arrays
        for var in vars:
            assert isinstance(results[var]['median'], type(np.array([])))

        # Ensure binned data returned
        for var in vars:
            assert 'data' in results[var].keys()

        return


class TestXarrayBasics(TestBasics):
    """Reapply basic tests to 1D xarray data sources."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing_xarray',
                                         clean_level='clean')
        self.bounds1 = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 3))
        self.bounds2 = (dt.datetime(2009, 1, 1), dt.datetime(2009, 1, 2))

        self.long_bins = [0., 360., 24]
        self.mlt_bins = [0., 24., 24]
        self.auto_bin = True

        return


class TestBasicsMeanBy():
    """Test basic functions using pandas 1D data sources."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean')
        self.bounds1 = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 3))
        self.bounds2 = (dt.datetime(2009, 1, 1), dt.datetime(2009, 1, 2))

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.bounds1, self.bounds2

        return

    def test_basic_daily_mean(self):
        """Test basic daily mean."""
        self.testInst.bounds = self.bounds1
        ans = avg.mean_by_day(self.testInst, 'dummy4')
        assert np.all(ans == 86399 / 2.0)

        return

    def test_basic_orbit_mean(self):
        """Test basic orbital mean."""
        orbit_info = {'kind': 'local time', 'index': 'mlt'}
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean',
                                         orbit_info=orbit_info)
        self.testInst.bounds = self.bounds2
        ans = avg.mean_by_orbit(self.testInst, 'mlt')

        # Note last orbit is incomplete thus not expected to satisfy relation
        ans = ans[:-1]

        assert np.allclose(ans.values.tolist(), np.full(len(ans), 12.), 1.0E-2)

        return

    def test_basic_file_mean(self):
        """Test basic file mean."""
        index = pds.date_range(*self.bounds1)
        names = [''.join((date.strftime('%Y-%m-%d'), '.nofile'))
                 for date in index]
        self.testInst.bounds = (names[0], names[-1])
        ans = avg.mean_by_file(self.testInst, 'dummy4')
        assert np.all(ans == 86399 / 2.0)

        return


class TestXarrayBasicsMeanBy(TestBasicsMeanBy):
    """Reapply basic tests to 1D xarray data sources."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing_xarray',
                                         clean_level='clean')
        self.bounds1 = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 3))
        self.bounds2 = (dt.datetime(2009, 1, 1), dt.datetime(2009, 1, 2))

        return


class TestFrameProfileAverages():
    """Test bin averaging dataframes from pandas data sources."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing2D',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 1, 3))
        self.dname = 'alt_profiles'
        self.test_vals = np.arange(50) * 1.2
        self.test_fracs = np.arange(50) / 50.0

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.dname, self.test_vals, self.test_fracs

        return

    def test_basic_seasonal_2Dmedian(self):
        """Test the basic seasonal 2D median."""

        results = avg.median2D(self.testInst, [0., 360., 24], 'longitude',
                               [0., 24., 24], 'mlt', [self.dname])

        # Test medians.
        for i, row in enumerate(results[self.dname]['median']):
            for j, item in enumerate(row):
                assert np.all(item['density'] == self.test_vals)
                assert np.all(item['fraction'] == self.test_fracs)

        # No variation in the median, all values should be the same
        for i, row in enumerate(results[self.dname]['avg_abs_dev']):
            for j, item in enumerate(row):
                assert np.all(item['density'] == 0)
                assert np.all(item['fraction'] == 0)

        return

    def test_basic_seasonal_1Dmedian(self):
        """Test the basic seasonal 1D median."""

        results = avg.median1D(self.testInst, [0., 24, 24], 'mlt',
                               [self.dname])

        # Test medians.
        for i, row in enumerate(results[self.dname]['median']):
            assert np.all(row['density'] == self.test_vals)
            assert np.all(row['fraction'] == self.test_fracs)

        # No variation in the median, all values should be the same.
        for i, row in enumerate(results[self.dname]['avg_abs_dev']):
            assert np.all(row['density'] == 0)
            assert np.all(row['fraction'] == 0)

        return


class TestSeriesProfileAverages():
    """Test bin averaging series profile data from pandas data sources."""

    def setup(self):
        """Runs before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing2D',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 2, 1))
        self.dname = 'series_profiles'
        self.test_vals = np.arange(50) * 1.2

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.dname

        return

    def test_basic_seasonal_median2D(self):
        """Test basic seasonal 2D median."""
        results = avg.median2D(self.testInst, [0., 360., 24], 'longitude',
                               [0., 24., 24], 'mlt', [self.dname])

        # Test medians.
        for i, row in enumerate(results[self.dname]['median']):
            for j, item in enumerate(row):
                assert np.all(item[self.dname] == self.test_vals)

        # No variation in the median, all values should be the same.
        for i, row in enumerate(results[self.dname]['avg_abs_dev']):
            for j, item in enumerate(row):
                assert np.all(item[self.dname] == 0)

        return

    def test_basic_seasonal_median1D(self):
        """Test basic seasonal 1D median."""
        results = avg.median1D(self.testInst, [0., 24., 24], 'mlt',
                               [self.dname])

        # Test medians.
        for i, row in enumerate(results[self.dname]['median']):
            assert np.all(row[self.dname] == self.test_vals)

        # No variation in the median, all values should be the same.
        for i, row in enumerate(results[self.dname]['avg_abs_dev']):
            assert np.all(row[self.dname] == 0)

        return


class TestXarrayProfileAverages():
    """Test bin averaging profile data from xarray data sources."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing2D_xarray',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 2, 1))
        self.dname = 'profiles'
        self.test_val_length = 15

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.dname, self.test_val_length

        return

    def test_basic_seasonal_median2D(self):
        """Test basic seasonal 2D median for xarray data sources."""
        results = avg.median2D(self.testInst, [0., 360., 24], 'longitude',
                               [0., 24., 24], 'mlt', [self.dname])

        # Values in xarray instrument depend upon longitude and mlt location.
        for i, row in enumerate(results[self.dname]['median']):
            mlt_val = i
            for j, item in enumerate(row):
                long_val = j * 1000.
                test_vals = mlt_val + long_val
                assert np.all(item[self.dname].values == test_vals)

        # No variation in the median, all values should be the same.
        for i, row in enumerate(results[self.dname]['avg_abs_dev']):
            for j, item in enumerate(row):
                assert np.all(item[self.dname].values == 0)

        return

    def test_basic_seasonal_median1D(self):
        """Test basic seasonal 1D median for xarray data sources."""
        results = avg.median1D(self.testInst, [0., 24., 24], 'mlt',
                               [self.dname])

        for i, row in enumerate(results[self.dname]['median']):
            # Define truth values. There is a variation in value based on
            # longitude, at thousands level. MLT only shows at ones/tens level.
            test_vals = [i] * self.test_val_length
            vals = []
            for val in row[self.dname].values:
                if not isinstance(val, np.float64):
                    # Provide support for testing higher order data sources.
                    val = val[0]
                vals.append(int(str(int(val))[-2:]))
            assert np.all(vals == test_vals)

        # There is a variation in binned value based upon longitude.
        for i, row in enumerate(results[self.dname]['avg_abs_dev']):
            assert np.all(row[self.dname] >= 0)

        return


class TestXarrayVariableProfileAverages(TestXarrayProfileAverages):
    """Test bin averaging variable profile data from xarray data sources."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing2D_xarray',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 2, 1))
        self.dname = 'variable_profiles'
        self.test_val_length = 15

        return


class TestXarrayImageAverages(TestXarrayProfileAverages):
    """Test bin averaging image data from xarray data sources."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing2D_xarray',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 2, 1))
        self.dname = 'images'
        self.test_val_length = 17

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.dname, self.test_val_length

        return


class TestConstellation():
    def setup(self):
        """Run before every method to create a clean testing setup."""
        insts = []
        for i in range(5):
            insts.append(pysat.Instrument('pysat', 'testing',
                                          clean_level='clean'))
        self.testC = pysat.Constellation(instruments=insts)
        self.testI = pysat.Instrument('pysat', 'testing', clean_level='clean')
        self.bounds = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 3))

        # Apply bounds to all Instruments in Constellation, and solo Instrument.
        self.testC.bounds = self.bounds
        self.testI.bounds = self.bounds

        # Define variables for 1D testing
        self.one_d_vars = ['dummy1', 'dummy2', 'dummy3']
        self.unequal_one_d_vars = []

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testC, self.testI, self.bounds

        return

    def test_constellation_median2D(self):
        """Test constellation implementation of 2D median."""

        vars = ['dummy1', 'dummy2', 'dummy3']

        resultsC = avg.median2D(self.testC, [0., 360., 24], 'longitude',
                                [0., 24., 24], 'mlt', vars)
        resultsI = avg.median2D(self.testI, [0., 360., 24], 'longitude',
                                [0., 24., 24], 'mlt', vars)

        output_labels = ['median', 'avg_abs_dev']
        for var in vars:
            for output in output_labels:
                assert np.array_equal(resultsC[var][output],
                                      resultsI[var][output])

        return

    def test_constellation_median1D(self):
        """Test constellation implementation of 1D median."""

        vars = []
        for var in [self.one_d_vars, self.unequal_one_d_vars]:
            vars.extend(var)

        resultsC = avg.median1D(self.testC, [0., 24, 24], 'mlt',
                                vars)
        resultsI = avg.median1D(self.testI, [0., 24, 24], 'mlt',
                                vars)

        output_labels = ['median', 'avg_abs_dev']
        for var in self.one_d_vars:
            for output in output_labels:
                assert np.array_equal(resultsC[var][output],
                                      resultsI[var][output])

        # Check parameters that are not the same.
        output_labels = ['median']
        for var in self.unequal_one_d_vars:
            for output in output_labels:
                assert not np.array_equal(resultsC[var][output],
                                          resultsI[var][output])

        return


class TestHeterogenousConstellation(TestConstellation):
    """Test with a Constellation of Instruments with different parameters."""
    def setup(self):
        """Run before every method to create a clean testing setup."""
        insts = []
        for i in range(2):
            r_date = dt.datetime(2009, 1, i + 1)
            insts.append(pysat.Instrument('pysat', 'testing',
                                          clean_level='clean',
                                          root_date=r_date))
        self.testC = pysat.Constellation(instruments=insts)
        self.testI = pysat.Instrument('pysat', 'testing', clean_level='clean')
        self.bounds = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 3))

        # Apply bounds to all Instruments in Constellation, and solo Instrument.
        self.testC.bounds = self.bounds
        self.testI.bounds = self.bounds

        # Define variables for 1D testing. A more limited set that only
        # depends upon 'mlt'. Other variables also include longitude, which
        # can differ between instruments when only binning by 'mlt'.
        self.one_d_vars = ['dummy1']
        self.unequal_one_d_vars = ['dummy2', 'dummy3']

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testC, self.bounds, self.one_d_vars, self.unequal_one_d_vars
        del self.testI

        return


class Test2DConstellation(TestSeriesProfileAverages):

    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.insts = []
        self.testInst = pysat.Instrument('pysat', 'testing2D',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 1, 3))
        self.insts.append(self.testInst)
        self.insts.append(self.testInst)

        self.dname = 'series_profiles'
        self.test_vals = np.arange(50) * 1.2

        self.testC = pysat.Constellation(instruments=self.insts)

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testC, self.insts, self.testInst, self.dname, self.test_vals

        return


class TestSeasonalAverageUnevenBins(TestBasics):
    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 1, 3))

        self.bounds1 = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 3))
        self.bounds2 = (dt.datetime(2009, 1, 1), dt.datetime(2009, 1, 2))

        self.long_bins = np.linspace(0., 360., 25)
        self.mlt_bins = np.linspace(0., 24., 25)

        self.auto_bin = False

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.bounds1, self.bounds2, self.long_bins
        del self.mlt_bins

        return

    def test_nonmonotonic_bins(self):
        """Test 2D median failure when provided with a non-monotonic bins."""
        with pytest.raises(ValueError) as verr:
            avg.median2D(self.testInst, np.array([0., 300., 100.]), 'longitude',
                         np.array([0., 24., 13.]), 'mlt',
                         ['dummy1', 'dummy2', 'dummy3'], auto_bin=False)

        estr = 'bins must be monotonically increasing or decreasing'
        assert str(verr).find(estr) >= 0

        return

    def test_bin_data_depth(self):
        """Test failure when an array-like of length 1 is given to median2D."""
        with pytest.raises(TypeError) as verr:
            avg.median2D(self.testInst, 1, 'longitude', 24, 'mlt',
                         ['dummy1', 'dummy2', 'dummy3'], auto_bin=False)

        estr = 'len() of unsized object'
        assert str(verr).find(estr) >= 0

        return

    def test_bin_data_type(self):
        """Test failure when a non array-like is given to median2D."""
        with pytest.raises(TypeError) as verr:
            avg.median2D(self.testInst, ['1', 'a', '23', '10'], 'longitude',
                         ['0', 'd', '24', 'c'], 'mlt',
                         ['dummy1', 'dummy2', 'dummy3'], auto_bin=False)

        estr = "Cannot cast array data from"
        assert str(verr).find(estr) >= 0

        return

    def test_median2D_bad_input(self):
        """Test failure of median2D with non Constellation or Instrument input.
        """
        with pytest.raises(ValueError) as verr:
            avg.median2D([], [0., 360., 24], 'longitude', [0., 24., 24], 'mlt',
                         ['longitude'])

        assert str(verr).find('Parameter must be an Instrument') > 0

        return


class TestInstMed1D():
    def setup(self):
        """Run before every method to create a clean testing setup"""
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean',
                                         update_files=True)
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 1, 31))
        self.test_bins = [0, 24, 24]
        self.test_label = 'slt'
        self.test_data = ['dummy1', 'dummy2']
        self.out_keys = ['count', 'avg_abs_dev', 'median', 'bin_x']
        self.out_data = {'dummy1':
                         {'count': [111780., 111320., 111780., 111320.,
                                    111780., 111320., 111780., 111320.,
                                    111780., 111320., 111780., 111320.,
                                    111780., 111320., 111918., 111562.,
                                    112023., 111562., 112023., 111412.,
                                    111780., 111320., 111780., 111320.],
                          'avg_abs_dev': np.zeros(shape=24),
                          'median': np.linspace(0.0, 23.0, 24)},
                         'dummy2':
                         {'count': [111780., 111320., 111780., 111320.,
                                    111780., 111320., 111780., 111320.,
                                    111780., 111320., 111780., 111320.,
                                    111780., 111320., 111918., 111562.,
                                    112023., 111562., 112023., 111412.,
                                    111780., 111320., 111780., 111320.],
                          'avg_abs_dev': np.zeros(shape=24) + 6.0,
                          'median': [11., 12., 11., 11., 12., 11., 12., 11.,
                                     12., 12., 11., 12., 11., 12., 11., 11.,
                                     12., 11., 12., 11., 11., 11., 11., 12.]}}
        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.test_bins, self.test_label, self.test_data
        del self.out_keys, self.out_data

        return

    def test_median1D_default(self):
        """Test success of median1D with default options."""

        med_dict = avg.median1D(self.testInst, self.test_bins, self.test_label,
                                self.test_data)

        # Test output type
        assert isinstance(med_dict, dict)
        assert len(med_dict.keys()) == len(self.test_data)

        # Test output keys
        for kk in med_dict.keys():
            assert kk in self.test_data
            assert np.all([jj in self.out_keys
                           for jj in med_dict[kk].keys()])

            # Test output values
            for jj in self.out_keys[:-1]:
                assert len(med_dict[kk][jj]) == self.test_bins[-1]
                assert np.all(med_dict[kk][jj] == self.out_data[kk][jj])

            jj = self.out_keys[-1]
            assert len(med_dict[kk][jj]) == self.test_bins[-1] + 1
            assert np.all(med_dict[kk][jj] == np.linspace(self.test_bins[0],
                                                          self.test_bins[1],
                                                          self.test_bins[2]
                                                          + 1))
        del med_dict, kk, jj

        return

    def test_median1D_bad_data(self):
        """Test failure of median1D with string data instead of list."""
        with pytest.raises(KeyError) as verr:
            avg.median1D(self.testInst, self.test_bins, self.test_label,
                         self.test_data[0])

        estr = self.test_data[0][0]
        assert str(verr).find(estr) >= 0

        return

    def test_median1D_bad_input(self):
        """Test failure of median1D with non Constellation or Instrument input.
        """
        with pytest.raises(ValueError) as verr:
            avg.median1D([], self.test_bins, self.test_label,
                         self.test_data[0])

        assert str(verr).find('Parameter must be an Instrument') > 0

        return

    def test_median1D_bad_label(self):
        """Test failure of median1D with unknown label."""
        with pytest.raises(KeyError) as verr:
            avg.median1D(self.testInst, self.test_bins, "bad_label",
                         self.test_data)

        estr = "bad_label"
        assert str(verr).find(estr) >= 0

        return

    def test_nonmonotonic_bins(self):
        """Test median1D failure when provided with a non-monotonic bins."""
        with pytest.raises(ValueError) as verr:
            avg.median1D(self.testInst, [0, 13, 5], self.test_label,
                         self.test_data, auto_bin=False)

        estr = 'bins must be monotonically increasing or decreasing'
        assert str(verr).find(estr) >= 0

        return

    def test_bin_data_depth(self):
        """Test failure when array-like of length 1 is given to median1D."""
        with pytest.raises(TypeError) as verr:
            avg.median1D(self.testInst, 24, self.test_label, self.test_data,
                         auto_bin=False)

        estr = 'len() of unsized object'
        assert str(verr).find(estr) >= 0

        return

    def test_bin_data_type(self):
        """Test failure when median 1D is given non array-like bins."""
        with pytest.raises(TypeError) as verr:
            avg.median2D(self.testInst, ['0', 'd', '24', 'c'], self.test_label,
                         self.test_data, auto_bin=False)

        estr = 'median2D() missing 2 required positional arguments'
        assert str(verr).find(estr) >= 0

        return


class TestDeprecation(object):
    """Unit test for deprecation warnings."""

    def setup(self):
        """Set up the unit test environment for each method."""

        warnings.simplefilter("always", DeprecationWarning)

        orbit_info = {'index': 'slt', 'kind': 'lt'}
        self.tinst = pysat.Instrument('pysat', 'testing', orbit_info=orbit_info)
        self.tinst.bounds = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 2))

        self.warn_msgs = []
        self.war = ""
        return

    def teardown(self):
        """Clean up the unit test environment after each method."""
        # self.in_kwargs, self.ref_time,
        del self.warn_msgs, self.war
        return

    def eval_warnings(self):
        """Evaluate the number and message of the raised warnings."""

        # Ensure the minimum number of warnings were raised.
        assert len(self.war) >= len(self.warn_msgs)

        # Test the warning messages, ensuring each attribute is present.
        testing.eval_warnings(self.war, self.warn_msgs)
        return

    @pytest.mark.parametrize("func,dim_set", [(avg.median1D, 1),
                                              (avg.median2D, 2),
                                              ])
    @pytest.mark.parametrize("return_flag", [True, False])
    def test_returnData_kwarg_ndimensional(self, func, dim_set, return_flag):
        """Test deprecation of kwarg `returnData`.

        Parameters
        ----------
        func : function
            Function under test.
        dim_set : int
            Number of dimensions for function call.
        return_flag : bool
            Setting to be applied to returnBins.

        """
        # Set up function calls
        bin = [0, 24, 10]
        if dim_set == 1:
            bin_axes = ['data']
            args = (self.tinst, bin, 'longitude', ['slt'])
        elif dim_set == 2:
            bin_axes = ['data']
            args = (self.tinst, bin, 'longitude', bin, 'latitude', ['slt'])

        # Catch the warnings
        with warnings.catch_warnings(record=True) as self.war:
            data = func(*args, returnData=return_flag)

        # Ensure bins are returned or not, as directed
        for var in data.keys():
            for bin_ax in bin_axes:
                flag = bin_ax in data[var].keys()
                if not return_flag:
                    flag = not flag

                assert flag

        estr = '"returnData" has been deprecated in favor of '
        self.warn_msgs = np.array([estr])

        # Evaluate the warning output
        self.eval_warnings()
        return
