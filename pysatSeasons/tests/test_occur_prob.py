#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022, pysat development team
# Full license can be found in License.md
# DOI:10.5281/zenodo.3475493
#
# Review Status for Classified or Controlled Information by NRL
# -------------------------------------------------------------
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Test pysatSeasons occur_prob object and code."""

import datetime as dt
import numpy as np
from packaging import version as pack_version
import pytest
import warnings

import pysat
from pysat.utils import testing
from pysatSeasons import occur_prob


class TestBasics(object):
    """Basic tests using pandas data source."""

    def setup_method(self):
        """Run before every method to create a clean testing setup."""
        orbit_info = {'index': 'longitude', 'kind': 'longitude'}
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean',
                                         orbit_info=orbit_info)

        # Assign short bounds.
        test_date = pysat.instruments.pysat_testing._test_dates['']['']
        self.testInst.bounds = (test_date, test_date + dt.timedelta(days=1))

        return

    def teardown_method(self):
        """Run after every method to clean up previous testing."""
        del self.testInst

        return

    def test_occur_prob_daily_2D_w_bins(self):
        """Run a basic probability routine daily 2D w/ bins."""
        ans = occur_prob.daily2D(self.testInst, [0, 24, 2], 'slt',
                                 [-60, 60, 3], 'latitude', ['slt'], [12.],
                                 returnBins=True)
        assert abs(ans['slt']['prob'][:, 0] - 0.0).max() < 1.0e-6
        assert abs(ans['slt']['prob'][:, 1] - 1.0).max() < 1.0e-6
        assert (ans['slt']['prob']).shape == (3, 2)
        assert abs(ans['slt']['bin_x'] - [0, 12, 24]).max() < 1.0e-6
        assert abs(ans['slt']['bin_y'] - [-60, -20, 20, 60]).max() < 1.0e-6

        return

    def test_occur_prob_daily_2D_w_bad_data_label(self):
        """Input a data_label that is not list-like."""

        ans = occur_prob.daily2D(self.testInst, [0, 360, 4], 'longitude',
                                 [-60, 60, 3], 'latitude', 'slt', [12.])

        ans2 = occur_prob.daily2D(self.testInst, [0, 360, 4], 'longitude',
                                  [-60, 60, 3], 'latitude', ['slt'], [12.])

        assert np.array_equal(ans['slt']['prob'], ans2['slt']['prob'])

        return

    def test_occur_prob_daily_2D_w_bad_gate(self):
        """Input a gate that is not list-like."""

        ans = occur_prob.daily2D(self.testInst, [0, 360, 4], 'longitude',
                                 [-60, 60, 3], 'latitude', ['slt'], 12.)

        ans2 = occur_prob.daily2D(self.testInst, [0, 360, 4], 'longitude',
                                  [-60, 60, 3], 'latitude', ['slt'], [12.])

        assert np.array_equal(ans['slt']['prob'], ans2['slt']['prob'])

        return

    def test_occur_prob_daily_2D_w_mismatched_gate_and_data_label(self):
        """Catch a gate that does not match the data_label."""
        with pytest.raises(ValueError) as verr:
            occur_prob.daily2D(self.testInst, [0, 360, 4], 'longitude',
                               [-60, 60, 3], 'latitude', ['slt'], [12., 18.])

        estr = 'Must have a gate value for each data_label'
        assert str(verr).find(estr) >= 0

        return

    def test_occur_prob_by_orbit_2D_w_bins(self):
        """Run a basic probability routine by orbit 2D."""
        ans = occur_prob.by_orbit2D(self.testInst, [0, 24, 2], 'slt',
                                    [-60, 60, 3], 'latitude', ['slt'], [12.],
                                    returnBins=True)
        assert abs(ans['slt']['prob'][:, 0] - 0.0).max() < 1.0e-6
        assert abs(ans['slt']['prob'][:, 1] - 1.0).max() < 1.0e-6
        assert (ans['slt']['prob']).shape == (3, 2)
        assert abs(ans['slt']['bin_x'] - [0, 12, 24]).max() < 1.0e-6
        assert abs(ans['slt']['bin_y'] - [-60, -20, 20, 60]).max() < 1.0e-6

        return

    def test_occur_prob_daily_3D_w_bins(self):
        """Run a basic probability routine daily 3D."""
        ans = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                 [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                 ['slt'], [12.], returnBins=True)
        assert abs(ans['slt']['prob'][0, :, :] - 0.0).max() < 1.0e-6
        assert abs(ans['slt']['prob'][-1, :, :] - 1.0).max() < 1.0e-6
        assert (ans['slt']['prob']).shape == (2, 3, 4)
        assert abs(ans['slt']['bin_x'] - [0, 90, 180, 270, 360]).max() < 1.0e-6
        assert abs(ans['slt']['bin_y'] - [-60, -20, 20, 60]).max() < 1.0e-6
        assert abs(ans['slt']['bin_z'] - [0, 12, 24]).max() < 1.0e-6

        return

    def test_occur_prob_daily_3D_w_bad_data_label(self):
        """Catch a data_label that is not list-like."""
        ans = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                 [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                 'slt', [12.])

        ans2 = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                  [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                  ['slt'], [12.])

        assert np.array_equal(ans['slt']['prob'], ans2['slt']['prob'])

        return

    def test_occur_prob_daily_3D_w_bad_gate(self):
        """Catch a gate that is not list-like."""
        ans = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                 [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                 ['slt'], 12.)

        ans2 = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                  [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                  ['slt'], [12.])

        assert np.array_equal(ans['slt']['prob'], ans2['slt']['prob'])

        return

    def test_occur_prob_daily_3D_w_mismatched_gate_and_data_label(self):
        """Catch a gate that does not match the data_label."""
        with pytest.raises(ValueError) as verr:
            occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                               [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                               ['slt'], [12., 18.])

        estr = 'Must have a gate value for each data_label'
        assert str(verr).find(estr) >= 0

        return

    def test_occur_prob_by_orbit_3D_w_bins(self):
        """Run a basic probability routine by orbit 3D."""
        ans = occur_prob.by_orbit3D(self.testInst, [0, 360, 4], 'longitude',
                                    [-60, 60, 3], 'latitude',
                                    [0, 24, 2], 'slt', ['slt'], [12.],
                                    returnBins=True)
        assert abs(ans['slt']['prob'][0, :, :] - 0.0).max() < 1.0e-6
        assert abs(ans['slt']['prob'][-1, :, :] - 1.0).max() < 1.0e-6
        assert (ans['slt']['prob']).shape == (2, 3, 4)
        assert abs(ans['slt']['bin_x'] - [0, 90, 180, 270, 360]).max() < 1.0e-6
        assert abs(ans['slt']['bin_y'] - [-60, -20, 20, 60]).max() < 1.0e-6
        assert abs(ans['slt']['bin_z'] - [0, 12, 24]).max() < 1.0e-6

        return


class TestXarrayBasics(TestBasics):
    """Reapply basic tests with xarray data source."""

    def setup_method(self):
        """Run before every method to create a clean testing setup."""
        orbit_info = {'index': 'longitude', 'kind': 'longitude'}
        self.testInst = pysat.Instrument('pysat', 'ndtesting',
                                         clean_level='clean',
                                         orbit_info=orbit_info)

        # Assign short bounds.
        test_date = pysat.instruments.pysat_ndtesting._test_dates['']['']
        self.testInst.bounds = (test_date, test_date + dt.timedelta(days=1))

        return


class TestConstellationBasics(TestBasics):
    """Basic tests using Constellations and pandas data source."""

    def setup_method(self):
        """Run before every method to create a clean testing setup."""
        orbit_info = {'index': 'longitude', 'kind': 'longitude'}
        self.rawInst = pysat.Instrument('pysat', 'testing',
                                        clean_level='clean',
                                        orbit_info=orbit_info)

        # Assign short bounds.
        test_date = pysat.instruments.pysat_testing._test_dates['']['']
        self.rawInst.bounds = (test_date, test_date + dt.timedelta(days=1))

        self.testInst = pysat.Constellation(instruments=[self.rawInst,
                                                         self.rawInst.copy()])

        return

    def teardown_method(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.rawInst

        return


class TestXarrayConstellationBasics(TestBasics):
    """Basic tests using Constellations and xarray data source."""

    def setup_method(self):
        """Run before every method to create a clean testing setup."""
        orbit_info = {'index': 'longitude', 'kind': 'longitude'}
        self.rawInst = pysat.Instrument('pysat', 'ndtesting',
                                        clean_level='clean',
                                        orbit_info=orbit_info)

        # Assign short bounds.
        test_date = pysat.instruments.pysat_ndtesting._test_dates['']['']
        self.rawInst.bounds = (test_date, test_date + dt.timedelta(days=1))

        self.testInst = pysat.Constellation(instruments=[self.rawInst,
                                                         self.rawInst.copy()])

        return

    def teardown_method(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.rawInst

        return


@pytest.mark.skipif(pack_version.Version(pysat.__version__)
                    < pack_version.Version('3.0.2'),
                    reason=''.join(('Requires testing functions in pysat ',
                                    ' v3.0.2 or later.')))
class TestDeprecation(object):
    """Unit test for deprecation warnings."""

    def setup_method(self):
        """Set up the unit test environment for each method."""

        warnings.simplefilter("always", DeprecationWarning)

        orbit_info = {'index': 'slt', 'kind': 'lt'}
        self.tinst = pysat.Instrument('pysat', 'testing', orbit_info=orbit_info)
        self.tinst.bounds = (dt.datetime(2008, 1, 1), dt.datetime(2008, 1, 2))

        self.warn_msgs = []
        self.war = ""
        return

    def teardown_method(self):
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

    @pytest.mark.parametrize("func,dim_set", [(occur_prob.daily2D, 2),
                                              (occur_prob.by_orbit2D, 2),
                                              (occur_prob.daily3D, 3),
                                              (occur_prob.by_orbit3D, 3)
                                              ])
    @pytest.mark.parametrize("return_flag", [True, False])
    def test_returnBins_kwarg_ndimensional(self, func, dim_set, return_flag):
        """Test deprecation of kwarg `returnBins`.

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
        if dim_set == 2:
            bin_axes = ['bin_x', 'bin_y']
            args = (self.tinst, bin, 'longitude', bin, 'latitude', 'slt', 22.)
        elif dim_set == 3:
            bin_axes = ['bin_x', 'bin_y', 'bin_z']
            args = (self.tinst, bin, 'longitude', bin, 'latitude', bin,
                    'altitude', 'slt', 22.)

        # Catch the warnings
        with warnings.catch_warnings(record=True) as self.war:
            data = func(*args, returnBins=return_flag)

        # Ensure bins are returned or not, as directed
        for var in data.keys():
            for bin_ax in bin_axes:
                flag = bin_ax in data[var].keys()
                if not return_flag:
                    flag = not flag

                assert flag

        estr = '"returnBins" has been deprecated in favor of '
        self.warn_msgs = np.array([estr])

        # Evaluate the warning output
        self.eval_warnings()
        return
