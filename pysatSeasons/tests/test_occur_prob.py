"""
Test pysatSeasons occur_prob object and code.
"""
import datetime as dt
import numpy as np
import pytest

import pysat
from pysatSeasons import occur_prob


class TestBasics():
    """Basic tests using pandas data source."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        orbit_info = {'index': 'longitude', 'kind': 'longitude'}
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean',
                                         orbit_info=orbit_info)
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 1, 31))

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst

        return

    def test_occur_prob_daily_2D_w_bins(self):
        """Run a basic probability routine daily 2D w/ bins"""
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
        """Input a gate that is not list-like"""

        ans = occur_prob.daily2D(self.testInst, [0, 360, 4], 'longitude',
                                 [-60, 60, 3], 'latitude', ['slt'], 12.)

        ans2 = occur_prob.daily2D(self.testInst, [0, 360, 4], 'longitude',
                                  [-60, 60, 3], 'latitude', ['slt'], [12.])

        assert np.array_equal(ans['slt']['prob'], ans2['slt']['prob'])

        return

    def test_occur_prob_daily_2D_w_mismatched_gate_and_data_label(self):
        """Catch a gate that does not match the data_label"""
        with pytest.raises(ValueError) as verr:
            occur_prob.daily2D(self.testInst, [0, 360, 4], 'longitude',
                               [-60, 60, 3], 'latitude', ['slt'], [12., 18.])

        estr = 'Must have a gate value for each data_label'
        assert str(verr).find(estr) >= 0

        return

    def test_occur_prob_by_orbit_2D_w_bins(self):
        """Run a basic probability routine by orbit 2D"""
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
        """Run a basic probability routine daily 3D"""
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
        """Catch a data_label that is not list-like"""
        ans = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                 [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                 'slt', [12.])

        ans2 = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                  [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                  ['slt'], [12.])

        assert np.array_equal(ans['slt']['prob'], ans2['slt']['prob'])

        return

    def test_occur_prob_daily_3D_w_bad_gate(self):
        """Catch a gate that is not list-like"""
        ans = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                 [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                 ['slt'], 12.)

        ans2 = occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                                  [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                                  ['slt'], [12.])

        assert np.array_equal(ans['slt']['prob'], ans2['slt']['prob'])

        return

    def test_occur_prob_daily_3D_w_mismatched_gate_and_data_label(self):
        """Catch a gate that does not match the data_label"""
        with pytest.raises(ValueError) as verr:
            occur_prob.daily3D(self.testInst, [0, 360, 4], 'longitude',
                               [-60, 60, 3], 'latitude', [0, 24, 2], 'slt',
                               ['slt'], [12., 18.])

        estr = 'Must have a gate value for each data_label'
        assert str(verr).find(estr) >= 0

        return

    def test_occur_prob_by_orbit_3D_w_bins(self):
        """Run a basic probability routine by orbit 3D"""
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

    def setup(self):
        """Run before every method to create a clean testing setup."""
        orbit_info = {'index': 'longitude', 'kind': 'longitude'}
        self.testInst = pysat.Instrument('pysat', 'testing_xarray',
                                         clean_level='clean',
                                         orbit_info=orbit_info)
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 1, 31))

        return


class TestConstellationBasics(TestBasics):
    """Basic tests using Constellations and pandas data source."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        orbit_info = {'index': 'longitude', 'kind': 'longitude'}
        self.rawInst = pysat.Instrument('pysat', 'testing',
                                        clean_level='clean',
                                        orbit_info=orbit_info)
        self.rawInst.bounds = (dt.datetime(2008, 1, 1),
                               dt.datetime(2008, 1, 31))

        self.testInst = pysat.Constellation(instruments=[self.rawInst,
                                                         self.rawInst.copy()])

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.rawInst

        return


class TestXarrayConstellationBasics(TestBasics):
    """Basic tests using Constellations and xarray data source."""

    def setup(self):
        """Run before every method to create a clean testing setup."""
        orbit_info = {'index': 'longitude', 'kind': 'longitude'}
        self.rawInst = pysat.Instrument('pysat', 'testing_xarray',
                                        clean_level='clean',
                                        orbit_info=orbit_info)
        self.rawInst.bounds = (dt.datetime(2008, 1, 1),
                               dt.datetime(2008, 1, 31))

        self.testInst = pysat.Constellation(instruments=[self.rawInst,
                                                         self.rawInst.copy()])

        return

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.rawInst

        return
