#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022, pysat development team
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Test pysatSeasons plotting code."""

import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt

import pysat
from pysatSeasons import plot


class TestBasics(object):
    """Tests to ensure the plot objects work as expected."""

    def setup_method(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 1, 1))

        return

    def teardown_method(self):
        """Run after every method to clean up previous testing."""
        del self.testInst
        plt.close()

        return

    def test_scatterplot_w_ioff(self):
        """Check if scatterplot generates figures."""

        plt.ioff()
        figs = plot.scatterplot(self.testInst, 'longitude', 'latitude',
                                'slt', [0.0, 24.0])

        axes = figs[0].get_axes()
        assert len(figs) == 1
        assert len(axes) == 3
        assert not mpl.is_interactive()

        return

    def test_scatterplot_w_ion(self):
        """Check if scatterplot generates and resets to interactive mode."""

        plt.ion()
        figs = plot.scatterplot(self.testInst, 'longitude', 'latitude',
                                'slt', [0.0, 24.0])

        axes = figs[0].get_axes()
        assert len(figs) == 1
        assert len(axes) == 3
        assert mpl.is_interactive()

        return

    def test_scatterplot_w_limits(self):
        """Check if scatterplot generates with appropriate limits."""

        figs = plot.scatterplot(self.testInst, 'longitude', 'latitude',
                                'slt', [0.0, 24.0],
                                xlim=[0, 360], ylim=[-80, 80])

        axes = figs[0].get_axes()
        assert len(figs) == 1
        assert len(axes) == 3
        assert axes[0].get_xlim() == (0, 360)
        assert axes[1].get_xlim() == (0, 360)
        assert axes[0].get_ylim() == (-80, 80)
        assert axes[1].get_ylim() == (-80, 80)

        return

    def test_multiple_scatterplots(self):
        """Check if multiple scatterplots generate."""
        figs = plot.scatterplot(self.testInst, 'longitude', 'latitude',
                                ['slt', 'mlt'], [0.0, 24.0])

        axes = figs[0].get_axes()
        axes2 = figs[1].get_axes()
        assert len(figs) == 2
        assert len(axes) == 3
        assert len(axes2) == 3

        return


class TestXarrayBasics(TestBasics):
    """Reapply basic tests with xarray data source."""

    def setup_method(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing_xarray',
                                         clean_level='clean')
        self.testInst.bounds = (dt.datetime(2008, 1, 1),
                                dt.datetime(2008, 1, 1))

        return


class TestConstellationBasics(TestBasics):
    """Reapply basic tests with Constellation data source."""

    def setup_method(self):
        """Run before every method to create a clean testing setup."""
        self.rawInst = pysat.Instrument('pysat', 'testing',
                                        clean_level='clean')
        self.rawInst.bounds = (dt.datetime(2008, 1, 1),
                               dt.datetime(2008, 1, 31))

        self.testInst = pysat.Constellation(instruments=[self.rawInst,
                                                         self.rawInst.copy()])

        return

    def teardown_method(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.rawInst
        plt.close()

        return


class TestXarrayConstellationBasics(TestXarrayBasics):
    """Reapply basic tests with Constellation xarray data source."""

    def setup_method(self):
        """Run before every method to create a clean testing setup."""
        self.rawInst = pysat.Instrument('pysat', 'testing_xarray',
                                        clean_level='clean')
        self.rawInst.bounds = (dt.datetime(2008, 1, 1),
                               dt.datetime(2008, 1, 31))

        self.testInst = pysat.Constellation(instruments=[self.rawInst,
                                                         self.rawInst.copy()])

        return

    def teardown_method(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.rawInst
        plt.close()

        return
