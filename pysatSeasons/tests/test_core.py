"""
tests the pysat _core code
"""
import xarray as xr

import pysat
import pysatSeasons as ps


class TestCore(object):
    def setup(self):
        """Runs before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing',
                                         clean_level='clean')
        self.bounds1 = self.testInst.inst_module._test_dates['']['']

    def teardown(self):
        """Runs after every method to clean up previous testing."""
        del self.testInst, self.bounds1

    def test_comp_form_simple_data(self):
        """Test computational_form with inst.data"""
        self.testInst.load(date=self.bounds1)

        self.out = ps.computational_form(self.testInst.data)
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' not in self.out.dims
        return

    def test_comp_form_instrument_variable(self):
        """Test computational_form with inst[var]"""

        self.testInst.load(date=self.bounds1)

        self.out = ps.computational_form(self.testInst['mlt'])
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' not in self.out.dims
        return

    def test_comp_form_list_vars(self):
        """Test computational_form with [inst[var], inst[var2], ...]"""
        self.testInst.load(date=self.bounds1)
        self.out = ps.computational_form([self.testInst['mlt'],
                                          self.testInst['longitude']])
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' in self.out.dims
        return

    def test_comp_form_list_data(self):
        """Test computational_form with [inst.data, inst.data, ...]"""
        self.testInst.load(date=self.bounds1)
        self.out = ps.computational_form([self.testInst.data,
                                          self.testInst.data])
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' in self.out.dims
        return


class TestCoreXarray(TestCore):
    def setup(self):
        """Runs before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing_xarray',
                                         clean_level='clean')
        self.bounds1 = self.testInst.inst_module._test_dates['']['']

    def teardown(self):
        """Runs after every method to clean up previous testing."""
        del self.testInst, self.bounds1
