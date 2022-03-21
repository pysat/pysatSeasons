"""Test the pysat _core code."""

import xarray as xr

import pysat
from pysat import instruments as pinsts
import pysatSeasons as pyseas


class TestCore(object):
    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument(inst_module=pinsts.pysat_testing,
                                         clean_level='clean')
        self.bounds1 = self.testInst.inst_module._test_dates['']['']

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.bounds1

    def test_comp_form_simple_data(self):
        """Test to_xarray_dataset with inst.data"""
        self.testInst.load(date=self.bounds1)

        self.out = pyseas.to_xarray_dataset(self.testInst.data)
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' not in self.out.dims
        return

    def test_comp_form_instrument_variable(self):
        """Test to_xarray_dataset with inst[var]"""

        self.testInst.load(date=self.bounds1)

        self.out = pyseas.to_xarray_dataset(self.testInst['mlt'])
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' not in self.out.dims
        return

    def test_comp_form_numbers(self):
        """Test to_xarray_dataset with [float1, float2, ...., floatn]"""

        self.testInst.load(date=self.bounds1)

        vals = self.testInst['mlt'].values.tolist()
        self.out = pyseas.to_xarray_dataset(vals)
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' not in self.out.dims
        return

    def test_comp_form_list_vars(self):
        """Test to_xarray_dataset with [inst[var], inst[var2], ...]"""
        self.testInst.load(date=self.bounds1)
        self.out = pyseas.to_xarray_dataset([self.testInst['mlt'],
                                             self.testInst['longitude']])
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' in self.out.dims
        return

    def test_comp_form_list_data(self):
        """Test to_xarray_dataset with [inst.data, inst.data, ...]"""
        self.testInst.load(date=self.bounds1)
        self.out = pyseas.to_xarray_dataset([self.testInst.data,
                                             self.testInst.data])
        assert isinstance(self.out, xr.Dataset)
        assert 'pysat_binning' in self.out.dims
        return


class TestCoreXarray(TestCore):
    def setup(self):
        """Run before every method to create a clean testing setup."""
        self.testInst = pysat.Instrument(inst_module=pinsts.pysat_testing_xarray,
                                         clean_level='clean')
        self.bounds1 = self.testInst.inst_module._test_dates['']['']

    def teardown(self):
        """Run after every method to clean up previous testing."""
        del self.testInst, self.bounds1
