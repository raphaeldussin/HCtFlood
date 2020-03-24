# requires pytest-datafiles
import xarray as xr
import numpy as np
import os
import pytest


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data/',
    )


def create_arrays(datafile):
    ds = xr.open_dataset(FIXTURE_DIR + datafile, decode_times=False)
    temp = ds['temp'].isel(depth=0).squeeze().to_masked_array()
    data = temp.data
    data[np.isnan(data)] = 1e+15
    mask = np.ones(data.shape)
    mask[temp.mask] = 0
    return data, mask


@pytest.mark.parametrize("datafile", ['temp_woa13_1deg.nc',
                                      'temp_woa13_025deg.nc'])
def test_flood_kara_raw(datafile):
    from HCtFlood.kara import flood_kara_raw
    temp, mask = create_arrays(datafile)
    out = flood_kara_raw(temp, mask)
    assert (np.isnan(out) == False).all()


@pytest.mark.parametrize("datafile", ['temp_woa13_1deg.nc',
                                      'temp_woa13_025deg.nc'])
def test_flood_kara_ma(datafile):
    from HCtFlood.kara import flood_kara_ma
    ds = xr.open_dataset(FIXTURE_DIR + datafile, decode_times=False)
    temp = ds['temp'].isel(depth=0).squeeze().to_masked_array()
    out = flood_kara_ma(temp, spval=1e+15)

    assert isinstance(out, np.ndarray)
    assert (np.isnan(out) == False).all()

    # test all values are NaN
    test = np.ma.masked_array(data=[[np.nan, np.nan],[np.nan, np.nan]],
                              mask=[[True, True], [True, True]])
    out = flood_kara_ma(test, spval=1e+15)
    assert (np.isnan(out) == True).all()


@pytest.mark.parametrize("datafile", ['temp_woa13_1deg.nc',
                                      'temp_woa13_025deg.nc'])
def test_flood_kara_xr(datafile):
    from HCtFlood.kara import flood_kara_xr
    ds = xr.open_dataset(FIXTURE_DIR + datafile, decode_times=False)
    temp = ds['temp'].isel(depth=0)
    out = flood_kara_xr(temp, spval=1e+15)

    assert isinstance(out, np.ndarray)
    assert (np.isnan(out) == False).all()


@pytest.mark.parametrize("datafile", ['temp_woa13_1deg.nc',
                                      'temp_woa13_025deg.nc'])
def test_flood_kara(datafile):
    from HCtFlood.kara import flood_kara
    ds = xr.open_dataset(FIXTURE_DIR + datafile, decode_times=False)
    temp = ds['temp']
    out = flood_kara(temp, zdim='depth', tdim='time')

    assert isinstance(out, xr.core.dataarray.DataArray)

    out.compute()

    assert (np.isnan(out) == False).all()
