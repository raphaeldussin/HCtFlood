# requires pytest-datafiles
import xarray as xr
import numpy as np
import os


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


data_1deg, mask_1deg = create_arrays('temp_woa13_1deg.nc')
data_025deg, mask_025deg = create_arrays('temp_woa13_025deg.nc')


def test_kara():
    from HCtFlood.kara import flood_kara
    out = flood_kara(data_1deg, mask_1deg)
    assert (np.isnan(out) == False).all()

    out = flood_kara(data_025deg, mask_025deg)
    assert (np.isnan(out) == False).all()
    return None
