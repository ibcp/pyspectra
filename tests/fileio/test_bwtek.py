import os

import pytest
import numpy as np
import numpy.testing as npt

from pyspectra import Spectra
from pyspectra.fileio import read_bwtek

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SPECTRA_FILES_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "spectra_files", "txt.BWTek"
)
TRANCATED_BWTEK_FILE = os.path.join(
    SPECTRA_FILES_DIR, "bwtek_exmaple_01_manually_truncated.txt"
)
# ref = pd.read_csv(
#     TRANCATED_BWTEK_FILE,
#     sep=";",
#     decimal=",",
#     skiprows=88,
#     na_values=("", " ", "  ", "   ")
# )
ref = {
    "Pixel": [
        0,
        10,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        510,
        511,
    ],
    "Wavelength": [
        np.nan,
        np.nan,
        1083.93,
        1123.97,
        1164.15,
        1204.50,
        1245.03,
        1285.75,
        1326.67,
        1367.81,
        1409.18,
        1450.79,
        np.nan,
        np.nan,
    ],
    "Wavenumber": [
        np.nan,
        np.nan,
        9225.68,
        8897.07,
        8589.95,
        8302.19,
        8031.93,
        7777.56,
        7537.65,
        7310.94,
        7096.31,
        6892.78,
        np.nan,
        np.nan,
    ],
    "Raman Shift": [
        np.nan,
        np.nan,
        170.61,
        499.22,
        806.34,
        1094.10,
        1364.36,
        1618.73,
        1858.63,
        2085.35,
        2299.97,
        2503.51,
        np.nan,
        np.nan,
    ],
    "Dark": [
        9878.0,
        12709.0,
        16305.0,
        16044.0,
        8813.0,
        18128.0,
        12880.0,
        16864.0,
        16264.0,
        18326.0,
        18169.0,
        7530.0,
        16534.0,
        13271.0,
    ],
    "Raw data #1": [
        16448.0,
        12574.0,
        58329.0,
        27690.0,
        14834.0,
        22365.0,
        16843.0,
        21785.0,
        18503.0,
        19471.0,
        17989.0,
        7394.0,
        16455.0,
        13328.0,
    ],
    "Dark Subtracted #1": [
        6570.0,
        -135.0,
        42024.0,
        11646.0,
        6021.0,
        4237.0,
        3963.0,
        4921.0,
        2239.0,
        1145.0,
        -180.0,
        -136.0,
        -79.0,
        57.0,
    ],
}
ref_meta = {
    "Date": "2018-11-07 14:21:58",
    "c code": "RSM",
    "Relative Intensity Correction Flag": "0",  # the last line
}


def filter_nans(x, y):
    x = np.array(x)
    y = np.array(y)
    is_not_nan = np.logical_not(np.isnan(x))
    return x[is_not_nan], y[is_not_nan]


@pytest.mark.parametrize(
    "wl, spc",
    [
        ("Raman Shift", "Dark Subtracted #1"),  # default
        ("Wavelength", "Dark Subtracted #1"),
        ("Wavenumber", "Dark Subtracted #1"),
        ("Raman Shift", "Dark"),
        ("Raman Shift", "Raw data #1"),
        ("Pixel", "Dark"),
    ],
)
def test_read_bwtek_xy_parameters(wl, spc):
    """Test different wl and spc columns"""
    s = read_bwtek(TRANCATED_BWTEK_FILE, wl=wl, spc=spc)
    ref_wl, ref_spc = filter_nans(ref[wl], ref[spc])

    assert isinstance(s, Spectra), "The result must be a Spectra object"
    assert s.nwl == len(ref_wl), "Incorrect number of wavelength points"
    assert s.nspc == 1, "There must be only one spectrum"
    npt.assert_equal(s.wl, ref_wl)
    npt.assert_equal(s.spc.values[0, :], ref_spc)
    npt.assert_equal(s.data.shape, (1, 0), "There must be no data")


@pytest.mark.parametrize("param, value", list(ref_meta.items()))
def test_read_bwtek_single_metadata(param, value):
    """Test meta data reading"""
    s = read_bwtek(TRANCATED_BWTEK_FILE, meta=param)
    assert s.data.shape[1] == 1, "Only one column"
    assert s.data.columns[0] == param, "Colname is the same as the metadata's"
    assert s.data.iloc[0, 0] == value, "The value is correct"


def test_read_bwtek_list_metadata():
    """Test many matadata parameters"""
    s = read_bwtek(TRANCATED_BWTEK_FILE, meta=list(ref_meta.keys()))
    assert s.data.shape[1] == len(ref_meta), "All parameters are presented"
    npt.assert_equal(
        sorted(s.data.columns),
        sorted(ref_meta.keys()),
        "Colnames are the same as the metadata'",
    )
    npt.assert_equal(
        s.data.values[0, :],
        np.array([ref_meta[k] for k in s.data.columns]),
        "Values must be the same",
    )


def test_read_bwtek_incorrect_name():
    """Test incorrect wl column name"""
    with pytest.raises(ValueError):
        assert read_bwtek(TRANCATED_BWTEK_FILE, wl="whatever")
