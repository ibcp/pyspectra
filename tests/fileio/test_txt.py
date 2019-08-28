import os

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd

from pyspectra import Spectra
from pyspectra.fileio import read_txt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SPECTRA_FILES_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "spectra_files", "txt"
)


@pytest.mark.parametrize(
    "filename",
    [
        "ftir_from_opus_01.txt",
        "ftir_from_opus_02.txt",
        "ftir_from_opus_03.txt",
    ],
)
def test_read_txt_default(filename):
    path = os.path.join(SPECTRA_FILES_DIR, filename)
    s = read_txt(path)
    ref = pd.read_csv(path, sep=",", decimal=".", header=None)

    assert isinstance(s, Spectra), "The result must be a Spectra object"
    assert s.nwl == ref.shape[0], "Incorrect number of wavelength points"
    assert s.nspc == 1, "There must be only one spectrum"
    npt.assert_equal(s.wl, ref.values[:, 0], "Wavelenght values are the same")
    npt.assert_equal(
        s.spc.values[0, :], ref.values[:, 1], "Spectra values are the same"
    )
    npt.assert_equal(s.data.shape, (1, 0), "There must be no data")


def test_read_txt_with_parameters():
    path = os.path.join(SPECTRA_FILES_DIR, "ftir_na_01.txt")
    s = read_txt(path, sep="\t", dtype=np.float16)
    ref = pd.read_csv(
        path, sep="\t", decimal=".", header=None, dtype=np.float16
    )

    assert isinstance(s, Spectra), "The result must be a Spectra object"
    assert s.nwl == ref.shape[0], "Incorrect number of wavelength points"
    assert s.nspc == 1, "There must be only one spectrum"
    npt.assert_equal(s.wl, ref.values[:, 0], "Wavelenght values are the same")
    npt.assert_equal(
        s.spc.values[0, :], ref.values[:, 1], "Spectra values are the same"
    )
    assert s.spc.values.dtype == np.float16, "dtype parameter was applied"
    npt.assert_equal(s.data.shape, (1, 0), "There must be no data")
