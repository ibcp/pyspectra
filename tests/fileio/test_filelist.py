import os

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from pyspectra import Spectra
from pyspectra.fileio import read_txt, read_bwtek, read_filelist

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TXT_FILES_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "spectra_files", "txt"
)
TXT_FILES_WITH_COMPATIBLE_WL = [
    os.path.join(TXT_FILES_DIR, file)
    for file in (
        "ftir_from_opus_01.txt",
        "ftir_from_opus_02.txt",
        "ftir_from_opus_03.txt",
    )
]

BWTEK_FILES_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "spectra_files", "txt.BWTek"
)
BWTEK_FILES_WITH_COMPATIBLE_WL = [
    os.path.join(BWTEK_FILES_DIR, file)
    for file in (
        "bwtek_exmaple_01.txt",
        "bwtek_exmaple_02.txt",
        "bwtek_exmaple_03.txt",
    )
]
BWTEK_FILES_WITH_INCOMPATIBLE_WL = [
    os.path.join(BWTEK_FILES_DIR, file)
    for file in (
        "bwtek_exmaple_01.txt",
        "bwtek_exmaple_01_manually_truncated.txt",
    )
]


def test_filelist_default():
    """Test default behaviour"""
    s = read_filelist(TXT_FILES_WITH_COMPATIBLE_WL)
    s_list = [read_txt(file) for file in TXT_FILES_WITH_COMPATIBLE_WL]

    assert isinstance(s, Spectra), "The result must be a Spectra object"
    assert s.nspc == len(
        TXT_FILES_WITH_COMPATIBLE_WL
    ), "Number of spectra is the same as the number of files"
    for i, single_spec in enumerate(s_list):
        npt.assert_equal(s.wl, single_spec.wl), "Wavelenght are the same"
        npt.assert_equal(
            s.spc.values[i, :], single_spec.spc.values[0, :]
        ), "Spectra values are the same"
    pdt.assert_frame_equal(
        s.data.reset_index(drop=True),
        pd.DataFrame({"filename": TXT_FILES_WITH_COMPATIBLE_WL}),
        "File names are included as metadata",
    )


def test_filelist_default_without_filenames():
    """Test default behaviour with keep_file_names=False"""
    s = read_filelist(TXT_FILES_WITH_COMPATIBLE_WL, keep_file_names=False)
    s_list = [read_txt(file) for file in TXT_FILES_WITH_COMPATIBLE_WL]

    assert isinstance(s, Spectra), "The result must be a Spectra object"
    assert s.nspc == len(
        TXT_FILES_WITH_COMPATIBLE_WL
    ), "Number of spectra is the same as the number of files"
    for i, single_spec in enumerate(s_list):
        npt.assert_equal(s.wl, single_spec.wl), "Wavelenght are the same"
        npt.assert_equal(
            s.spc.values[i, :], single_spec.spc.values[0, :]
        ), "Spectra values are the same"
    npt.assert_equal(
        s.data.shape, (len(TXT_FILES_WITH_COMPATIBLE_WL), 0)
    ), "Data part must be empty"


def test_filelist_callback():
    """Test with a changed callback function"""
    s = read_filelist(BWTEK_FILES_WITH_COMPATIBLE_WL, callback=read_bwtek)
    s_list = [read_bwtek(file) for file in BWTEK_FILES_WITH_COMPATIBLE_WL]

    assert isinstance(s, Spectra), "The result must be a Spectra object"
    assert s.nspc == len(
        BWTEK_FILES_WITH_COMPATIBLE_WL
    ), "Number of spectra is the same as the number of files"
    for i, single_spec in enumerate(s_list):
        npt.assert_equal(s.wl, single_spec.wl), "Wavelength are the same"
        npt.assert_equal(
            s.spc.values[i, :], single_spec.spc.values[0, :]
        ), "Spectra values are the same"
    pdt.assert_frame_equal(
        s.data.reset_index(drop=True),
        pd.DataFrame({"filename": BWTEK_FILES_WITH_COMPATIBLE_WL}),
        "File names are included as metadata",
    )


def test_filelist_bwtek_with_arguments():
    """Test kwards passed to the callback function"""
    s = read_filelist(
        BWTEK_FILES_WITH_COMPATIBLE_WL,
        callback=read_bwtek,
        meta=["Date", "laser_wavelength"],
    )
    ref_data = pd.DataFrame(
        {
            "filename": BWTEK_FILES_WITH_COMPATIBLE_WL,
            "Date": [
                "2018-11-07 14:21:58",
                "2018-11-07 14:26:34",
                "2018-11-07 14:00:59",
            ],
            "laser_wavelength": ["1064,25", "1064,25", "1064,25"],
        }
    )
    pdt.assert_frame_equal(
        s.data.reset_index(drop=True),
        ref_data[s.data.columns],
        check_names=False,
    )


def test_filelist_incompatible_wl_strict_join():
    """Test incompatible wavelengths in "strict"(default) join mode"""
    with pytest.warns(None):
        s = read_filelist(
            BWTEK_FILES_WITH_INCOMPATIBLE_WL,
            callback=read_bwtek,
            keep_file_names=False,
        )
    s_list = [read_bwtek(file) for file in BWTEK_FILES_WITH_INCOMPATIBLE_WL]

    assert isinstance(s, list), "The result must be a list of Spectra objects"
    for i, spec in enumerate(s):
        assert isinstance(spec, Spectra), "Each item must be a Spectra object"
        pdt.assert_frame_equal(spec.spc, s_list[i].spc)
        pdt.assert_frame_equal(spec.data, s_list[i].data)


@pytest.mark.parametrize(
    "join_mode, ref_file",
    [
        ("inner", "bwtek_exmaple_01_manually_truncated.txt"),
        ("outer", "bwtek_exmaple_01.txt"),
    ],
)
def test_filelist_incompatible_wl_inner_join(join_mode, ref_file):
    """Test incompatible wavelengths in "inner" and "outer" join modes"""
    s = read_filelist(
        BWTEK_FILES_WITH_INCOMPATIBLE_WL, callback=read_bwtek, join=join_mode
    )
    s_dict = {
        os.path.basename(file): read_bwtek(file)
        for file in BWTEK_FILES_WITH_INCOMPATIBLE_WL
    }

    pdt.assert_frame_equal(
        s.spc, pd.concat([obj.spc for obj in s_dict.values()], join=join_mode)
    )
    pdt.assert_frame_equal(
        s.data.reset_index(drop=True),
        pd.DataFrame({"filename": BWTEK_FILES_WITH_INCOMPATIBLE_WL}),
    )
    # Wavelength of inner join must be equal to the truncated wavelength
    # and of outer join - to wavelength of the complete file
    npt.assert_equal(s.wl, s_dict[ref_file].wl)
