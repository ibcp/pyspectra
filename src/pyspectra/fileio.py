"""Methods for import and export files with spectral data"""

import warnings
from typing import List, Callable, Union, Any

import numpy as np
import pandas as pd

from .reshape import rbind
from .spectra import Spectra

__all__ = ["read_txt", "read_bwtek", "read_filelist"]


def read_txt(path: str, **kwargs: Any) -> Spectra:
    """Read a spectrum file of txt or csv extension

    Usually, txt extension means csv-formatted data.  Thus, this function is
    merely a wrapper around `pandas.read_csv`.  By default, the file supposed
    to have only two columns "wavelength" and "spectral data" (the order is
    important) separated by comma and without a header line, decimal point is
    `.`.  These parameters can be changed using `**kwargs`.

    Parameters
    ----------
    path : str
        Path to a spectrum file of txt or csv extension
    **kwargs
        Arguments to be passed to `pandas.read_csv`. If number of columns or
        their order differs from expected by default (i.e. wavelength, value),
        then use `names` parameter: set name of wavelengths column to "wl",
        and name of spectral data values to "spc", i.e. like
        `names=["spc", "wl"]`

    Returns
    -------
    Spectra
        A Spectra object
    """
    options = {
        "header": None,
        "sep": ",",
        "decimal": ".",
        "names": ["wl", "spc"],
    }
    options.update(kwargs)
    data = pd.read_csv(path, **options)
    return Spectra(spc=data.spc, wl=data.wl)


def read_bwtek(
    path: str,
    wl: str = "Raman Shift",
    spc: str = "Dark Subtracted #1",
    meta: Union[str, List[str], None] = None,
) -> Spectra:
    """Read txt files of BWSpec software (BWTek)

    Parameters
    ----------
    path : str
        Path to the file to import
    wl : str, optional
        The name of the column, corresponding to wavelength values.
        Must be one of: "Pixel", "Wavelength", "Wavenumber", "Raman Shift".
        By default, "Raman Shift".
    spc : str, optional
        The name of the column, corresponding to spectral data values.
        By default, "Dark Subtracted #1".
    meta : Union[str, List[str], None], optional
        Optionally, some meta-data from the file can be imported as well.
        For example, `meta="date"` allows to import datetime of the
        spectra file. `meta=["date", "model"] imports both datetime and
        model of the equipment.

    Notes
    -----
        Rows with missing wavelength values are removed.
    """

    # Check parameters
    wl = wl.title()
    if wl not in ("Pixel", "Wavelength", "Wavenumber", "Raman Shift"):
        raise ValueError(
            f'Unexpected wl column name: {wl}. Must be one of "Pixel", '
            '"Wavelength", "Wavenumber", "Raman Shift"'
        )

    # Find a row where the data starts
    with open(path, "r") as fp:
        line = fp.readline()
        cnt = 0
        while line and not line.startswith("Pixel;"):
            line = fp.readline()
            cnt += 1
    if not line.startswith("Pixel;"):
        raise TypeError(
            "Incorrect BWTek file format. Could not to find a "
            'row starting with "Pixel;"'
        )

    # Read meta-data
    if meta is not None:
        if isinstance(meta, str):
            meta = [meta]
        meta_data = pd.read_csv(
            path, nrows=cnt, sep=";", header=None, index_col=0
        )
        meta_data = meta_data.loc[meta, :]
        meta_data = meta_data.transpose()
    else:
        meta_data = None

    # CSV read options
    options = {
        "skiprows": cnt,
        "sep": ";",
        "decimal": ",",
        "na_values": ("", " ", "  ", "   ", "    "),
        "usecols": [wl, spc],
        "dtype": np.float64,
    }
    # First, try to read with comma as a decimal separator
    try:
        data = pd.read_csv(path, **options)
    except Exception as e:
        warnings.warn(str(e))
        data = None
    # If failed try a dot as a decimal separator
    if data is None:
        options["decimal"] = "."
        try:
            data = pd.read_csv(path, **options)
        except Exception as e:
            warnings.warn(str(e))
    # If could not read the data by any of separators
    if data is None:
        raise TypeError(
            f"Could not read BWtek file {path}. It seems to be incorrect "
            "file format. "
        )
    # Filter rows where wl is missing
    data = data[data[wl].notnull()]
    return Spectra(
        spc=data[spc], wl=data[wl], data=meta_data, keep_indexes=False
    )


def read_filelist(
    files: List[str],
    callback: Callable[..., Spectra] = read_txt,
    join: str = "strict",
    keep_file_names: bool = True,
    **kwargs: Any,
) -> Union[List[Spectra], Spectra]:
    """Read a list of spectra files of same format

    Read set of spectra files of same format. By default, tries to join
    the spectra by wavelength.  If it is not possible, then a list of Spectra
    objects is returned.  The method is useful in combination with `glob.glob`
    or `os.listdir`.

    Parameters
    ----------
    files : List[str]
        List of spectral file paths
    callback : Callable[[str, ...], Spectra], optional
        A function to be called on each file.
        I.e. read_jdx for jdx format, etc.
    join : str, optional
        A join strategy. The value is passed to the `reshape` method.
        See it for the details.
    keep_file_names : bool, optional
        Keep file names as a `data` part of the output Spectra object?
    **kwargs
        Arguments to be passed to the callback function
    """
    spectra = [callback(f, **kwargs) for f in files]
    if keep_file_names:
        for i, spec in enumerate(spectra):
            spec.data["filename"] = files[i]
    if join:
        try:
            spectra = rbind(*spectra, join=join)  # type: ignore
        except Exception as e:
            warnings.warn(str(e))
            warnings.warn(
                "Could not join spectra from files. List of spectra is "
                "returned "
            )
    return spectra
