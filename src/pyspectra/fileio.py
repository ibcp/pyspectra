"""TODO: Docstring"""

import warnings

import numpy as np
import pandas as pd

from .reshape import rbind
from .spectra import Spectra

__all__ = ["read_txt", "read_bwtek", "read_fileset"]


def read_txt(path):
    """TODO: Docstring"""
    data = pd.read_csv(path, header=None, names=["wl", "y"], dtype=np.float64)
    return Spectra(spc=data.y, wl=data.wl)


def read_bwtek(path, x="Raman Shift", y="Dark Subtracted #1"):
    """TODO: Docstring"""
    with open(path, "r") as fp:
        line = fp.readline()
        cnt = 0
        while line and not line.startswith("Pixel;"):
            line = fp.readline()
            cnt += 1
    if not line.startswith("Pixel;"):
        raise TypeError("Incorrect BWTek file format.")
    # Try with comma as decimal separator
    na_values = ("", " ", "  ", "   ", "    ")
    try:
        data = pd.read_csv(
            path,
            skiprows=cnt,
            sep=";",
            decimal=",",
            na_values=na_values,
            usecols=[x, y],
            dtype=np.float64,
        )
    except Exception as e:
        warnings.warn(str(e))
        data = None
    # If failed try a dot as a decimal separator
    if data is None:
        try:
            data = pd.read_csv(
                path,
                skiprows=cnt,
                sep=";",
                decimal=".",
                na_values=na_values,
                usecols=[x, y],
                dtype=np.float64,
            )
        except Exception as e:
            warnings.warn(str(e))
    # If cound not read the data by any of separators
    if data is None:
        raise TypeError(
            f"Cound not read bwtek file {path}. It seems to be incorrect file format."
        )
    # Filter rows where wl is missing
    data = data[data[x].notnull()]
    return Spectra(spc=data[y], wl=data[x])


def read_fileset(files, callback=read_txt, join="strict", keep_file_names=True):
    """TODO: Docstring"""
    spectra = [callback(f) for f in files]
    if keep_file_names:
        for i, spec in enumerate(spectra):
            spec.data["filename"] = files[i]
    if join:
        try:
            spectra = rbind(*spectra, join=join)
        except Exception as e:
            warnings.warn(str(e))
            warnings.warn(
                "Could not join spectra from files. List of spectra is returned"
            )
    return spectra
