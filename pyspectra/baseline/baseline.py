"""
SpectraBaselineMethods
---------
TODO:
"""

import warnings
import logging

import numpy as np
import pandas as pd

from ..spectra import Spectra

logging.captureWarnings(True)

__all__ = ["SpectraBaselineMethods"]


class SpectraBaselineMethods:
    """TODO: docstring"""

    def __init__(self, obj: Spectra):
        self.obj = obj

    def als(self, lam: float, p: float, niter: int = 10, remove: bool = False, inplace: bool = False):
        """ TODO: Import docstring from vector version """
        from .als import vector_als

        if not self.obj.is_equally_spaced:
            warnings.warn("Wavelengths are not equally spaced. Current "
                          "Asymmetric Least Squares implementation assumes "
                          "wavelengths to be equally spaced, this might "
                          "lead to (usually minor) mistakes in result")

        bl = np.apply_along_axis(vector_als, 1, self.obj.spc.values, lam=lam, p=p, niter=niter)
        if remove:
            bl = self.obj.spc.values - bl
        if inplace:
            self.obj.spc = pd.DataFrame(bl, columns=self.obj.wl)
            return self.obj
        else:
            return Spectra(spc=bl, wl=self.obj.wl, data=self.obj.data)
