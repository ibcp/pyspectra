"""TODO: Docstring"""

from typing import Optional

import numpy as np
import pandas as pd

from .spectra import Spectra

__all__ = ["rbind"]


def rbind(
    *objs: Spectra,
    join: str = "strict",
    data_join: Optional[str] = None,
    spc_join: Optional[str] = None
) -> Spectra:
    """TODO: Docstring"""
    if data_join is None:
        data_join = join
    if spc_join is None:
        spc_join = join

    allowed_joins = ("strict", "outer", "inner")
    if (spc_join not in allowed_joins) or (data_join not in allowed_joins):
        raise ValueError("Incorrect join strategy")
    if len(objs) <= 1:
        raise ValueError("No data to bind.")

    if spc_join == "strict":
        for obj in objs:
            if not np.array_equal(obj.wl, objs[0].wl):
                raise ValueError(
                    "Strict join is not possible: Spectra have different "
                    "wavelengths. "
                )
        spc_join = "outer"

    if data_join == "strict":
        for obj in objs:
            if not np.array_equal(obj.data.columns, objs[0].data.columns):
                raise ValueError(
                    "Strict join is not possible: Data have different columns."
                )
        data_join = "outer"

    return Spectra(
        spc=pd.concat([obj.spc for obj in objs], join=spc_join),
        data=pd.concat([obj.data for obj in objs], join=data_join),
    )
