"""
Spectra
---------
TODO:
"""

import logging
import warnings
from typing import Any, Optional, Union, Tuple, List, Callable, NoReturn

import numpy as np
import pandas as pd

logging.captureWarnings(True)

__all__ = ["Spectra"]


def _is_empty_slice(param: Any) -> bool:
    return (
        isinstance(param, slice)
        and (param.start is None)
        and (param.stop is None)
        and (param.step is None)
    )


class Spectra:
    """ TODO:
    Parameters
    ----------
    Examples
    --------
    See also
    --------
    """

    # ----------------------------------------------------------------------
    # Constructor

    def __init__(  # noqa: C901
        self,
        spc: Any = None,
        wl: Any = None,
        data: Any = None,
        labels: Union[dict, Tuple[str, str], List[str], None] = None,
        keep_indexes: bool = True,
    ) -> None:

        # Parse spc and wl
        if spc is None and wl is None:
            raise ValueError("At least one of spc or wl must be provided!")

        # Prepare SPC
        if spc is not None:
            if isinstance(spc, list) or isinstance(spc, tuple):
                spc = np.array(spc)

            if isinstance(spc, np.ndarray):
                if spc.ndim == 1:
                    spc = pd.DataFrame(spc.reshape(1, len(spc)))
                elif spc.ndim == 2:
                    spc = pd.DataFrame(spc)
                else:
                    raise ValueError("Incorrect spc is provided!")

            if isinstance(spc, pd.Series):
                spc = pd.DataFrame(spc).T
        else:
            spc = pd.DataFrame(columns=pd.Float64Index(wl))

        # Prepare wl
        if wl is None:
            if isinstance(spc, pd.DataFrame) and isinstance(
                spc.columns, pd.Float64Index
            ):
                wl = spc.columns
            else:
                warnings.warn(
                    "Wavelength is not provided: using range 0:ncol(spc) "
                    "instead. "
                )
                wl = list(range(spc.shape[1]))

        # Combine spc and wl
        spc.columns = pd.Float64Index(wl)
        # spc = spc.reindex(sorted(spc.columns), axis="columns", copy=False)
        self.spc = spc

        # Parse data
        if data is None:
            data = pd.DataFrame(index=self.spc.index)
        elif isinstance(data, dict):
            data = pd.DataFrame(data, index=self.spc.index)
        self.data = data

        # Parse labels
        if labels:
            if (
                isinstance(labels, dict)
                and labels.get("x", False)
                and labels.get("y", False)
            ):
                self.labels = {"x": labels["x"], "y": labels["y"]}
            elif (isinstance(labels, tuple) or isinstance(labels, list)) and (
                len(labels) == 2
            ):
                self.labels = {"x": labels[0], "y": labels[1]}
            else:
                raise ValueError("Incorrect labels type!")
        else:
            self.labels = {"x": None, "y": None}

        # Checks
        # if self.spc.shape[1] != len(self.wl):
        #    raise ValueError("length of wavelength must be equal to number "
        #                     "of columns in spc!")
        if self.spc.shape[0] != self.data.shape[0]:
            raise ValueError(
                "data must have the same number of instances(rows) as spc has!"
            )

        # Reset indexes to make them the same
        if not np.array_equal(
            np.array(self.spc.index), np.array(self.data.index)
        ):
            if keep_indexes:
                raise ValueError(
                    "spc and data have different indexes. Make them equal or "
                    "set keep_indexes to False "
                )
            else:
                self.spc.reset_index(drop=True, inplace=True)
                self.data.reset_index(drop=True, inplace=True)
        if not self.spc.index.is_unique:
            warnings.warn("Index is not unique.")

    # ----------------------------------------------------------------------
    # Internal helpers

    def _parse_string_or_column_param(
        self, param: Union[str, pd.Series, np.ndarray, list, tuple]
    ) -> pd.Series:
        if isinstance(param, str) and (param in self.data.columns):
            return self.data[param]
        elif isinstance(param, pd.Series) and (param.shape[0] == self.nspc):
            return param
        elif (
            isinstance(param, np.ndarray)
            and (param.ndim == 1)
            and (param.shape[0] == self.nspc)
        ):
            return pd.Series(param)
        elif isinstance(param, (list, tuple)) and (len(param) == self.nspc):
            return pd.Series(param)
        else:
            raise TypeError(
                "Incorrect parameter. It must be either a string of a data "
                "column name or pd.Series / np.array / list / tuple of "
                "lenght equal to number of spectra. "
            )

    # ----------------------------------------------------------------------
    # Properties for quick access

    @property
    def wl(self) -> np.ndarray:
        """Get wavelengths

        """
        return self.spc.columns.values

    @property
    def shape(self) -> Tuple[int, int, int]:
        """A tuple representing the dimensionality of the Spectra

        Returns
        -------
        Tuple[int, int, int]:
            A tuple of the following structure:
            1. number of spectra (i.e. number of rows)
            2. number of data columns
            3. number of wavelength points
        """
        return self.nspc, self.data.shape[1], self.nwl

    @property
    def nwl(self) -> int:
        """Number of wavelength points

        """
        return self.spc.shape[1]

    @property
    def nspc(self) -> int:
        """Number of spectra in the object

        """
        return self.spc.shape[0]

    @property
    def is_equally_spaced(self) -> bool:
        """Are wavelength values equally spaced?

        """
        return len(np.unique(self.wl[1:] - self.wl[:-1])) == 1

    # ----------------------------------------------------------------------
    # Coping

    def copy(self) -> "Spectra":
        return Spectra(spc=self.spc.copy(), data=self.data.copy())

    def deepcopy(self) -> "Spectra":
        return Spectra(spc=self.spc.deepcopy(), data=self.data.deepcopy())

    # ----------------------------------------------------------------------
    # Pre-processing

    @property
    def baseline(
        self
    ) -> "SpectraBaselineMethods":  # type: ignore # noqa: F821
        from .baseline import SpectraBaselineMethods

        return SpectraBaselineMethods(self)

    def smooth(
        self,
        how: Union[str, Callable[[np.ndarray], Union[int, float]]],
        w: int,
        inplace: bool = False,
        **kwargs: Any,
    ) -> "Spectra":
        """ TODO: Move to sub-module like baseline """
        if not (w % 2):
            raise ValueError("The window size must be an odd number.")
        if w < 3:
            raise ValueError("The window size is too small.")
        if self.nwl < w:
            raise ValueError(
                "The window size is bigger than number of wl points."
            )

        if how == "savgol":
            from scipy.signal import savgol_filter

            if not self.is_equally_spaced:
                warnings.warn(
                    "Wavelengths are not equally spaced. Current "
                    "Savitzky-Golay implementation assumes "
                    "wavelengths to be equally spaced, this might "
                    "lead to (usually minor) mistakes in result"
                )

            newspc = pd.DataFrame(
                savgol_filter(
                    self.spc.values,
                    w,
                    **kwargs,
                    axis=1,
                    mode="constant",
                    cval=np.nan,
                ),
                columns=self.spc.columns,
            )
        elif how == "mean":
            newspc = self.spc.rolling(w, axis=1, center=True).mean()
        elif how == "median":
            newspc = self.spc.rolling(w, axis=1, center=True).median()
        elif callable(how):
            newspc = self.spc.rolling(w, axis=1, center=True).apply(
                how, raw=True
            )

        if inplace:
            self.spc = newspc
            return self
        return Spectra(spc=newspc, data=self.data)

    def outliers(
        self,
        how: str = "iqr",
        out: str = "bool",
        iqr_width: float = 1.5,
        **kwargs: Any,
    ) -> np.ndarray:
        if how == "iqr":
            q1 = self.spc.quantile(q=0.25, **kwargs)
            q3 = self.spc.quantile(q=0.75, **kwargs)
            iqr = q3 - q1
            lower_bound = q1 - iqr_width * iqr
            upper_bound = q3 + iqr_width * iqr
            is_outlier = ~(
                self.spc.apply(
                    lambda x: x.between(lower_bound, upper_bound), axis=1
                )
                .all(axis=1)
                .values
            )
        else:
            raise ValueError("Unknown type of outliers detection.")
        # Prepare output according to format in `out`
        if out == "bool":
            return is_outlier
        elif out == "label":
            return self.spc.index[is_outlier].values
        elif out == "index":
            return np.where(is_outlier)
        else:
            raise ValueError("Unknown output format.")

    def approx_na(self, inplace: bool = False, **kwargs: Any) -> "Spectra":
        kwargs.pop("axis", None)
        method = kwargs.pop("method", "index")
        if inplace:
            self.spc.interpolate(method=method, axis=1, inplace=True, **kwargs)
            return self
        else:
            result = self.copy()
            result.spc.interpolate(
                method=method, axis=1, inplace=True, **kwargs
            )
            return result

    # ----------------------------------------------------------------------
    # Get and set items

    def __setitem__(self, given: Tuple[Any, Any, Any], value: Any) -> None:
        if (type(given) == tuple) and (len(given) == 3):
            rows, cols, wls = (
                [x]
                if (np.size(x) == 1)
                and (not isinstance(x, (slice, list, tuple)))
                else x
                for x in given
            )
            if _is_empty_slice(cols) and not _is_empty_slice(wls):
                idx = pd.IndexSlice
                self.spc.loc[rows, idx[wls]] = value
            elif not _is_empty_slice(cols) and _is_empty_slice(wls):
                self.data.loc[rows, cols] = value
            else:
                raise ValueError(
                    "Incorrect subset value for assignment. Either of data "
                    "columns for wavelengths indexes must be `:` "
                )
        else:
            raise ValueError(
                "Incorrect subset value. Provide 3 values in format <row, "
                "column, wl>. "
            )

    def __getitem__(self, given: Tuple[Any, Any, Any]) -> "Spectra":
        logging.debug(f"Trying to get item by {given}")
        if (type(given) == tuple) and (len(given) == 3):
            rows, cols, wls = (
                [x]
                if (np.size(x) == 1)
                and (not isinstance(x, (slice, list, tuple)))
                else x
                for x in given
            )
            logging.debug(f"  Rows selector is {rows}")
            logging.debug(f"  Data columns selector is {cols}")
            logging.debug(f"  Spectra wavelength range is {wls}")
            idx = pd.IndexSlice
            return Spectra(
                spc=self.spc.loc[rows, idx[wls]],
                data=self.data.loc[rows, cols],
            )
        else:
            raise ValueError(
                "Incorrect subset value. Provide 3 values in format "
                "<row, column:, wl>. "
            )

    # ----------------------------------------------------------------------

    def reset_index(
        self, drop: bool = False, inplace: bool = False
    ) -> "Spectra":
        if inplace:
            self.spc.reset_index(drop=True, inplace=True)
            self.data.reset_index(drop=drop, inplace=True)
            return self
        else:
            result = self.copy()
            result.spc.reset_index(drop=True, inplace=True)
            result.data.reset_index(drop=drop, inplace=True)
            return result

    # ----------------------------------------------------------------------
    # Arithmetic operations +, -, *, /, **, abs, round, ceil, etc.

    def __add__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__add__(other), wl=self.wl, data=self.data
        )

    def __sub__(self, other: Any) -> "Spectra":
        if isinstance(other, type(self)):
            return Spectra(
                spc=self.spc.values.__sub__(other.spc.values),
                wl=self.wl,
                data=self.data,
            )
        else:
            return Spectra(
                spc=self.spc.values.__sub__(other.spc.values),
                wl=self.wl,
                data=self.data,
            )

    def __mul__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__mul__(other), wl=self.wl, data=self.data
        )

    def __truediv__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__truediv__(other), wl=self.wl, data=self.data
        )

    def __pow__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__pow__(other), wl=self.wl, data=self.data
        )

    def __radd__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__radd__(other), wl=self.wl, data=self.data
        )

    def __rsub__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__rsub__(other), wl=self.wl, data=self.data
        )

    def __rmul__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__rmul__(other), wl=self.wl, data=self.data
        )

    def __rtruediv__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__rtruediv__(other), wl=self.wl, data=self.data
        )

    def __iadd__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__iadd__(other), wl=self.wl, data=self.data
        )

    def __isub__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__isub__(other), wl=self.wl, data=self.data
        )

    def __imul__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__imul__(other), wl=self.wl, data=self.data
        )

    def __itruediv__(self, other: Any) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__itruediv__(other), wl=self.wl, data=self.data
        )

    def __abs__(self) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__abs__(), wl=self.wl, data=self.data
        )

    def __round__(self, n: int) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__round__(n), wl=self.wl, data=self.data
        )

    def __floor__(self) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__floor__(), wl=self.wl, data=self.data
        )

    def __ceil__(self) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__ceil__(), wl=self.wl, data=self.data
        )

    def __trunc__(self) -> "Spectra":
        return Spectra(
            spc=self.spc.values.__trunc__(), wl=self.wl, data=self.data
        )

    # ----------------------------------------------------------------------
    # Plots

    def plot(
        self,
        columns: Any = None,
        rows: Any = None,
        color: Any = None,
        palette: Optional[List[str]] = None,
        fig: Optional["plt.Figure"] = None,  # type: ignore
        sharex: bool = False,
        sharey: bool = False,
        legend_params: Optional[dict] = None,
        title_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> "plt.Figure":  # type: ignore # noqa: F821
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # Prepare columns and rows
        if rows is None:
            row = pd.Series(np.repeat("dummy", self.nspc), dtype="category")
        else:
            row = self._parse_string_or_column_param(
                rows
            ).cat.remove_unused_categories()
        row = (
            row.cat.add_categories("NA")
            .fillna("NA")
            .cat.remove_unused_categories()
            .reset_index(drop=True)
        )

        if columns is None:
            col = pd.Series(np.repeat("dummy", self.nspc), dtype="category")
        else:
            col = self._parse_string_or_column_param(columns)
        col = (
            col.cat.add_categories("NA")
            .fillna("NA")
            .cat.remove_unused_categories()
            .reset_index(drop=True)
        )

        nrows = len(row.cat.categories)
        ncols = len(col.cat.categories)

        # Prepare colors and labels
        if color is None:
            labels = pd.Series(["spc"] * self.nspc, dtype="category")
        else:
            labels = (
                self._parse_string_or_column_param(color)
                .astype("category")
                .cat.add_categories("NA")
                .fillna("NA")
                .cat.remove_unused_categories()
            )
        ncolors = len(labels.cat.categories)
        if palette is None:
            palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cmap = dict(zip(labels.cat.categories, palette[:ncolors]))
        cmap.update({"NA": "gray"})
        colors = labels.cat.rename_categories(cmap)

        # Prepare figure with subplots
        if fig is None:
            fig = plt.figure()

        fig, ax = plt.subplots(
            nrows,
            ncols,
            sharex=sharex,
            sharey=sharey,
            num=fig.number,
            clear=True,
            squeeze=False,
        )

        # Prepare legend lines if needed
        if "legend" in kwargs:
            show_legend = kwargs.get("legend")
        else:
            show_legend = True
        legend_lines = [
            Line2D([0], [0], color=c, lw=4) for c in colors.cat.categories
        ]
        # For each combination of row and column categories
        for i, vrow in enumerate(row.cat.categories):
            for j, vcol in enumerate(col.cat.categories):
                # Filter all spectra related to the current subplot
                rowfilter = (row == vrow) & (col == vcol)
                rowfilter = rowfilter.tolist()
                if np.any(rowfilter):
                    self.spc.loc[rowfilter, :].T.plot.line(
                        ax=ax[i, j], color=colors[rowfilter], **kwargs
                    )
                # Plot legend if needed
                if show_legend:
                    if legend_params is None:
                        legend_params = {}
                    ax[i, j].legend(
                        legend_lines, labels.cat.categories, **legend_params
                    )
                # For the first rows and columns set titles
                if title_params is None:
                    title_params = {}
                if (i == 0) and (columns is not None):
                    ax[i, j].set_title(str(vcol), **title_params)
                if (j == 0) and (rows is not None):
                    ax[i, j].set_ylabel(str(vrow), **title_params)
        return fig

    # ----------------------------------------------------------------------

    def __str__(self) -> str:
        return str(self.shape)
