import pytest
import numpy as np


@pytest.fixture
def line_plus_gauss_spectrum():
    x = np.linspace(-5, 5, 100)
    gauss = np.exp(-np.power(x, 2) / 2)
    line = -0.2 * x + 1
    y = gauss + line
    x = 10 * (x + 50)
    return {"wl": x.tolist(), "spc": y.tolist(), "bl": line.tolist()}
