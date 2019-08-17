import pytest

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pyspectra.baseline import vector_als


def reference_als(y, lam, p, niter=10):
    """This is reference implementation according to the article"""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


@pytest.mark.parametrize(
    "lam, p, niter",
    [(100, 0.01, 10), (1000, 0.1, 20), (10 ** 10, 0.1 ** 10, 50), (1, 0, 10)],
)
def test_vector_als_compare_with_ref(line_plus_gauss_spectrum, lam, p, niter):
    y = np.array(line_plus_gauss_spectrum["spc"])
    als_bl = vector_als(y, lam=lam, p=p, niter=niter)
    ref_bl = reference_als(y, lam=lam, p=p, niter=niter)
    assert np.allclose(als_bl, ref_bl), "Not same as reference implementation"


def test_vector_als_linear_baseline(line_plus_gauss_spectrum):
    y = np.array(line_plus_gauss_spectrum["spc"])
    als_bl = vector_als(y, lam=10 ** 9, p=0.1 ** 6)
    real_bl = np.array(line_plus_gauss_spectrum["bl"])
    assert np.allclose(als_bl, real_bl, atol=0.001), (
        "Too high deviation from reference baseline in case of "
        "linear background"
    )
