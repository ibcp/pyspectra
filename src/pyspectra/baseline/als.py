import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def vector_als(y: np.array, lam: float, p: float, niter: int = 10) -> np.array:
    """Baseline correction based on Asymmetric Least Squares [1]

    A Quote from the article:
    > There are two parameters: `p` for asymmetry and λ for smoothness.
    > Both have to be tuned to the data at hand.  We found that generally
    > 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks)
    > and 10^2 ≤ λ ≤ 10^9 , but exceptions may occur. In any case one should
    > vary λ on a grid that is approximately linear for log λ.

    Parameters
    ----------
    y : np.array
        A vector of values
    lam : float
        Smoothness parameter. The higher `lam` the smoother baseline
    p : float
        Asymmetry parameter. If original signal is higher than baseline
        then residual is weighted as `p`*`residual`; otherwise (baseline
        is above the signal) `1-p`*`residual`.
    niter : int, optional
        Number of iterations. Default is 10, as described in the original
        article.

    Returns
    -------
    np.array
        Vector of baseline values

    Footnotes
    ---------
    .. [1] "Baseline Correction with Asymmetric Least Squares Smoothing",
       P. Eilers and H. Boelens, 2005
    """
    m = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m, m - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(m)
    W = sparse.spdiags(w, 0, m, m)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z
