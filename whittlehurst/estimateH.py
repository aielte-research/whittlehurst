"""
Estimate the Hurst exponent of a time series using the Whittle likelihood method.

The initial implementation of Whittle's method was based on the following:
https://github.com/JFBazille/ICode/blob/master/ICode/estimators/whittle.py

For fractional Gaussian noise (fractional Brownian motion increments) we use methods for calculating the spectral density as described here:
https://onlinelibrary.wiley.com/doi/full/10.1111/jtsa.12750
"""

import numpy as np
from scipy.optimize import fminbound
from typing import Optional
from .spectraldensity import arfima, fGn, fGn_paxson, fGn_truncation, fGn_taylor

def whittle(
    seq,
    spectrum: str = "fGn",
    K: Optional[int] = None,
    spectrum_callback = None
) -> float:
    """
    Estimate the Hurst exponent of a time series using the Whittle likelihood method.

    This function computes an estimate of the Hurst exponent (H) by minimizing the Whittle
    likelihood function. It fits a theoretical spectral density model to the periodogram of the
    input sequence. The available spectral density models include:
      - "fGn": fGn spectrum epression using the Hurwitz zeta function.
      - "fGn_paxson": fGn spectrum using Paxson's approximation.
      - "fGn_truncation": fGn spectrum with a truncated calculation.
      - "fGn_taylor": fGn spectrum using Taylor-series approximation.
      - "arfima": ARFIMA(0, H - 0.5, 0) spectral density.

    Parameters
    ----------
    seq : Sequence
        1D array or sequence of numerical values representing the time series.
    spectrum : str, optional
        Identifier for the spectral density model to be used. Recognized values are:
        "fGn", "fGn_paxson", "fGn_truncation", "fGn_taylor", and "arfima". Default is "fGn".
    K : int, optional
        Model-specific parameter used in some spectral models (e.g., "fGn_paxson" and "fGn_truncation").
        If not provided, a default value is used (50 for "fGn_paxson" and 200 for "fGn_truncation").
    spectrum_callback : callable, optional
        A custom function that computes the theoretical spectral density given H and n.
        If None, a model is selected based on the `spectrum` parameter.

    Returns
    -------
    float
        The estimated Hurst exponent, determined by minimizing the Whittle likelihood function
        over the interval [0, 1].

    Raises
    ------
    Exception
        If an unrecognized spectral model string is provided and no custom callback is given.

    Notes
    -----
    The function computes the empirical periodogram of the input sequence using the Fourier transform,
    then compares it to the theoretical spectral density computed by the chosen model. The Whittle
    likelihood function is minimized using `scipy.optimize.fminbound` to estimate the optimal Hurst exponent.
    """
    if spectrum_callback is None:
        if spectrum.lower() == "fgn":
            spectrum_callback = fGn
        elif spectrum.lower() == "fgn_paxson":
            if K is None:
                K = 50
            spectrum_callback = lambda H, n: fGn_paxson(H, n, K)
        elif spectrum.lower() == "fgn_truncation":
            if K is None:
                K = 200
            spectrum_callback = lambda H, n: fGn_truncation(H, n, K)
        elif spectrum.lower() == "fgn_taylor":
            spectrum_callback = fGn_taylor
        elif spectrum.lower() == "arfima":
            spectrum_callback = arfima
        else:
            raise Exception("Unrecognized spectral model: {}".format(spectrum))

    n = len(seq)
    gammahat = np.abs(np.fft.fft(seq))[1 : (n-1)//2 + 1]**2
    func = lambda H: np.sum(gammahat/spectrum_callback(H, n))
    return fminbound(func, 0, 1) # type: ignore

def variogram(path, p: float = 1) -> float:
    """
    Estimate the Hurst exponent using a p-order variogram estimator.

    This estimator computes the pth-order variogram based on differences in the sample path.
    The order parameter p allows for different estimators:
      - p = 1 corresponds to the madogram,
      - p = 2 corresponds to the classical variogram,
      - p = 1/2 corresponds to the rodogram.

    Parameters
    ----------
    path : List[float]
        The sample path (e.g., a realization of fractional Brownian motion).
    p : float, optional
        The order of variation (default is 1).

    Returns
    -------
    np.float64
        The estimated Hurst exponent computed using the variogram method.
    """
    # Calculate summed absolute differences raised to the power p.
    sum1: float = np.sum([np.abs(path[i] - path[i-1])**p for i in range(len(path))])
    sum2: float = np.sum([np.abs(path[i] - path[i-2])**p for i in range(len(path))])

    def vp(increments: float, l: int) -> float:
        return (1 / (2 * (len(path) - l))) * increments

    return 1 / p * ((np.log(vp(sum2, 2)) - np.log(vp(sum1, 1))) / np.log(2))

def tdml(y):
    """
    Estimate the Hurst parameter H using the TDML method.
    y: 1D numpy array of fGn observations
    Returns the estimated H
    """
    # Optimize the negative log likelihood with respect to H
    return fminbound(lambda H: tdml_negll_fgn(H, y), 0, 1)

def tdml_negll_fgn(H, y):
    """
    Computes an optimized version of the negative profile log likelihood
    for a given H, using the Durbin-Levinson recursion.
    Any factors independent of H or y have been removed for efficiency.
    """
    import numpy as np

    n = len(y)
    k = np.arange(n)
    # Compute the theoretical autocovariances for fGn (with sigma^2 = 1)
    gamma = 0.5 * (np.abs(k-1)**(2*H) - 2*(k**(2*H)) + (k+1)**(2*H))
    
    S = y[0]**2
    log_v_sum = 0.0
    v_current = 1.0
    a_prev = np.array([])
    
    for t in range(1, n):
        # When a_prev is empty (t == 1), the dot product yields 0.
        kappa = (gamma[t] - np.dot(a_prev, gamma[t-1:0:-1])) / v_current
        a_new = np.empty(t)
        a_new[:-1] = a_prev - kappa * a_prev[::-1]
        a_new[-1] = kappa
        v_current *= (1 - kappa**2)
        log_v_sum += np.log(v_current)
        pred = np.dot(a_new, y[t-1::-1])
        err = y[t] - pred
        S += err**2 / v_current
        a_prev = a_new
        
    sigma2_hat = S / n
    return n * np.log(sigma2_hat) + log_v_sum