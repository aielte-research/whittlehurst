"""
Estimate the Hurst exponent of a time series using the Whittle likelihood method.

The initial implementation of Whittle's method was based on the following:
https://github.com/JFBazille/ICode/blob/master/ICode/estimators/whittle.py

For fractional Gaussian noise (fractional Brownian motion increments) we use methods for calculating the spectral density as described here:
https://onlinelibrary.wiley.com/doi/full/10.1111/jtsa.12750
"""

import numpy as np
import scipy.optimize as so
from .spectraldensity import arfima, fGn, fGn_paxson, fGn_truncation, fGn_taylor


def whittle(seq, spectrum="fGn", K=None, spectrum_callback=None):
    """
    Estimate the Hurst exponent of a time series using the Whittle likelihood method.

    This function computes an estimate of the Hurst exponent (H) by minimizing the Whittle
    likelihood function. It fits a theoretical spectral density model to the periodogram of
    the input sequence. The available spectral density models include:
      - 'fGn' or 'fGn_paxson': Fractional Gaussian noise using Paxson's approximation.
      - 'fGn_truncated': Fractional Gaussian noise with a truncated calculation of spectral density.
      - 'arfima': ARFIMA(0, H - 0.5, 0) spectral density model.

    Parameters
    ----------
    seq : array_like
        1D array or sequence of numerical values representing the time series.
    spectrum : str, optional
        A string identifier for the spectral density model to be used. The default is "fGn".
        Recognized values include "fGn", "fGn_paxson", "fGn_truncated", and "arfima".
    spectrum_callback : callable, optional
        A custom function that computes the theoretical spectral density given H and n.
        If None, the function will select a default model based on the `spectrum` parameter.

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
                K=50 
            spectrum_callback = lambda H, n: fGn_paxson(H, n, K)
        elif spectrum.lower() == "fgn_truncation":
            if K is None:
                K=200
            spectrum_callback = lambda H, n: fGn_truncation(H, n, K)
        elif spectrum.lower() == "fgn_taylor":
            spectrum_callback = fGn_taylor
        elif spectrum.lower() == "arfima":
            spectrum_callback = arfima
        else:
            raise Exception("Unrecognized spectral model: {}".format(spectrum))

    n = len(seq)
    tmp = np.abs(np.fft.fft(seq))
    gammahat = np.exp(
        2 * np.log(tmp[1:int((n - 1) / 2) + 1])) / (2 * np.pi * n)
    func = lambda H: whittlefunc(H, gammahat, n, spectrum_callback)
    return so.fminbound(func, 0, 1)

def whittlefunc(H, gammahat, n, spectrum_callback):
    """
    Evaluate the Whittle likelihood function for a given Hurst exponent.

    This function calculates the value of the Whittle likelihood function for a candidate
    Hurst exponent H by comparing the empirical spectral density (derived from the periodogram)
    with the theoretical spectral density computed using the provided `spectrum_callback` function.
    The resulting value serves as the objective for the optimization routine that estimates H.

    Parameters
    ----------
    H : float
        The candidate Hurst exponent to evaluate.
    gammahat : numpy.ndarray
        The empirical spectral density values obtained from the periodogram of the time series.
    n : int
        The length of the time series, used for normalization.
    spectrum_callback : callable
        A function that returns the theoretical spectral density given a Hurst exponent H and the number
        of data points n.

    Returns
    -------
    float
        The computed value of the Whittle likelihood function for the candidate H, which is minimized
        during the estimation process.

    Notes
    -----
    The function computes the theoretical spectral density using the provided callback, then calculates
    the ratio between the empirical and theoretical densities. This ratio is summed and scaled to form the
    Whittle likelihood value.
    """
    gammatheo = spectrum_callback(H, n)
    qml = gammahat / gammatheo
    return 2 * (2 * np.pi / n) * np.sum(qml)
