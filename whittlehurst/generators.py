"""
fBm and ARFIMA generators for testing
"""

import warnings
import numpy as np
import cmath
from scipy.fftpack import fft, ifft
from scipy.stats import levy_stable, norm
from typing import List, Optional

def fbm(H: float, n: int = 1600, length: float = 1) -> np.ndarray:
    """
    Generate a one-dimensional fractional Brownian motion (fBm) path using an FFT-based method.

    This function computes a realization of fBm, W(t), on the interval [0, length] using n equally spaced
    grid points. The process is characterized by the Hurst exponent H, which determines the roughness of the path.
    The spectral (FFT) method is used to generate the process as described in the literature.

    Args:
        H (float): Hurst exponent in the interval [0, 1]. Values outside this range will trigger a warning.
        n (int, optional): Number of grid points to generate (default: 1600).
        length (float, optional): The endpoint of the time interval [0, length] (default: 1).

    Returns:
        np.ndarray: A realization of the fBm path over the interval [0, length].

    Raises:
        Warning: If H is not in the interval [0, 1], a warning is issued.

    References:
        Kroese, D. P., & Botev, Z. I. (2015). Spatial Process Simulation.
        In Stochastic Geometry, Spatial Statistics and Random Fields (pp. 369-404).
        Springer International Publishing. DOI: 10.1007/978-3-319-10064-7_12.
        Available at: https://sci-hub.se/10.1007/978-3-319-10064-7_12
    """
    if H < 0 or H > 1:
        return warnings.warn("Hurst parameter must be between 0 and 1")

    # Compute the autocovariance function for fBm increments
    r = np.zeros(n + 1)
    for i in range(n + 1):
        if i == 0:
            r[0] = 1
        else:
            r[i] = 0.5 * ((i+1)**(2*H) - 2*i**(2*H) + ((i-1)**(2*H)))

    # Form a symmetric sequence for the FFT
    r = np.concatenate([r, r[::-1][1:-1]])

    # Compute the FFT and then take the real part after scaling
    lmbd = np.real(fft(r) / (2*n))
    sqrt_vals = np.array([cmath.sqrt(x) for x in lmbd])

    # Generate complex Gaussian noise
    noise = np.random.normal(size=2 * n) + np.random.normal(size=2*n) * complex(0, 1)

    # Apply the FFT-based method to generate the fBm increments
    W = fft(sqrt_vals * noise)
    W = n**(-H) * np.cumsum(np.concatenate(([0], np.real(W[1:(n + 1)]))))

    # Rescale the path for the final interval [0, length]
    W = (length**H) * W
    return W

def arfima(
    ar_params: List[float] = [],
    d: float = 0,
    ma_params: List[float] = [],
    n: int = 200,
    sigma: float = 1,
    noise_alpha: float = 2,
    warmup: int = 0,
    H: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a time series from an ARFIMA process.

    The ARFIMA process incorporates an AR component, fractional differencing, and an MA component.
    If `H` is provided, the differencing order `d` is set to `H - 0.5`.

    Args:
        ar_params (List[float], optional): Coefficients for the AR component.
        d (float, optional): Differencing order for the ARFIMA process.
        ma_params (List[float], optional): Coefficients for the MA component.
        n (int, optional): Number of data points to generate (after warmup).
        sigma (float, optional): Scale (standard deviation) of the noise.
        noise_alpha (float, optional): Parameter for the alpha-stable noise distribution.
            A value of 2 corresponds to Gaussian noise.
        warmup (int, optional): Number of initial points to generate as a warmup to mitigate
            the effect of initial conditions.
        H (Optional[float], optional): If provided, sets the differencing order d to
            `H - 0.5`.

    Returns:
        np.ndarray: Generated time series of length `n`.
    """
    if H is not None:
        d = H - 0.5
    ma_series = __ma_model(ma_params, n + warmup, sigma=sigma, noise_alpha=noise_alpha)
    frac_ma = __frac_diff(ma_series, -d)
    series = __arma_model(ar_params, frac_ma)
    return series[-n:]

def __ma_model(params: List[float], n: int, sigma: float, noise_alpha: float) -> np.ndarray:
    """
    Generate a discrete time series using a Moving Average (MA) process.

    The process is defined as:
        x[t] = epsi[t] + params[0]*epsi[t-1] + params[1]*epsi[t-2] + ...,
    where the order of the MA process is inferred from the length of `params`.

    Args:
        params (List[float]): Coefficients for the MA process.
        n (int): Number of points to generate (before internal warmup removal).
        sigma (float): Scale (standard deviation) of the generated noise.
        noise_alpha (float): Parameter of the alpha-stable distribution (default: 2).
            The default value corresponds to Gaussian distribution.

    Returns:
        np.ndarray: Generated time series. Note that if `params` is non-empty,
            the returned series will have length n - len(params).
    """
    ma_order = len(params)
    if noise_alpha == 2:
        noise = norm.rvs(scale=sigma, size=(n + ma_order))
    else:
        noise = levy_stable.rvs(noise_alpha, 0, scale=sigma, size=(n + ma_order))

    if ma_order == 0:
        return noise
    ma_coeffs = np.append([1], params)
    ma_series = np.zeros(n)
    for idx in range(ma_order, n + ma_order):
        take_idx = np.arange(idx, idx - ma_order - 1, -1).astype(int)
        ma_series[idx - ma_order] = np.dot(ma_coeffs, noise[take_idx])
    return ma_series[ma_order:]

def __arma_model(params: List[float], noise: np.ndarray) -> np.ndarray:
    """
    Generate a discrete time series using an AutoRegressive Moving Average (ARMA) process.

    The AR component is defined as:
        x[t] = params[0]*x[t-1] + params[1]*x[t-2] + ... + noise[t],
    where the order of the AR process is inferred from the length of `params`.

    Args:
        params (List[float]): Coefficients for the AR component.
        noise (np.ndarray): Noise values for each step. The length of this array
            determines the length of the output series.

    Returns:
        np.ndarray: Generated time series with the same length as the `noise` array.
    """
    ar_order = len(params)
    if ar_order == 0:
        return noise
    n = len(noise)
    arma_series = np.zeros(n + ar_order)
    for idx in np.arange(ar_order, len(arma_series)):
        take_idx = np.arange(idx - 1, idx - ar_order - 1, -1).astype(int)
        arma_series[idx] = np.dot(params, arma_series[take_idx]) + noise[idx - ar_order]
    return arma_series[ar_order:]

def __frac_diff(x: np.ndarray, d: float) -> np.ndarray:
    """
    Compute the fractional difference of a time series using FFT (Jensen & Nielsen, 2014).

    The fractional differentiation is performed on the input series `x` with order `d`.

    Args:
        x (np.ndarray): Input time series.
        d (float): Order of differentiation. Typically -0.5 < d < 0.5, but the algorithm
            works for other reasonable values of d.

    Returns:
        np.ndarray: Fractionally differentiated time series.
    """

    def next_pow2(n: int) -> int:
        # Assumes that n > 1
        return (n - 1).bit_length()

    n = len(x)
    fft_len = 2 ** next_pow2(2 * n - 1)
    prod_ids = np.arange(1, n)
    frac_diff_coefs = np.append([1], np.cumprod((prod_ids - d - 1) / prod_ids))
    dx = ifft(fft(x, fft_len) * fft(frac_diff_coefs, fft_len))
    return np.real(dx[0:n])