import numpy as np
from scipy.special import gamma, zeta

def arfima(H: float, n: int):
    """
    Compute the spectral density of an ARFIMA(0, H - 0.5, 0) process.

    This function calculates the spectral density for a fractionally integrated process
    with differencing parameter d = H - 0.5. The density is computed for the first (n-1)/2 
    Fourier frequencies using the formula:
    
        f(λ) = (2sin(λ/2))^(-2*d) / (2π)
        
    Important Note
    ----------
    Terms independent from λ or H are omitted, as they are not required to minimize the whittle objective

    Parameters
    ----------
    H : float
        Hurst exponent; d is calculated as H - 0.5.
    n : int
        Total number of frequency points; spectral density is computed for (n-1)/2 frequencies.

    Returns
    -------
    numpy.ndarray
        Array of spectral density values.
    """
    
    dpl = np.arange(1, n//2 + 1) * (np.pi/n)
    return np.abs(2 * np.sin(dpl))**(1 - 2*H)


def fGn_hurwitz(H: float, n: int):
    """
    Compute the spectral density for fractional Gaussian noise (fGn)
    using the computationally feasible expression from Shi et al. (2024).

    This function calculates the spectral density at the Fourier frequencies
    using the expression based on the Hurwitz zeta function:
    
      f(λ) = 2 * CH * (1 - cos(λ)) * (2π)^(-1-2H) * [ζ(1+2H, 1 - λ/(2π)) + ζ(1+2H, λ/(2π))]
    
    where CH = (Γ(2H+1) * sin(πH)) / (2π).
    
    Parameters
    ----------
    H : float
        Hurst exponent (in (0, 1)).
    n : int
        Number of data points, which is used to define the Fourier frequencies.
    
    Returns
    -------
    numpy.ndarray
        Array of computed spectral density values at the Fourier frequencies.
    """
    s = 2*H + 1

    fspec = 2 * gamma(s) * np.sin(np.pi * H) * (2*np.pi)**(-s-1)
    
    dpl = np.arange(1, n//2 + 1) / n
    term1 = zeta(s, 1 - dpl)
    term2 = zeta(s, dpl)
    
    fspec *= (1 - np.cos(2 * np.pi * dpl)) * (term1 + term2)
    
    # Normalize the spectral density
    fspec /= np.exp(np.mean(np.log(fspec)))

    return fspec

def fGn_paxson(H: float, n: int, K: int = 10):
    """
    Compute the approximate spectral density for fractional Gaussian noise (fGn)
    using Paxson's approximation method.

    This function estimates the fGn spectral density at the Fourier frequencies
    for a given Hurst exponent H. The approximation involves a primary term, a summation
    over K truncation terms, and a correction term.

    Parameters
    ----------
    H : float
        Hurst exponent (in (0, 1)).
    n : int
        Number of data points, used to define the Fourier frequencies.
    K : int, optional
        Truncation parameter for the approximation (default is 50).

    Returns
    -------
    numpy.ndarray
        Array of approximated spectral density values at the Fourier frequencies.
    """
    # Compute parameters
    s = 2 * H + 1
    
    fspec = gamma(s) * np.sin(np.pi * H) / np.pi

    # Define Fourier frequencies
    lmbd = 2 * np.pi * np.arange(1, n//2 + 1) / n

    # Create an array of k values from -K to K
    k_vals = np.arange(-K, K + 1)
    
    # Compute the truncated sum over k for each Fourier frequency
    truncation = np.sum(np.abs(2 * np.pi * k_vals.reshape(-1, 1) + lmbd.reshape(1, -1)) ** (-s), axis=0)

    # Correction term: a(K,λ)
    def a_term(k, lmbd_val):
        return ((2 * np.pi * k + lmbd_val)**(1 - s) + (2 * np.pi * k - lmbd_val)**(1 - s)) / (4 * np.pi * H)

    correction = (a_term(K, lmbd) + a_term(K + 1, lmbd)) / 2

    # Combine terms to compute the spectral density
    fspec *= (1 - np.cos(lmbd)) * (truncation + correction)
    
    # Normalize the spectral density
    fspec /= np.exp(np.mean(np.log(fspec)))

    return fspec

def fGn_truncation(H: float, n: int, K: int = 2000):
    """
    Compute the approximate spectral density for fractional Gaussian noise (fGn)
    using the truncation method.

    This function estimates the fGn spectral density at the Fourier frequencies
    for a given Hurst exponent H by truncating the infinite summation in the spectral density 
    expression to a finite sum from k = -K to K.

    The spectral density of fGn is defined as:
      f(λ) = 2 CH (1 - cos(λ)) * Σₖ|2πk + λ|^(-1-2H),
    and here we approximate it by:
      f(λ) ≈ 2 CH (1 - cos(λ)) * Σₖ₌₋K^K |2πk + λ|^(-1-2H).

    Parameters
    ----------
    H : float
        Hurst exponent (in (0, 1)).
    n : int
        Number of data points, which defines the Fourier frequencies.
    K : int, optional
        Truncation parameter for the infinite sum (default is 2000).

    Returns
    -------
    numpy.ndarray
        Array of approximated spectral density values at the Fourier frequencies.
    """
    # Compute constant parameters
    # Compute parameters
    s = 2 * H + 1
    
    fspec = gamma(s) * np.sin(np.pi * H) / np.pi

    # Define Fourier frequencies
    lmbd = 2 * np.pi * np.arange(1, n//2 + 1) / n

    # Create an array of k values from -K to K
    k_vals = np.arange(-K, K + 1)
    
    # Compute the truncated sum over k for each Fourier frequency
    fspec *= (1 - np.cos(lmbd)) * np.sum(np.abs(2 * np.pi * k_vals.reshape(-1, 1) + lmbd.reshape(1, -1)) ** (-s), axis=0)
    
    # Normalize the spectral density
    fspec /= np.exp(np.mean(np.log(fspec)))

    return fspec

def fGn_taylor(H: float, n: int):
    """
    Compute the approximate spectral density for fractional Gaussian noise (fGn)
    using the Taylor-series expansion at near-zero frequencies.

    The Taylor-series approximation is valid as λ → 0 and approximates the spectral 
    density by:
    
        f(λ) ≈ C_H * λ^(1-2H),
        
    where 
        C_H = (Γ(2H+1) * sin(πH))/(2π).

    Parameters
    ----------
    H : float
        Hurst exponent (in (0, 1)).
    n : int
        Number of data points, which defines the Fourier frequencies.
    
    Returns
    -------
    numpy.ndarray
        Array of approximated spectral density values at the Fourier frequencies.
        (Note: the approximation is intended for small frequencies.)
    """
    # Compute constant C_H
    CH = gamma(2 * H + 1) * np.sin(np.pi * H) / (2 * np.pi)

    # Define Fourier frequencies (excluding zero)
    lmbd = 2 * np.pi * np.arange(1, n//2 + 1) / n

    # Taylor-series approximation for the spectral density as λ → 0
    fspec = CH * lmbd**(1 - 2 * H)
    
    # Normalize the spectral density
    fspec /= np.exp(np.mean(np.log(fspec)))

    return fspec