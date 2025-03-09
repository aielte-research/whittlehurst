import numpy as np
from scipy.special import gamma, zeta

def arfima(H: float, n: int):
    """
    Compute the spectral density of an ARFIMA(0, H - 0.5, 0) process.

    This function calculates the spectral density for a fractionally integrated process
    with differencing parameter d = H - 0.5. The density is computed for the first (n-1)/2 
    Fourier frequencies using the formula:

        fspec[i] = 2 * π * |2 * sin(dpl[i] / 2)|^(-2*d)

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
    d = H - 0.5
    nhalfm = int((n - 1) / 2)
    dpl = 2 * np.pi * np.arange(1, nhalfm + 1) / n
    fspec = np.ones(nhalfm)
    for i in np.arange(0, nhalfm):
        fspec[i] = 2 * np.pi * np.abs(2 * np.sin(dpl[i] / 2))**(-2 * d)
    return fspec


def fGn(H: float, n: int):
    """
    Compute the spectral density for fractional Gaussian noise (fGn)
    using the computationally feasible expression from Shi et al. (2024).

    This function calculates the spectral density at the Fourier frequencies
    using the expression:
    
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
    # Exponent for the Hurwitz zeta function
    s = 2*H + 1

    # Compute CH (assuming sigma^2 = 1)
    CH = gamma(s) * np.sin(np.pi * H) / np.pi

    # Define Fourier frequencies (positive frequencies)
    dpl = np.arange(1, (n-1)//2 + 1) / n

    # Compute Hurwitz zeta terms: zeta(s, q) computes ζ(s, q)
    term1 = zeta(s, 1 - dpl)
    term2 = zeta(s, dpl)

    # Compute the spectral density using the computationally feasible expression
    
    fspec = CH * (1 - np.cos(2 * np.pi * dpl)) * (2*np.pi)**(-s) * (term1 + term2)

    # Normalize the spectrum (geometric mean normalization)
    norm = np.exp(2 * np.sum(np.log(fspec)) / n)
    fspec = fspec / norm

    return fspec

def fGn_paxson(H: float, n: int, K: int = 50):
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
    gammaH = 2 * H + 1
    CH = gamma(2 * H + 1) * np.sin(np.pi * H) / (2 * np.pi)

    # Define Fourier frequencies
    nhalfm = (n - 1) // 2
    dpl = 2 * np.pi * np.arange(1, nhalfm + 1) / n  # shape: (nhalfm,)

    # Paxson's approximation: term1
    term1 = dpl**(-gammaH)

    # term2: sum_{j=1}^K b(j,λ)
    j_vals = np.arange(1, K + 1).reshape(-1, 1)  # shape: (K, 1)
    lam = dpl.reshape(1, -1)  # shape: (1, nhalfm)
    term2 = (2 * np.pi * j_vals + lam)**(-gammaH) + (2 * np.pi * j_vals -
                                                     lam)**(-gammaH)
    sum_term2 = np.sum(term2, axis=0)

    # Correction term: a(K,λ)
    def a_term(k, lam_val):
        return 1 / (4 * np.pi * H) * ((2 * np.pi * k + lam_val)**(1 - gammaH) +
                                      (2 * np.pi * k - lam_val)**(1 - gammaH))

    aK = a_term(K, dpl)
    aKp1 = a_term(K + 1, dpl)
    correction = 0.5 * (aK + aKp1)

    # Combine terms to compute the spectral density
    fspec = 2 * CH * (1 - np.cos(dpl)) * (term1 + sum_term2 + correction)

    # Normalize the spectrum
    norm = np.exp(2 * np.sum(np.log(fspec)) / n)
    fspec = fspec / norm

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
    gammaH = 2 * H + 1
    CH = gamma(2 * H + 1) * np.sin(np.pi * H) / (2 * np.pi)

    # Define Fourier frequencies (exclude zero frequency)
    nhalfm = (n - 1) // 2
    dpl = 2 * np.pi * np.arange(1, nhalfm + 1) / n  # shape: (nhalfm,)

    # Create an array of k values from -K to K
    k_vals = np.arange(-K, K + 1).reshape(-1, 1)  # shape: (2K+1, 1)

    # Reshape the frequency array for broadcasting
    lam = dpl.reshape(1, -1)  # shape: (1, nhalfm)

    # Compute the truncated sum over k for each Fourier frequency
    summation = np.sum(np.abs(2 * np.pi * k_vals + lam) ** (-gammaH), axis=0)

    # Compute the approximate spectral density
    fspec = 2 * CH * (1 - np.cos(dpl)) * summation

    # Normalize the spectrum
    norm = np.exp(2 * np.sum(np.log(fspec)) / n)
    fspec = fspec / norm

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
    nhalfm = (n - 1) // 2
    dpl = 2 * np.pi * np.arange(1, nhalfm + 1) / n  # Fourier frequencies

    # Taylor-series approximation for the spectral density as λ → 0
    fspec_taylor = CH * dpl**(1 - 2 * H)

    # Normalize the spectral density (following the normalization in other methods)
    norm = np.exp(2 * np.sum(np.log(fspec_taylor)) / n)
    fspec_taylor /= norm

    return fspec_taylor