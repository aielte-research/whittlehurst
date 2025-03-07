import warnings
import numpy as np
import cmath
from scipy.fftpack import fft

def fbm(H, n=1600, length = 1):
    """
        One dimensional fraction Brownian motion generator object through different methods.
        The output is a realization of a fBm (W_t) where t in [0, T] using 'n' equally spaced grid points.
        The object inputs:
        @arg hurst: the Hurst exponent of the fBm, [0, 1]
        @arg n: the length of data points
        @arg method: define the generator process to get the fBm path, default 'Cholesky'
        @arg T: the interval endpoint of 't', default [0, T=1]
        Input:
            @param H: Hurst exponent in [0, 1]
            @param n: number of grid points
            @param T: the endpoint of the time interval, default T=1
        Output:
            W_t the fBm path for t on [0, T]
        Example:
            fbm = FBMGenerators(
                hurst = 0.7, n=10000, method='fft', T=1)
            path = fbm.fbm()
        Reference:
            Kroese, D. P., & Botev, Z. I. (2015). Spatial Process Simulation.
            In Stochastic Geometry, Spatial Statistics and Random Fields(pp. 369-404)
            Springer International Publishing, DOI: 10.1007/978-3-319-10064-7_12
            https://sci-hub.se/10.1007/978-3-319-10064-7_12
    """
    if H < 0 or H > 1:
        return warnings.warn('hurst parameter must be between 0 and 1')

    r = np.zeros(n+1)
    for i in range(n+1):
        if i == 0:
            r[0] = 1
        else:
            r[i] = 1 / 2 * ((i + 1)**(2*H) - 2*i**(2*H) + (i - 1)**(2*H))
    r = np.concatenate([r, r[::-1][1:-1]])
    lmbd = np.real(fft(r) / (2*n))
    sqrt = [cmath.sqrt(x) for x in lmbd]
    W = fft(sqrt * (np.random.normal(size=2*n) + np.random.normal(size=2*n) * complex(0, 1)))
    W = n**(-H) * np.cumsum([0, *np.real(W[1:(n + 1)])])

    # rescale the for the final T
    W = (length ** H) * W
    return W