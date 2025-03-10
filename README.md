![Python 3x](https://img.shields.io/badge/python-3.x-blue.svg)
[![pypi](https://img.shields.io/pypi/v/whittlehurst.svg)](https://pypi.org/project/whittlehurst/)

## Overview
This module implements Whittle's likelihood estimation method for determining the Hurst exponent of a time series.
The method fits the theoretical spectral density to the periodogram computed from the time series realization.
This implementation includes spectral density approximations for fractional Gaussian noise (increments of fractional Brownian motion) and ARFIMA processes.

The Hurst exponent ($H$) controls the roughness, self-similarity, and long-range dependence of fBm paths:

* $H\in(0,0.5):~$ anti-persistent (mean-reverting) behavior. 
* $H\in(0.5,1):~$ persistent behavior.
* $H=0.5:~ \mathrm{fBm}(H)$ is the Brownian motion.
* $H\rightarrow 0:~ \mathrm{fBm}(H)\rightarrow$ White noise.
* $H\rightarrow 1:~ \mathrm{fBm}(H)\rightarrow$ Linear trend.

## Features
* Spectral density options:
  - **`fGn`**
  - **`arfima`**
  - `fGn_paxson`
  - `fGn_truncation`
  - `fGn_taylor`
* A flexible interface that supports custom spectral density callback functions.
* Good performance both in terms of speed and accuracy.
* Included generators for fBm and ARFIMA.

## Installation
```
pip install whittlehurst
```

## Usage
### fBm and fGn
```python
import numpy as np
from whittlehurst import whittle, fbm

# Original Hurst value to test with
H=0.42

# Generate an fBm realization
fBm_seq = fbm(H=H, n=10000)

# Calculate the increments (the estimator works with the fGn spectrum)
fGn_seq = np.diff(fBm_seq)

# Estimate the Hurst exponent
H_est = whittle(fGn_seq)

print(f"Original H: {H:0.04f}, estimated H: {H_est:0.04f}")
```

### ARFIMA
```python
import numpy as np
from whittlehurst import whittle, arfima

# Original Hurst value to test with
H=0.42

# Generate a realization of an ARFIMA(0, H - 0.5, 0) process.
arfima_seq = arfima(H=H, n=10000)

# No need to take the increments here
# Estimate the "Hurst exponent" using the ARFIMA spectrum
H_est = whittle(arfima_seq, spectrum="arfima")

print(f"Original H: {H:0.04f}, estimated H: {H_est:0.04f}")
```

## Performance
### Compared to other methods
Our Whittle-based estimator offers a compelling alternative to traditional approaches for estimating the Hurst exponent. In particular, we compare it with:

- **R/S Method:** Implemented in the [hurst](https://github.com/Mottl/hurst) package, this method has been widely used for estimating $H$.

- **Higuchi's Method:** Available through the [antropy](https://github.com/raphaelvallat/antropy) package, it performs quite well especially for smaller $H$ values, but its performance drops when $H\rightarrow 1$.

- **Variogram:** Our variogram implementation of order $p = 1$ (madogram) accessible as `from whittlehurst import variogram`.

![RMSE by Sequence Length](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/fBm_estimators/png/fBm_Hurst_RMSE.png?raw=true "RMSE by Sequence Length")

Inference times represent the computation time per input sequence, and were calculated as: $t = w\cdot T/k$, where $k=100000$ is the number of sequences, $w=42$ is the number of workers (processing threads), and $T$ is the total elapsed time. Single-thread performance is likely superior, the results are mainly comparative. 

![Compute Time](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/fBm_estimators/png/fBm_Hurst_calc_times.png?raw=true  "Compute Time")

The following results were calculated on $100000$ fBm realizations of length $n=1600$.

![Local RMSE at n=1600](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/fBm_estimators/png/fBm_Hurst_01600_RMSE.png?raw=true  "Local RMSE")

![Scatter Plot](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/fBm_estimators/png/fBm_Hurst_01600_scatter_grid.png?raw=true "Scatter Plot")

### fGn spectral density approximations
The fGn spectral density calculations recommended by Shi et al. are accessible within our package:
- **` fGn `**: The default recommended spectral model. It relies on the gamma function and the Hurwitz zeta function $\zeta(s,q)=\sum_{j=0}^{\infty}(j+q)^{-s}$ from [scipy](https://scipy.org/). Terms independent from $H$ or $\lambda$ are omitted, as they are not required for minimizing the Whittle objective. With $s=2H+1$:

  $g(\lambda,H) = \Gamma(s) \sin(\pi H) (1-\cos(\lambda))(2\pi)^{-s}\left[ \zeta\!\left(s, 1-\frac{\lambda}{2\pi}\right) + \zeta\!\left(s, \frac{\lambda}{2\pi}\right) \right].$
- ` fGn_Paxson `: Uses Paxson's approximation with a configurable parameter `K=50`.
- ` fGn_truncation `: Approximates the infinite series by a configurable truncation `K=200`.
- ` fGn_Taylor `: Uses a Taylor series expansion to approximate the spectral density at near-zero frequency.

![RMSE by Sequence Length](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/fBm_Whittle_variants/png/fBm_Hurst_RMSE.png?raw=true "RMSE by Sequence Length")

![Compute Time](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/fBm_Whittle_variants/png/fBm_Hurst_calc_times.png?raw=true  "Compute Time")

The following results were calculated on $100000$ fBm realizations of length $n=1600$.

![Local RMSE at n=1600](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/fBm_Whittle_variants/png/fBm_Hurst_01600_RMSE.png?raw=true  "Local RMSE")

![Scatter Plot](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/fBm_Whittle_variants/png/fBm_Hurst_01600_scatter_grid.png?raw=true "Scatter Plot")

### ARFIMA
For the $\text{ARFIMA}(0, H - 0.5, 0)$ process, the spectral density calculation is simpler. With terms independent from $H$ or $\lambda$ omitted, we use:

$g(\lambda,H) = (2\cdot\sin(\lambda/2))^{1 - 2H}$

![ARFIMA Local RMSE](https://github.com/aielte-research/whittlehurst/blob/main/tests/plots/arfima/png/ARFIMA_Hurst_local_RMSE.png?raw=true "ARFIMA Local RMSE")

## References
* The initial implementation of Whittle's method was adapted from:  
  
  https://github.com/JFBazille/ICode/blob/master/ICode/estimators/whittle.py

* For further details on spectral density models for fractional Gaussian noise, refer to:

  **Shuping Shi, Jun Yu, and Chen Zhang**. *Fractional gaussian noise: Spectral density and estimation methods*. Journal of Time Series Analysis, 2024. https://onlinelibrary.wiley.com/doi/full/10.1111/jtsa.12750

## License
This project is licensed under the MIT License (c) 2025 Bálint Csanády, aielte-research. See the LICENSE file for details.