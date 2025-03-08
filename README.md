### README

## Overview

This module provides an implementation of Whittle's likelihood method to estimate the Hurst exponent of a time series. The method fits a theoretical spectral density model to the periodogram of a time series. This implementation supports multiple spectral density approximations for fractional Gaussian noise (increments of fractional Brownian motion) and ARFIMA processes.

## Features

* Estimate the Hurst exponent (H) by minimizing the Whittle likelihood function.  
* Spectral density options:
  - **`fGn`**
  - **`arfima`**
  - `fGn_paxson`
  - `fGn_truncation`
  - `fGn_taylor`
* Flexible interface with an option for a custom spectral density callback.
* Included generators for fBm, and ARFIMA:

## Installation

```
pip install whittlehurst
```

## Usage

### fBM or fGn

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

# Generate an ARFIMA(0, H - 0.5, 0) realization
arfima_seq = arfima(H=H, n=10000)

# No need to take the increments here
# Estimate the "Hurst exponent"
H_est = whittle(arfima_seq, spectrum="arfima")

print(f"Original H: {H:0.04f}, estimated H: {H_est:0.04f}")
```

## Notes

* The default recommended spectral model is `fGn` which relies on Hurwitz's zeta function.
* `fGn_paxson`, `fGn_truncation`, `fGn_taylor` are experimental approximations of the fGn spectrum. 
* For models `fGn_paxson` and `fGn_truncation`, the parameter `K` is configurable (defaults: 50 and 200 respectively).  
* A custom spectral density function may be provided via the `spectrum_callback` parameter.

## References

The initial implementation of Whittle's method was based on:  
https://github.com/JFBazille/ICode/blob/master/ICode/estimators/whittle.py

For details on spectral density models for fractional Gaussian noise, refer to:  
https://onlinelibrary.wiley.com/doi/full/10.1111/jtsa.12750

## License

This project is licensed under the MIT License (c) 2025 Bálint Csanády, aielte-research. See the LICENSE file for details.

