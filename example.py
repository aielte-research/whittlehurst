import numpy as np
#######
# fBm #
#######
from whittlehurst import whittle, fbm

# Original Hurst value to test with
H=0.42

# Generate an fBm realization
fBm_seq = fbm(H=H, n=10000)

# Calculate the increments, because the estimator works with the fGn spectrum
fGn_seq = np.diff(fBm_seq)

# Estimate the Hurst exponent
H_est = whittle(fGn_seq)

print(f"fBm({H:0.04f}) estimated H: {H_est:0.04f}")

##########
# ARFIMA #
##########

from whittlehurst import arfima

# Original Hurst value to test with
H=0.42

# Generate an ARFIMA(0, H - 0.5, 0) realization
arfima_seq = arfima(H=H, n=10000)

# No need to take the increments here
# Estimate the "Hurst exponent"
H_est = whittle(arfima_seq, spectrum="arfima")

print(f"ARFIMA(0, {H - 0.5:0.04f}, 0) estimated: {H_est - 0.5:0.04f}")

############
# fBm TDML #
############
from whittlehurst import tdml

# Original Hurst value to test with
H=0.42

# Generate an fBm realization
fBm_seq = fbm(H=H, n=10000)

# Calculate the increments
fGn_seq = np.diff(fBm_seq)

# Estimate the Hurst exponent
H_est = tdml(fGn_seq)

print(f"fBm({H:0.04f}), tdml estimated H: {H_est:0.04f}")