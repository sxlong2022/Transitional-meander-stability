from __future__ import annotations
import numpy as np

# Simulate a simple OS bar stability diagnosis using the package.
# Note: For accurate results, use actual river hydraulic parameters.
from src.utils.diagnose_stability import diagnose_stability

def main():
    print("Running 1D linear stability diagnosis for Gaocun-Sunkou theoretical parameters.")
    
    # Representative parameters
    beta = 100.0  # Width-to-depth ratio
    Fr = 0.3      # Froude number
    Cf = 0.005    # Friction coefficient
    
    result = diagnose_stability(
        beta=beta,
        Fr=Fr,
        Cf=Cf,
        N_cheb=20,           # Number of Chebyshev collocation points
        nu_curvature=0.0,    # Straight channel (nu = H/R = 0)
        profile_mode="zs_turbulent"
    )
    print("\n--- Stability Diagnosis Results ---")
    print(f"Max amplification rate (c_i_max): {result.omega_i_max:.4f}")
    print(f"Critical wavenumber (alpha_crit): {result.alpha_crit:.4f}")
    print(f"Is channel unstable to alternate bars? {'Yes' if result.omega_i_max > 0 else 'No'}")

if __name__ == "__main__":
    main()
