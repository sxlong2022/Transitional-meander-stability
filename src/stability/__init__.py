"""Planar 2D SWE--Exner morphodynamic linear stability solver subpackage.

Provides Chebyshev spectral collocation discretization, 2D SWE--Exner generalized eigenvalue solvers,
exact curvature perturbation operators, and batch table reproduction routines.
"""

from .solve_bar_stability import (  # noqa: F401
    chebyshev_collocation,
    solve_bar_stability,
    find_most_amplified_mode,
    compute_curvature_modulation_exact,
    solve_modal_competition,
)
