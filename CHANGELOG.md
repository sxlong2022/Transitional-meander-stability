# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-09-02 (ESPL Major Revision Release)

This major release coincides with the comprehensive revision submitted to *Earth Surface Processes and Landforms* (ESPL). It provides the full, production-grade 2D shallow-water--Exner morphodynamic stability solver, Chebyshev spectral collocation discretization, multi-decadal satellite processing pipeline, and reproducible scripts for all manuscript figures and tables.

### Added
- **Planar 2D SWE--Exner Generalized Eigenvalue Solver (`src/stability/solve_bar_stability.py`)**:
  - Implements the complete linearized, depth-averaged two-dimensional shallow-water equations coupled with the Exner sediment continuity equation.
  - Arbitrary-order Chebyshev collocation spectral discretization with boundary conditions: lateral velocity $\hat{v}(\pm 1/2) = 0$ and zero lateral bedload transport $\partial_n \hat{z}_b(\pm 1/2) = 0$.
  - Exact secondary helical flow curvature perturbation operator $\nu = H/R$.
  - Transverse mode competition solver ($m = 1, 2, 3, 4$) with physical bed-mode filtering.
  - Golden-section bounded scalar optimization for locating the most amplified longitudinal wavenumber $k_{\max}$ and peak exponential growth rate $\sigma_{r,\max}$.
- **Comprehensive Table & Figure Reproduction Pipeline**:
  - `compute_all_revision_tables.py`: One-command generation of Tables 3, 4, S1, S2, S3, S4, S5.
  - `generate_si_uncertainty_csvs.py`: Generation of Tables S6, S7, S8, S9.
  - Complete standalone publication scripts in `publication_figures/` reproducing Figures 1 through 10.
- **26-Year Satellite Planform Analysis Suite (`src/morphodynamics/`, `src/spectral/`)**:
  - Continuous Landsat/Sentinel water-occurrence composite processing for 2000–2025.
  - Primary continuous channel trunk extraction ($135\text{--}155\,\mathrm{km}$) via undirected Dijkstra graph traversal.
  - Spectral FFT, single-sided PSD, autocorrelation $e$-folding length, and cross-correlation analysis.
- **Interactive Demos (`examples/`)**:
  - `example_01_solve_bar_stability.py`: Single-case 2D stability calculation and curvature modulation test.
  - `example_02_reproduce_all_tables.py`: Fast validation and full reproduction of all paper tables.
  - `example_03_plot_stability_curves.py`: Generation of wavenumber dispersion curves and celerity spectra.

### Changed
- Unified eigenvalue convention to $\exp(iks + \sigma t)$, where $\sigma_r = \mathrm{Re}(\sigma)$ represents the exponential temporal growth rate and $c_{\mathrm{migr}} = -\sigma_i / k$ represents downstream migration celerity.
- Updated literature benchmarks against Colombini et al. (1987) with relative error $< 0.3\%$.
- Unified production Chebyshev collocation grid resolution to $N_{\mathrm{cheb}} = 36$ across all tables and figures.

### Repository Links
- **GitHub**: [https://github.com/sxlong2022/Transitional-meander-stability](https://github.com/sxlong2022/Transitional-meander-stability)
- **Zenodo DOI**: [10.5281/zenodo.20361365](https://doi.org/10.5281/zenodo.20361365)
