# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-09-03 (ESPL Revision Polish & Supporting Information Monotonicity)

This release aligns the repository structure, data tables, and figures with the final revised Supporting Information and manuscript structure for ESPL:

### Added
- **Figure S1 Pipeline & Plotting Script (`publication_figures/plot_fig_s01_width_gradient.py`)**:
  - Adds standalone reproducible script generating Figure S1: 3-row vertically stacked along-channel local dimensionless channel width gradient ($|\mathrm{d}B/\mathrm{d}s|$) profiles and cumulative reach coverage distribution functions (CDFs).
  - High-resolution vector PDF (`publication_figures/output/fig_s01_width_gradient.pdf`) and 300 DPI preview PNG (`fig_s01_width_gradient.png`).
  - Plain-text in-situ threshold labels for weak ($|\mathrm{d}B/\mathrm{d}s| \le 0.10$) and moderate ($|\mathrm{d}B/\mathrm{d}s| \le 0.20$) variations.
- **Discharge-Induced Width Perturbation Sensitivity Data**:
  - Adds 4 systematic sensitivity test configurations (contraction $-25\%$, expansion $+25\%$, stage offset $\pm 50\,\mathrm{m}$, and random discharge noise $\pm 25\%$) to Table S6 (`table_s6_spectral_sensitivity.csv`), demonstrating that discharge-dependent width fluctuations alter the dominant wavelength by $< 1.0\%$.

### Changed
- **Strictly Monotonic Supporting Information Table Renumbering (Tables S1–S9)**:
  - Renamed and reordered all SI CSV tables in `results/` to establish exact 1-to-1 correspondence with the revised manuscript and Supporting Information:
    - `table_s1_sediment_sensitivity.csv` (Table S1, SI §S1.4: sediment transport closure sensitivity)
    - `table_s2_convergence.csv` (Table S2, SI §S2.2: Chebyshev spectral collocation convergence)
    - `table_s3_benchmark.csv` (Table S3, SI §S2.3: Colombini et al. 1987 literature benchmark)
    - `table_s4_curvature_sensitivity.csv` (Table S4, SI §S2.4: satellite-measured curvature sensitivity)
    - `table_s5_scenes.csv` (Table S5, SI §S6.2: Landsat/Sentinel scene statistics and flow conditioning)
    - `table_s6_spectral_sensitivity.csv` (Table S6, SI §S6.3: curvature smoothing and discharge width perturbations)
    - `table_s7_width_gradient.csv` (Table S7, SI §S6.4: multi-decadal channel width gradient distributions)
    - `table_s8_multi_beta.csv` (Table S8, SI §S7: 3D parameter space exploration)
    - `table_s9_mode_competition.csv` (Table S9, SI §S8: transverse mode competition)
- **Script Refactoring**:
  - Updated `src/stability/compute_all_revision_tables.py`, `src/stability/generate_si_uncertainty_csvs.py`, and `examples/example_02_reproduce_all_tables.py` to seamlessly output, verify, and document the new Table S1–S9 filenames.
  - Eliminated hardcoded drive letters in figure plotting scripts, ensuring cross-platform path portability across Windows, Linux, and macOS.

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
- **Zenodo DOI**: [10.5281/zenodo.22245579](https://doi.org/10.5281/zenodo.22245579)
