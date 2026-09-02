# Planar 2D SWE--Exner Stability of Alternate Bars in Transitional Rivers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![DOI: 10.5281/zenodo.20361365](https://zenodo.org/badge/DOI/10.5281/zenodo.20361365.svg)](https://doi.org/10.5281/zenodo.20361365)
[![Status: Peer-Reviewed Revision](https://img.shields.io/badge/Manuscript-ESPL%20Revision-brightgreen.svg)](https://onlinelibrary.wiley.com/journal/10969837)

Official computational repository and reproducible data archive for the manuscript:
> **"Planar 2D SWE--Exner stability of alternate bars in the transitional Lower Yellow River"**  
> *Earth Surface Processes and Landforms* (ESPL).

---

## 📌 Overview

Traditional morphodynamic river stability theories (e.g., Colombini et al., 1987; Tubino et al., 1999) have been extensively calibrated in narrow laboratory flumes ($\beta = B/H \sim 10\text{--}50$). However, their quantitative validity remains unverified in wide, high-energy transitional rivers ($\beta \sim 100\text{--}300$). 

This repository provides an end-to-end computational framework combining:
1. **Planar 2D Shallow-Water Equations coupled with Exner sediment continuity (2D SWE--Exner)** solved as a generalized eigenvalue problem via arbitrary-order Chebyshev spectral collocation.
2. **Exact curvature perturbation modeling** quantifying secondary helical flow effects ($\nu = H/R$) without ad-hoc empirical assumptions.
3. **Multi-mode transverse competition** ($m = 1, 2, 3, 4$) across varying width-to-depth ratios ($\beta = 15\text{--}300$).
4. **26-Year satellite planform analysis (2000–2025)** combining Landsat/Sentinel water-occurrence composites, automated RivGraph link extraction, and graph-theoretic primary channel trunk tracing across the 118-km Gaocun–Sunkou reach of the Lower Yellow River.

---

## 📐 Mathematical Formulation

The linear morphodynamic stability is formulated on a depth-averaged, dimensionless coordinate system $(s, n, t)$ scaled by channel width $B$, undisturbed depth $H$, and mean velocity $U$:

$$
\mathbf{L} \hat{\mathbf{q}} = -\sigma \mathbf{M} \hat{\mathbf{q}}
$$

where $\hat{\mathbf{q}} = [\hat{u}, \hat{v}, \hat{h}, \hat{z}_b]^T$ represents the transverse eigenmode vector, and perturbations follow $\mathbf{q}' = \hat{\mathbf{q}}(n) \exp(i k s + \sigma t)$:
- $\sigma_r = \mathrm{Re}(\sigma)$ is the **exponential temporal growth rate** ($\sigma_r > 0$ indicates linear instability).
- $c_{\mathrm{migr}} = -\mathrm{Im}(\sigma) / k$ is the **downstream bar migration celerity**.
- $k = 2\pi B / \lambda$ is the **dimensionless longitudinal wavenumber**.

### Dimensionless Governing Equations

**1. Flow Continuity:**

$$
i k \hat{u} + \frac{\partial \hat{v}}{\partial n} + i k \hat{h} = -\sigma \hat{h}
$$

**2. Streamwise Momentum:**

$$
\left(i k + 2 \beta C_f\right)\hat{u} + \left(\frac{i k}{\mathrm{Fr}^2} - \beta C_f\right)\hat{h} + \frac{i k}{\mathrm{Fr}^2}\hat{z}_b = -\sigma \hat{u}
$$

**3. Transverse Momentum:**

$$
\left(i k + \beta C_f\right)\hat{v} + \frac{1}{\mathrm{Fr}^2}\frac{\partial \hat{h}}{\partial n} + \frac{1}{\mathrm{Fr}^2}\frac{\partial \hat{z}_b}{\partial n} - \nu \hat{u} = -\sigma \hat{v}
$$

**4. Exner Sediment Continuity:**

$$
i k \gamma_u \hat{u} + \frac{\partial \hat{v}}{\partial n} - f_{\mathrm{sec}} \nu \frac{\partial \hat{u}}{\partial n} - \Gamma_\beta \frac{\partial^2 \hat{z}_b}{\partial n^2} = -\sigma \hat{z}_b
$$

**5. Boundary Conditions:**

At lateral walls ($n = \pm 1/2$), impermeable banks enforce zero lateral flow and zero lateral bedload transport:

$$
\hat{v}\left(\pm \frac{1}{2}\right) = 0, \quad \left.\frac{\partial \hat{z}_b}{\partial n}\right|_{n = \pm 1/2} = 0
$$

---

## 🗂️ Repository Structure

```
.
├── LICENSE                             # MIT License
├── README.md                           # Repository documentation (this file)
├── CHANGELOG.md                        # Version history & release notes
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment definition
├── pyproject.toml                      # Packaging metadata
├── src/                                # Core computational library
│   ├── config.py                       # Physical constants and reach parameters
│   ├── stability/                      # 2D SWE--Exner eigenvalue solver
│   │   ├── solve_bar_stability.py      # Chebyshev spectral generalized eigenvalue solver
│   │   ├── compute_all_revision_tables.py # Batch script reproducing Tables 3, 4, S1-S5
│   │   ├── generate_si_uncertainty_csvs.py# Batch script reproducing Tables S6-S9
│   │   ├── run_phase_diagram.py        # 3D (Cf, Fr, beta) parameter space sweep
│   │   └── os_operator_curved.py       # Curvature perturbation operator
│   ├── morphodynamics/                 # Remote sensing and channel network extraction
│   │   ├── extract_main_channel.py     # Graph-theoretic Dijkstra channel trunk router
│   │   ├── rivgraph_link_profiles.py   # Link-wise B(s) and C(s) profile extraction
│   │   └── run_rivgraph_batch.py       # 26-year batch execution wrapper
│   ├── spectral/                       # Spatial series and spectral analysis
│   │   └── analyze_profiles.py         # FFT, PSD, e-folding length, and cross-correlation
│   ├── bar_modes/                      # Bar mode predictors
│   │   └── predict_bars.py             # Crosato-Mosselman mode predictor
│   ├── diagnostics/                    # Unified stability diagnostic API
│   │   └── diagnose_stability.py       # Single and batch point diagnostic routines
│   └── data/                           # Hydrometric data compilation
│       └── compile_hydraulic_params.py # Compiles annual and spatial hydraulic parameters
├── results/                            # Pre-computed datasets and tables (100% reproducible)
│   ├── README.md                       # Data dictionary for all CSV tables
│   ├── temporal_stability_2d.csv       # Table 3 (Main text)
│   ├── spatial_2016_stability_2d.csv   # Table 4 (Main text)
│   ├── table_s1_convergence.csv        # Table S1 (Chebyshev convergence)
│   ├── table_s2_benchmark.csv          # Table S2 (Colombini 1987 benchmark)
│   ├── table_s3_sediment_sensitivity.csv # Table S3 (Sediment exponent sensitivity)
│   ├── table_s4_multi_beta.csv         # Table S4 (3D parameter space exploration)
│   ├── table_s5_mode_competition.csv   # Table S5 (Transverse mode competition)
│   ├── table_s6_scenes.csv             # Table S6 (Satellite scenes metadata)
│   ├── table_s7_spectral_sensitivity.csv # Table S7 (Curvature smoothing sensitivity)
│   ├── table_s8_width_gradient.csv     # Table S8 (Width gradient distributions)
│   ├── table_s9_curvature_sensitivity.csv # Table S9 (Satellite per-bend curvature)
│   ├── hydraulic_params_timeseries.csv # Annual reach-averaged hydraulics (2000-2021)
│   ├── hydraulic_params_spatial_2016.csv # Cross-sectional hydraulics along 2016 reach
│   ├── profiles/                       # 26-year RivGraph link profiles
│   ├── trunks/                         # 26-year primary continuous channel trunks
│   ├── spectral/                       # Spectral summary and annual PSD curves
│   └── phase_diagram/                  # Dense grid sweep data for Figure 10
├── publication_figures/                # Standalone plotting scripts for Figures 1–10
│   ├── figure_utils.py                 # Styling setup (Okabe-Ito, 600 DPI, STIX math)
│   ├── plot_fig01_study_area.py        # Figure 1: Study area & river reach
│   ├── plot_fig02_pipeline.py          # Figure 2: Integrated methodology workflow
│   ├── plot_fig03_hydraulic_timeseries.py # Figure 3: Hydraulic timeseries
│   ├── plot_fig04_spatial_hydraulic.py # Figure 4: Spatial hydraulic profiles
│   ├── plot_fig05_temporal_diagnostics.py # Figure 5: Temporal stability curves
│   ├── plot_fig06_alpha_sweep.py       # Figure 6: Wavenumber spectrum comparison
│   ├── plot_fig07_spatial_diagnostics.py # Figure 7: Spatial stability profiles
│   ├── plot_fig08_spectral.py          # Figure 8: Spectral characteristics of B(s) and C(s)
│   ├── plot_fig09_nubeta_collapse.py   # Figure 9: Model-derived curvature scaling
│   ├── plot_fig10_phase_diagram.py     # Figure 10: 3D stability phase diagram
│   └── output/                         # Pre-compiled high-res vector figures (PDF / PNG)
└── examples/                           # Interactive runnable example scripts
    ├── example_01_solve_bar_stability.py # Single cross-section eigenvalue solve demo
    ├── example_02_reproduce_all_tables.py # One-command table verification & reproduction
    └── example_03_plot_stability_curves.py # Publication-quality dispersion curve generator
```

---

## ⚡ Installation & Setup

### Option A: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/sxlong2022/Transitional-meander-stability.git
cd Transitional-meander-stability

# Create and activate the conda environment
conda env create -f environment.yml
conda activate transitional-meander
```

### Option B: Using Pip

```bash
# Clone the repository
git clone https://github.com/sxlong2022/Transitional-meander-stability.git
cd Transitional-meander-stability

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start Examples

The `examples/` directory contains self-contained, commented scripts demonstrating key functionalities:

### 1. Solve Single Stability State
Solve the 2D SWE--Exner eigenvalue problem for representative field conditions ($\beta = 130, C_f = 0.002, \mathrm{Fr} = 0.25, \theta = 0.55$) and calculate the curvature modulation factor $E$:
```bash
python examples/example_01_solve_bar_stability.py
```
*Expected Output:*
```text
Straight Channel Results (Mode m=1, Alternate Bars):
  Preferred Wavenumber k_max      = 3.890
  Peak Growth Rate sigma_r_max    = 0.2355 (exponential growth rate)
  Dimensionless Celerity c_migr   = 0.5823 (downstream bar migration)
  Dimensional Wavelength lambda_m = 726.8 m (~ 1.6 channel widths)
  Curvature Modulation Factor E   = +0.013% (|E| <= 0.09% < 0.1%)
```

### 2. Verify and Reproduce All Tables
Verify that all 11 manuscript and SI data tables in `results/` are present and consistent:
```bash
python examples/example_02_reproduce_all_tables.py
```
To recompute the convergence and benchmark tables from scratch:
```bash
python examples/example_02_reproduce_all_tables.py --fast
```
To recompute all tables from scratch (~3–4 minutes):
```bash
python examples/example_02_reproduce_all_tables.py --recompute
```

### 3. Plot Wavenumber Dispersion Curves
Compute continuous dispersion curves $\sigma_r(k)$ and migration celerity $c_{\mathrm{migr}}(k)$ comparing early wide channel states (year 2000) with recent narrowed states (year 2019):
```bash
python examples/example_03_plot_stability_curves.py
```
The resulting vector graphic is exported to `publication_figures/output/example_dispersion_comparison.pdf`.

---

## 📊 Reproducing Manuscript Figures

All manuscript figures (Figures 1–10) can be independently generated using the standalone scripts in `publication_figures/`:

```bash
# Figure 1: Study area map and Google Earth overview
python publication_figures/plot_fig01_study_area.py

# Figure 2: Integrated diagnostic framework pipeline flowchart
python publication_figures/plot_fig02_pipeline.py

# Figure 3: Multi-decadal hydraulic parameters timeseries (2000-2021)
python publication_figures/plot_fig03_hydraulic_timeseries.py

# Figure 4: Spatial hydraulic profiles along the 2016 reach
python publication_figures/plot_fig04_spatial_hydraulic.py

# Figure 5: Temporal stability growth rates and dimensional wavelengths
python publication_figures/plot_fig05_temporal_diagnostics.py

# Figure 6: Planar 2D dispersion curves for early vs recent years
python publication_figures/plot_fig06_alpha_sweep.py

# Figure 7: Along-stream spatial stability diagnostics for 2016 reach
python publication_figures/plot_fig07_spatial_diagnostics.py

# Figure 8: Spectral analysis and autocorrelation of B(s) and C(s) profiles
python publication_figures/plot_fig08_spectral.py

# Figure 9: Model-derived curvature scaling and bend parameter sensitivity
python publication_figures/plot_fig09_nubeta_collapse.py

# Figure 10: 3D stability phase diagram across (Cf, Fr, beta) space
python publication_figures/plot_fig10_phase_diagram.py
```
All generated figures are saved to `publication_figures/output/` in both 600 DPI vector PDF and 300 DPI preview PNG formats using the Okabe-Ito colorblind-safe palette.

---

## 🔬 Benchmark Validation

The Chebyshev collocation solver is validated against the classical benchmark of **Colombini, Seminara, and Tubino (1987)** (Table S2):
- At reference parameters ($C_{f0} = 0.003, \beta = 20.0, d_s = 0.01$):
  - **Literature Benchmark:** $k_{\max} = 0.360$, $\omega_{i,\max} = 0.0520$, $c_{\mathrm{migr}} = 0.812$
  - **Our 2D SWE--Exner Solver ($N_{\mathrm{cheb}} = 36$):** $k_{\max} = 0.361$, $\sigma_{r,\max} = 0.0519$, $c_{\mathrm{migr}} = 0.813$
  - **Relative Discrepancy:** $< 0.3\%$, demonstrating strict numerical convergence and formulation equivalence.

---

## 📜 Citation

If you use this code, data, or theoretical framework in your research, please cite:

```bibtex
@article{Song2026ESPL,
  author    = {Song, Xiaolong and Xu, Haijue and Bai, Yuchuan},
  title     = {Planar {2D} {SWE}--{Exner} stability of alternate bars in the transitional {Lower} {Yellow} {River}},
  journal   = {Earth Surface Processes and Landforms},
  year      = {2026},
  doi       = {10.1002/esp.XXXX},
  note      = {In revision}
}
```

Permanent archival record:
```bibtex
@misc{Song2026Zenodo,
  author    = {Song, Xiaolong and Xu, Haijue and Bai, Yuchuan},
  title     = {Planar {2D} {SWE}--{Exner} Stability of Alternate Bars in Transitional Rivers: Software and Data Archive},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.20361365},
  url       = {https://doi.org/10.5281/zenodo.20361365}
}
```

---

## ⚖️ License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact

For questions, issues, or collaborations regarding the theoretical solver or datasets, please open an issue on GitHub or contact:
- **Xiaolong Song**: [xlsong@tju.edu.cn](mailto:xlsong@tju.edu.cn)
- State Key Laboratory of Hydraulic Engineering Intelligent Construction and Operation, Tianjin University, Tianjin 300354, China
- Institute for Sedimentation on River and Coastal Engineering, Tianjin University, Tianjin 300354, China
