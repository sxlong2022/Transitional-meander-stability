# Results & Versioned Datasets

This directory contains the versioned data tables, stability calculation results, and remote sensing time series used in the manuscript:
> **"Planar 2D shallow-water--Exner stability diagnosis of bar formation in the transitional Lower Yellow River"**  
> *Earth Surface Processes and Landforms* (ESPL).

---

## Summary of Tables

All tables strictly follow the sequential numbering in the revised ESPL manuscript and Supporting Information:

| File Name | Manuscript / SI Reference | Description | Rows / Coverage |
|---|---|---|---|
| `temporal_stability_2d.csv` | **Table 3** (Main Text §4.1) | Multi-decadal linear stability results for 14 post-dam years (2000–2019) | 14 years |
| `spatial_2016_stability_2d.csv` | **Table 4** (Main Text §4.2) | Along-stream linear stability results for 26 cross-sections in 2016 | 26 sections (118 km) |
| `table_s1_sediment_sensitivity.csv` | **Table S1** (SI §S1.4) | Sensitivity of peak growth rate to sediment transport closure parameters ($\theta_c, b, \Gamma$) | 15 sensitivity cases |
| `table_s2_convergence.csv` | **Table S2** (SI §S2.2) | Chebyshev polynomial degree spectral convergence analysis ($N = 16$ to $64$) | 7 polynomial orders |
| `table_s3_benchmark.csv` | **Table S3** (SI §S2.3) | Benchmark validation against Colombini et al. (1987) / Tubino et al. (1999) | 6 benchmark cases |
| `table_s4_curvature_sensitivity.csv` | **Table S4** (SI §S2.4) | Sensitivity of growth rate and curvature enhancement $E$ to satellite-measured curvature $R$ | 6 curvature scales |
| `table_s5_scenes.csv` | **Table S5** (SI §S6.2) | Landsat/Sentinel satellite scene metadata, flow conditioning, and width comparison | 11 representative years |
| `table_s6_spectral_sensitivity.csv` | **Table S6** (SI §S6.3) | Curvature smoothing, step size, tapering, and discharge-induced width perturbation sensitivity | 15 sensitivity cases |
| `table_s7_width_gradient.csv` | **Table S7** (SI §S6.4) | Multi-decadal along-stream channel width gradient distributions ($|\mathrm{d}B/\mathrm{d}s|$) | 11 annual distributions |
| `table_s8_multi_beta.csv` | **Table S8** (SI §S7) | Comprehensive 3D parameter sweep ($\beta \in [15, 250]$, $C_f$, $\mathrm{Fr}$) | 48 combinations |
| `table_s9_mode_competition.csv` | **Table S9** (SI §S8) | Transverse mode competition ($m = 1, 2, 3, 4$) across aspect ratios | 5 reach/epoch stages |
| `trunk_validation_metrics.csv` | **SI §S6.2** (Figure S1) | Quantitative spatial overlap and topological validation metrics for braided years | 4 braided years |

> **Note on Supporting Figures**:
> - **Figure S1** (SI Section S6.2): Algorithmic validation of automated trunk extraction against manual reference in braided reaches, generated via `publication_figures/plot_fig_s01_trunk_validation.py` (`fig_s01_trunk_validation.pdf`).
> - **Figure S2** (SI Section S6.4): Multi-decadal along-stream spatial profiles and reach coverage CDF of local width gradient $|\mathrm{d}B/\mathrm{d}s|$, generated via `publication_figures/plot_fig_s02_width_gradient.py` (`fig_s02_width_gradient.pdf`).
---

## Core Hydrometric & Morphologic Time Series

- **`hydraulic_params_timeseries.csv`**: Annual reach-averaged hydrometric parameters (2000–2021) at Gaocun station during bankfull/flood flow:
  - `year`: Year of observation
  - `beta`: Aspect ratio $B/H$
  - `Cf_energy`: Energy-slope based friction coefficient $C_f = g H S / U^2$
  - `Fr`: Froude number $\mathrm{Fr} = U / \sqrt{g H}$
  - `Shields`: Dimensionless Shields stress $\theta = \tau_b / [(\rho_s - \rho_w) g D_{50}]$
  - `B_m`, `H_m`, `U_ms`: Dimensional channel width, depth, and flow velocity

- **`hydraulic_params_spatial_2016.csv`**: Reach-scale cross-sectional hydraulic data for 26 surveyed cross-sections (2016):
  - `dam_km`: Distance from Xiaolangdi Dam (km, 303 to 421 km)
  - `beta(s)`, `Fr(s)`, `Cf(s)`, `Shields(s)`: Local reach parameters

- **`sigma_width_summary.csv`**: 26-year summary of dimensionless channel width gradient $\sigma_{\mathrm{width}} = \frac{1}{B}\frac{\mathrm{d}B}{\mathrm{d}s}$.

---

## Subdirectories

- **`profiles/`**: 26 annual CSV files containing channel width $B(s)$ and curvature $C(s)$ along all link segments extracted by RivGraph.
- **`trunks/`**: 26 annual CSV files containing primary continuous channel trunks ($135\text{--}155\,\mathrm{km}$) extracted via graph-theoretic shortest path routing.
- **`spectral/`**: Power spectral density (PSD) curves and `spectral_summary.csv` containing dominant wavelengths ($\lambda_B, \lambda_C$), cross-correlation coefficients, and $e$-folding persistence lengths.
- **`phase_diagram/`**: Dense numerical sweep data `phase_diagram_omega_i.csv` ($50 \times 50$ grid) used to render the 3D $(C_f, \mathrm{Fr}, \beta)$ stability phase diagram (Figure 10).
- **`beta_sweep/`**: Intermediate convergence, profile, and aspect-ratio sweep tables.
