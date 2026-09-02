# Results & Versioned Datasets

This directory contains the versioned data tables, stability calculation results, and remote sensing time series used in the manuscript:
> **"Planar 2D SWE--Exner stability of alternate bars in the transitional Lower Yellow River"**  
> *Earth Surface Processes and Landforms* (ESPL).

---

## Summary of Tables

| File Name | Manuscript / SI Reference | Description | Rows / Coverage |
|---|---|---|---|
| `temporal_stability_2d.csv` | **Table 3** (Main Text) | Multi-decadal linear stability results for 14 post-dam years (2000–2019) | 14 years |
| `spatial_2016_stability_2d.csv` | **Table 4** (Main Text) | Along-stream linear stability results for 26 cross-sections in 2016 | 26 sections (118 km) |
| `table_s1_convergence.csv` | **Table S1** (SI Section S5.1) | Chebyshev polynomial degree convergence analysis ($N = 12$ to $48$) | 7 polynomial orders |
| `table_s2_benchmark.csv` | **Table S2** (SI Section S5.2) | Benchmark validation against Colombini et al. (1987) / Tubino et al. (1999) | 6 benchmark cases |
| `table_s3_sediment_sensitivity.csv` | **Table S3** (SI Section S1.3) | Sensitivity to sediment transport velocity exponent ($b = 1.5$ vs $2.5$) | 15 sensitivity cases |
| `table_s4_multi_beta.csv` | **Table S4** (SI Section S6) | Comprehensive 3D parameter sweep ($\beta \in [15, 130]$, $C_f$, $\mathrm{Fr}$) | 48 combinations |
| `table_s5_mode_competition.csv` | **Table S5** (SI Section S7) | Transverse mode competition ($m = 1, 2, 3, 4$) across aspect ratios | 5 aspect ratio stages |
| `table_s6_scenes.csv` | **Table S6** (SI Section S8.1) | Landsat/Sentinel satellite scene metadata and flow conditioning | 11 representative years |
| `table_s7_spectral_sensitivity.csv` | **Table S7** (SI Section S8.2) | Curvature smoothing window, step size, and tapering sensitivity | 11 spectral tests |
| `table_s8_width_gradient.csv` | **Table S8** (SI Section S8.3) | Along-stream channel width gradient distributions ($|\mathrm{d}B/\mathrm{d}s|$) | 11 annual distributions |
| `table_s9_curvature_sensitivity.csv` | **Table S9** (SI Section S8.4) | Sensitivity of growth rate to satellite-measured per-bend curvature | 6 curvature scales |

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
