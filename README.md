# Field-scale Orr-Sommerfeld Diagnosis of Bar Instability

> **Open-Source Code for the Manuscript:**  
> *Field-scale Orr-Sommerfeld diagnosis of ubiquitous bar instability in the transitional Lower Yellow River* (Submitted to *Geomorphology*)

This repository provides the core theoretical framework and analysis tools used to evaluate the linear morphodynamic stability of extremely wide, transitional meandering-braided rivers.

## Overview
Traditional river stability theory has predominantly been tested under laboratory conditions. This project scales the classic Orr-Sommerfeld (OS) fluid mechanics stability diagnosis to field-scale geomorphic parameters. 

### Key Capabilities
- **`os_solver/`**: 
  - `solve_os.py`: Core Chebyshev collocation eigenvalue solver for bar instability.
  - `os_operator_curved.py`: Implements curvature-induced secondary flow (bend effects) into the basic OS equations.
  - `u1_shape.py`: Solves for the unperturbed base flow shape function in a curved channel.
- **`utils/`**:
  - `diagnose_stability.py`: A wrapper to easily diagnose if a reach (given Width/Depth ratio, Froude number, friction coefficient, and curvature) is unstable, extracting maximum growth rates and critical wavelengths.
  - `analyze_profiles.py`: Spatial spectral analysis tools (FFT, Autocorrelation) for identifying fundamental length scales from planform variations.
  - `predict_bars.py`: Bar mode predictor combining Crosato-Mosselman empirical formula with OS instability window.
  - **`utils/`**:
  - `diagnose_stability.py`: A wrapper to easily diagnose if a reach (given Width/Depth ratio, Froude number, friction coefficient, and curvature) is unstable, extracting maximum growth rates and critical wavelengths.
  - `analyze_profiles.py`: Spatial spectral analysis tools (FFT, Autocorrelation) for identifying fundamental length scales from planform variations.

(Note: `predict_bars.py` and spectral modules are not included in this release.)
  - `analyze_profiles.py`: Spatial spectral analysis tools (FFT, Autocorrelation) for identifying fundamental length scales from planform variations.
  - `predict_bars.py`: Bar mode predictor using empirical and physics-based models (e.g., Crosato-Mosselman).

## Setup & Requirements

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

*Main dependencies:* `numpy`, `scipy`, `matplotlib`, `pandas`.

## Usage

See the `examples/` directory for a quick start.

```bash
# Run a simple 1D linear stability diagnosis
python -m examples.example_01_solve_os
```

## Citation
If you use this code in your research, please cite our corresponding *Geomorphology* manuscript.

## License
MIT License.
