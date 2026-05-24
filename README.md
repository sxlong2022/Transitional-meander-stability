# Planar 2D SWE--Exner Stability of Alternate Bars in Transitional Rivers

> **Open-Source Code for the Manuscript:**  
> *Planar 2D SWE--Exner stability of alternate bars in the transitional Lower Yellow River* (Submitted to *Earth Surface Processes and Landforms*)

This repository provides the core theoretical framework and analysis tools used to evaluate the linear morphodynamic stability of extremely wide, transitional meandering-braided rivers.

## Overview
Traditional river stability theory has predominantly been tested under laboratory conditions. This project scales planar two-dimensional shallow-water equation (2D SWE) coupled with the Exner sediment continuity equation (planar 2D SWE--Exner) stability diagnosis to field-scale geomorphic parameters. 

### Key Capabilities
- **`os_solver/`**: 
  - `solve_os.py`: Core Chebyshev collocation eigenvalue solver for bar instability.
  - `os_operator_curved.py`: Implements curvature-induced secondary flow (bend effects) into the basic 2D SWE--Exner stability equations.
  - `u1_shape.py`: Solves for the unperturbed base flow shape function in a curved channel.
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
If you use this code in your research, please cite our corresponding *Earth Surface Processes and Landforms* (ESPL) manuscript.

## License
MIT License.
