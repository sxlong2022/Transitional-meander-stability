"""Global configuration and physical constants for the Transitional Meander Stability Project."""
from __future__ import annotations

from pathlib import Path

# ── Directory Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "src" / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "publication_figures"
OUTPUT_DIR = FIGURES_DIR / "output"

# ── Physical Constants ───────────────────────────────────────────────────────
G = 9.80665          # Gravitational acceleration (m/s^2)
RHO_W = 1000.0       # Water density (kg/m^3)
RHO_S = 2650.0       # Quartz sediment density (kg/m^3)
DELTA = (RHO_S - RHO_W) / RHO_W  # Relative submerged density = 1.65

# ── River Reach Constants (Gaocun-Sunkou reach, Lower Yellow River) ──────────
GAOCUN_DAM_KM = 303.0    # Distance of Gaocun station from Xiaolangdi Dam (km)
SUNKOU_DAM_KM = 421.0    # Distance of Sunkou station from Xiaolangdi Dam (km)
REACH_LENGTH_KM = SUNKOU_DAM_KM - GAOCUN_DAM_KM  # ~ 118 km

# Geographic coordinates (WGS84)
GAOCUN_LON = 115.0759    # Gaocun station longitude (°E)
GAOCUN_LAT = 35.3641     # Gaocun station latitude (°N)
SUNKOU_LON = 115.9052    # Sunkou station longitude (°E)
SUNKOU_LAT = 35.9340     # Sunkou station latitude (°N)

if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"RESULTS_DIR  = {RESULTS_DIR}  (exists={RESULTS_DIR.exists()})")
    print(f"FIGURES_DIR  = {FIGURES_DIR}  (exists={FIGURES_DIR.exists()})")
