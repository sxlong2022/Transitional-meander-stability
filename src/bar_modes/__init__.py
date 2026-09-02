"""Subproject-3 bar mode prediction and diagnostics subpackage."""
from .predict_bars import (  # noqa: F401
    BarDiagnostic,
    compute_sigma_width,
    compute_sigma_width_stats,
    compute_unstable_window,
    diagnose_bar_regime,
    predict_bar_mode_cm,
    run_sigma_width_batch,
)
