"""
OCDM Doppler estimation - shared simulation infrastructure.

Modules
-------
params         SystemParams: one object describes an entire experiment.
signal_model   OCDM transmitter and single-path Doppler receiver, Y(t).
analytical_af  Closed-form AF of the received signal ([OCDM] Eqns. 8-12).
cbsfs          CB-SFS cycle-frequency / Doppler estimation ([CBSFS] 27-34).

The estimators in later stages (curve-fitting baseline, and the planned
closed-loop detector) build on these without modifying them.
"""

from .params import SystemParams, preset_noiseless_31jan, preset_noisy_22jan
from .signal_model import OCDMSignal, draw_symbols, chirp_matrix
from . import analytical_af
from . import cbsfs

__all__ = [
    "SystemParams",
    "preset_noiseless_31jan",
    "preset_noisy_22jan",
    "OCDMSignal",
    "draw_symbols",
    "chirp_matrix",
    "analytical_af",
    "cbsfs",
]