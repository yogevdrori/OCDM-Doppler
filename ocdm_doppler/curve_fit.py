"""
Section-4 curve-fitting Doppler estimator -- Step 3 of the results documents.

Given the windowed empirical AF (af_estimation.estimate_windowed_af), place
N_a candidate Doppler values uniformly in (a_min, a_max) and output the one
minimising the MSE between the empirical AF and the analytical |Pi * Gamma|,
[OCDM] Eqns. (11)-(12).

The analytical curve also needs |A_p0|^2 and tau_p0. As in [OCDM] Sec. 4 they
are estimated from the same data:

    |A0|^2_hat  = 4 sigma_hat^2_Y(t_sel) / sigma_D^2                  (2a)
    tau_p0_hat  = (t_sel - tau_hat_win_r) (1 + a_sr)                  (2b)

with tau_hat_win_r the upper end of the detected lag window. No noise-power
subtraction is applied in (2a), so |A0|^2_hat is biased by 4 P_n / sigma_D^2;
this matches the 22-Jan table (NMSE 0.095 at 10 dB = (4 * 0.1 / 1.265)^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import SystemParams
from .af_estimation import WindowedAF
from . import analytical_af

_CHUNK = 8192


def estimate_A0_sq(w: WindowedAF, params: SystemParams) -> float:
    """|A0|^2_hat = 4 sigma_hat^2_Y(t_sel) / sigma_D^2."""
    return 4.0 * max(w.sigma2) / params.sigma_D2


def estimate_tau_p0(w: WindowedAF) -> float:
    """tau_p0_hat = (t_sel - tau_hat_win_r) (1 + a_sr)."""
    return (w.t_sel - float(w.lags_win[-1])) * (1.0 + w.a_sr)


@dataclass
class CurveFitResult:
    a_hat: float
    """Doppler estimate a_hat_mag."""
    A0_sq: float
    """|A_p0|^2 used in the analytical AF (estimated unless given)."""
    tau_p0: float
    """tau_p0 used in the analytical AF (estimated unless given)."""
    a_grid: np.ndarray
    cost: np.ndarray
    """MSE between empirical and analytical AF for every candidate."""


def cost_curve(w: WindowedAF, params: SystemParams, a_grid: np.ndarray,
               A0_sq: float, tau_p0: float) -> np.ndarray:
    """MSE over the windowed lags for each candidate Doppler value."""
    tau = w.lags_win[None, :]
    emp = w.af_win[None, :]
    A = np.sqrt(A0_sq) + 0j
    out = np.empty(a_grid.size)
    for s in range(0, a_grid.size, _CHUNK):
        a = a_grid[s:s + _CHUNK, None]
        ana = analytical_af.af_magnitude(w.t_sel, tau, a, params,
                                         tau_p0=tau_p0, A_p0=A)
        out[s:s + a.shape[0]] = np.mean((emp - ana) ** 2, axis=1)
    return out


def fit_doppler(w: WindowedAF, params: SystemParams,
                N_a: int | None = None,
                A0_sq: float | None = None,
                tau_p0: float | None = None) -> CurveFitResult:
    """Curve-fitting Doppler estimate over N_a candidates in (a_min, a_max).

    `A0_sq` / `tau_p0` default to their estimates (2a) / (2b); pass values to
    use known ones instead.
    """
    N_a = params.N_a if N_a is None else int(N_a)
    A0_sq = estimate_A0_sq(w, params) if A0_sq is None else float(A0_sq)
    tau_p0 = estimate_tau_p0(w) if tau_p0 is None else float(tau_p0)

    # N_a values strictly inside the open interval (a_min, a_max)
    lo, hi = params.a_min_eff, params.a_max
    a_grid = lo + (hi - lo) * np.arange(1, N_a + 1) / (N_a + 1)

    cost = cost_curve(w, params, a_grid, A0_sq, tau_p0)
    return CurveFitResult(float(a_grid[int(np.argmin(cost))]), A0_sq, tau_p0,
                          a_grid, cost)
