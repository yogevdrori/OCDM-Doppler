"""
Early-late Doppler discriminator in the cycle-frequency domain.

Built on the CB-SFS cost function C(alpha), [CBSFS] Eqn. (29), which peaks at
the cycle frequencies of the received signal, k (1 + a) / N_sym. Given a
current Doppler estimate a_hat, the harmonic is predicted at

    alpha_p = k (1 + a_hat) / N_sym

and the cost is evaluated at two points straddling it:

    C+ = C(alpha_p + delta),   C- = C(alpha_p - delta)
    e  = (C+ - C-) / (C+ + C-)      (normalised; raw: C+ - C-)

SIGN CONVENTION: e > 0 means a_hat is too SMALL. If a_hat < a then
alpha_p < k (1 + a) / N_sym, i.e. the prediction lies below the peak, so the
late point alpha_p + delta is nearer the peak and C+ > C-. The loop update is
therefore a_hat <- a_hat + mu * e, and near the truth
e ~ K_d (a - a_hat) with K_d > 0.

The main lobe of C around a harmonic has nulls at about +/- 1/N_seg. The
default delta is 0.25 / N_seg: in the Step-3 study (study_scurve.py) it gave
the lowest sigma_e / K_d of {0.25, 0.5, 0.75} / N_seg while its pull-in range
(+/- 9.9e-3) still covers the prior range a_min..a_max; larger delta
steepens the S-curve and widens the pull-in, but the noise grows faster
because the probe points sit lower on the lobe. A Doppler offset
a_hat - a moves alpha_p by k (a_hat - a) / N_sym, so the discriminator keeps
the correct sign roughly while |a_hat - a| < N_sym / (k N_seg) (the pull-in
range).

Only two cost evaluations are needed per update, instead of the grid search
of [CBSFS] Sec. V-B; `cbsfs.cost_function` accepts arbitrary alpha arrays,
so nothing in cbsfs.py changes.
"""

from __future__ import annotations

import numpy as np

from .params import SystemParams
from . import cbsfs


def predicted_alpha(a_hat, params: SystemParams, k: int = 1):
    """alpha_p = k (1 + a_hat) / N_sym (a_hat may be an array)."""
    return k * (1.0 + np.asarray(a_hat, dtype=float)) / params.N_sym


def error_from_costs(c_plus, c_minus, normalized: bool = True):
    """e from the late / early costs; e > 0 means a_hat is too small."""
    c_plus = np.asarray(c_plus, dtype=float)
    c_minus = np.asarray(c_minus, dtype=float)
    if not normalized:
        return c_plus - c_minus
    den = c_plus + c_minus
    return np.divide(c_plus - c_minus, den,
                     out=np.zeros(np.broadcast(c_plus, c_minus).shape),
                     where=den > 0)


def early_late(y: np.ndarray, a_hat, params: SystemParams, k: int = 1,
               delta: float | None = None, N_seg: int | None = None,
               normalized: bool = True):
    """Early-late discriminator output e at one or more Doppler estimates.

    `delta` defaults to 0.25 / N_seg (lowest sigma_e / K_d in the Step-3
    study while still covering the prior range; see the module docstring).
    All 2 * len(a_hat) cycle frequencies are evaluated in a single
    `cbsfs.cost_function` call. Returns a scalar for a scalar `a_hat`.
    """
    N_seg = params.N_seg if N_seg is None else int(N_seg)
    delta = 0.25 / N_seg if delta is None else float(delta)
    a_arr = np.atleast_1d(np.asarray(a_hat, dtype=float))
    ap = predicted_alpha(a_arr, params, k)
    cost = cbsfs.cost_function(y, np.concatenate([ap + delta, ap - delta]),
                               params, N_seg=N_seg)
    e = error_from_costs(cost[:a_arr.size], cost[a_arr.size:], normalized)
    return float(e[0]) if np.ndim(a_hat) == 0 else e


def s_curve(y: np.ndarray, a_true: float, offsets, params: SystemParams,
            **kwargs) -> np.ndarray:
    """Discriminator S-curve: e at a_hat = a_true + offsets."""
    offsets = np.asarray(offsets, dtype=float)
    return early_late(y, a_true + offsets, params, **kwargs)
