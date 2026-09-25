"""
Empirical AF estimation along the lag axis -- Step 2 of the results documents.

Implements, on top of the Step-1 infrastructure (nothing there is modified):

  2(a)  time-value selection by the variance estimate, Eqn. (1)
  2(b)  uniform lag grid, empirical AF at each lag, and 'window' detection
        (the floor(T_is / ((1 + a_sr) tau_reso)) consecutive lags of highest
        power)

This is shared by the figures that reproduce the results documents and by the
Section-4 curve-fitting estimator (Step 2 of the plan) that will follow.

TIME ORIGIN
-----------
The documents average over i = -(N_avg-1)/2 .. +(N_avg-1)/2 around t' = 0 or
t' = T_sym / (3 (1 + a_max)), i.e. they assume the signal exists at negative
times. `OCDMSignal` only has symbols m >= 0, so the receiver's clock origin is
placed at the start of received symbol K (K > (N_avg-1)/2):

    t_origin = K * T_sym / (1 + a)

This shifts (1 + a) t - tau_p0 by exactly K T_sym, so m -> m + K and nothing
else changes (the within-symbol position, Gamma and |Pi| are identical). All
document time values t' are interpreted relative to t_origin.

MODULUS PLACEMENT
-----------------
The documents write the modulus *inside* the average. The published figures
are only consistent with the modulus *outside*: e.g. in the 31-Jan noiseless
figure the empirical AF drops to ~0.02 at the first null, whereas
E{|Y(t)| |Y(t - tau)|} would stay near sigma_D^2 |A0|^2 / 4 * pi/4 ~ 0.25
there. Hence `modulus="outside"` is the default; "inside" is kept for testing
(see README open question 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .params import SystemParams
from .signal_model import OCDMSignal
from . import analytical_af

Modulus = Literal["outside", "inside"]


def make_signal(params: SystemParams, n_extra_symbols: int = 0,
                seed: int | None = None,
                a: float | None = None) -> tuple[OCDMSignal, float]:
    """Signal long enough for N_avg symbols centred on the time origin.

    Returns (signal, t_origin). `n_extra_symbols` extends the record (e.g. so
    CB-SFS, which samples from t = 0, has M symbols available).
    """
    a = params.a_true_eff if a is None else a
    K = (params.N_avg - 1) // 2 + 4
    n_sym = max(2 * K + 4, n_extra_symbols + 4)
    sig = OCDMSignal(params, n_symbols=n_sym, seed=seed)
    return sig, K * params.T_sym / (1.0 + a)


def lag_grid(half_width: float, N_lag: int) -> tuple[np.ndarray, float]:
    """N_lag uniform lags strictly inside (-half_width, half_width).

    The interval is divided into N_lag + 1 portions ([OCDM] Sec. 4, step
    2(b)), so tau_reso = 2 half_width / (N_lag + 1) and the lags are
    -half_width + k tau_reso, k = 1 .. N_lag.
    """
    reso = 2.0 * half_width / (N_lag + 1)
    return -half_width + reso * np.arange(1, N_lag + 1), reso


def _avg_times(t: float, T_sr: float, N_avg: int) -> np.ndarray:
    i = np.arange(N_avg) - (N_avg - 1) // 2
    return t + i * T_sr


def variance_estimate(sig: OCDMSignal, t: float, T_sr: float) -> float:
    """sigma_hat^2_Y(t'), results documents Eqn. (1)."""
    idx = sig.grid_index(_avg_times(t, T_sr, sig.p.N_avg))
    return float(np.mean(np.abs(sig.rx_at_grid(idx)) ** 2))


def empirical_af(sig: OCDMSignal, t: float, lags: np.ndarray, T_sr: float,
                 modulus: Modulus = "outside") -> np.ndarray:
    """c_hat_Y(t, tau_k) for every lag, averaged over N_avg received symbols.

    Every time instant is quantised to the T_s grid, floor(x / T_s + 1/2).
    """
    tt = _avg_times(t, T_sr, sig.p.N_avg)
    y1 = sig.rx_at_grid(sig.grid_index(tt))
    out = np.empty(len(lags))
    for k, lag in enumerate(lags):
        y2 = sig.rx_at_grid(sig.grid_index(tt - lag))
        prod = y1 * np.conj(y2)
        out[k] = (np.abs(prod).mean() if modulus == "inside"
                  else np.abs(prod.mean()))
    return out


def detect_window(af: np.ndarray, length: int) -> slice:
    """The `length` consecutive lag samples with the highest power."""
    if length >= af.size:
        return slice(0, af.size)
    p = np.concatenate([[0.0], np.cumsum(af ** 2)])
    energy = p[length:] - p[:-length]
    s = int(np.argmax(energy))
    return slice(s, s + length)


@dataclass
class WindowedAF:
    t_sel: float
    """Selected time value, relative to the time origin."""
    t_abs: float
    """Selected time value on the signal's own time axis."""
    lags: np.ndarray
    """All lags on the grid."""
    af: np.ndarray
    """Empirical AF magnitude at every lag."""
    window: slice
    """Detected window (indices into `lags`)."""
    tau_reso: float
    a_sr: float
    sigma2: tuple[float, ...]
    """Variance estimates at the candidate time values."""

    @property
    def lags_win(self) -> np.ndarray:
        return self.lags[self.window]

    @property
    def af_win(self) -> np.ndarray:
        return self.af[self.window]

    def analytical(self, a: float, params: SystemParams) -> np.ndarray:
        """Noiseless |c_Y(t_sel, tau)| on the windowed lags for Doppler a."""
        return analytical_af.af_magnitude(self.t_abs, self.lags_win, a, params)


def estimate_windowed_af(sig: OCDMSignal, t_origin: float, a_sr: float,
                         t_candidates: Sequence[float],
                         lag_half_width: float | None = None,
                         lags: np.ndarray | None = None,
                         modulus: Modulus = "outside") -> WindowedAF:
    """Steps 2(a)-(b) of the results documents.

    Parameters
    ----------
    a_sr
        Initial (symbol-rate) Doppler estimate; T_sr = T_sym / (1 + a_sr).
    t_candidates
        Candidate time values t'_k (relative to the time origin); the one
        with the largest variance estimate is selected.
    lag_half_width
        Lags are placed in (-lag_half_width, lag_half_width). Defaults to
        T_is / (1 + a_min) as in the 22-Jan document.
    lags
        Explicit lag values; overrides the uniform grid (used for the
        'known window' experiment). No window detection is applied then.
    """
    p = sig.p
    T_sr = p.T_sym / (1.0 + a_sr)

    sig2 = tuple(variance_estimate(sig, t_origin + t, T_sr)
                 for t in t_candidates)
    t_sel = float(t_candidates[int(np.argmax(sig2))])
    t_abs = t_origin + t_sel

    if lags is None:
        hw = p.T_is / (1.0 + p.a_min_eff) if lag_half_width is None \
            else lag_half_width
        lags, reso = lag_grid(hw, p.N_lag)
        af = empirical_af(sig, t_abs, lags, T_sr, modulus)
        win = detect_window(af, int(np.floor(p.T_is / ((1.0 + a_sr) * reso))))
    else:
        lags = np.asarray(lags, dtype=float)
        reso = float(lags[1] - lags[0]) if lags.size > 1 else float("nan")
        af = empirical_af(sig, t_abs, lags, T_sr, modulus)
        win = slice(0, lags.size)

    return WindowedAF(t_sel, t_abs, lags, af, win, reso, a_sr, sig2)


def true_window(t_abs: float, params: SystemParams,
                a: float | None = None) -> tuple[float, float]:
    """The analytical lag window (lo, hi] at time t_abs, [OCDM] Sec. 4-2(b).

    t - (tau_p0 + m T_sym)/(1+a) - T_is/(1+a) < tau <= t - (tau_p0 + m T_sym)/(1+a)
    """
    a = params.a_true_eff if a is None else a
    m = analytical_af.symbol_index(t_abs, a, params)
    hi = t_abs - (params.tau_p0 + m * params.T_sym) / (1.0 + a)
    return float(hi - params.T_is / (1.0 + a)), float(hi)
