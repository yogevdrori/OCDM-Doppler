"""
Analytical autocorrelation function of the received OCDM signal.

Single path, noiseless -- [OCDM] Eqns. (8)-(12):

    c_Y(t, tau) = Pi_{a,A0}(tau) * exp(j Theta_{a,tau_p0}(t, tau))
                                 * Gamma_{a,tau_p0}(t, tau)

The curve-fitting stage of the Section-4 algorithm compares the *magnitude*
of an empirical AF estimate against |Pi * Gamma| evaluated for candidate
Doppler values, so `af_magnitude` is the workhorse here.
"""

from __future__ import annotations

import numpy as np

from .params import SystemParams

_TINY = 1e-300


def symbol_index(t: np.ndarray, a: float, params: SystemParams,
                 tau_p0: float | None = None) -> np.ndarray:
    """m_{a,tau_p0,t} = floor(((1+a) t - tau_p0) / T_sym)  -- [OCDM] Eqn. (10)."""
    tau_p0 = params.tau_p0 if tau_p0 is None else tau_p0
    t = np.asarray(t, dtype=float)
    return np.floor(((1.0 + a) * t - tau_p0) / params.T_sym)


def Pi(tau: np.ndarray, a: float, params: SystemParams,
       A_p0: complex | None = None) -> np.ndarray:
    """Pi_{a,A0}(tau) -- [OCDM] Eqn. (11).

    Dirichlet kernel in tau * (1 + a); this is the factor whose *shape*
    carries the Doppler information in the lag domain.

    At tau = 0 the ratio tends to N_is, giving the peak value
    sigma_D^2 |A0|^2 / 4 (0.324 for the documents' parameters).
    """
    A_p0 = params.A_p0 if A_p0 is None else A_p0
    tau = np.asarray(tau, dtype=float)

    scale = params.sigma_D2 * (abs(A_p0) ** 2) / (4.0 * params.N_is)
    x = np.pi * tau * (1.0 + a) / params.T_is

    num = np.sin(params.N_is * x)
    den = np.sin(x)
    ratio = np.where(np.abs(den) < 1e-12,
                     float(params.N_is) * np.cos(params.N_is * x) / np.where(
                         np.abs(np.cos(x)) < _TINY, 1.0, np.cos(x)),
                     num / np.where(np.abs(den) < 1e-12, 1.0, den))
    return scale * ratio


def Theta(t: np.ndarray, tau: np.ndarray, a: float, params: SystemParams,
          tau_p0: float | None = None) -> np.ndarray:
    """Theta_{a,tau_p0}(t, tau) -- [OCDM] Eqn. (12)."""
    tau_p0 = params.tau_p0 if tau_p0 is None else tau_p0
    t = np.asarray(t, dtype=float)
    tau = np.asarray(tau, dtype=float)

    m = symbol_index(t, a, params, tau_p0)
    T_is, T_sym, N_is = params.T_is, params.T_sym, params.N_is
    g = 1.0 + a

    term1 = 2.0 * params.Delta_f_of_a(a) * tau
    term2 = N_is * tau * g * (g * (tau - 2.0 * t) + 2.0 * tau_p0) / T_is ** 2
    term3 = tau * g / T_is * (N_is - 1.0)
    term4 = 2.0 * N_is * tau * g * m * T_sym / T_is ** 2
    return np.pi * (term1 + term2 + term3 + term4)


def Gamma(t: np.ndarray, tau: np.ndarray, a: float, params: SystemParams,
          tau_p0: float | None = None) -> np.ndarray:
    """Gamma_{a,tau_p0}(t, tau) -- [OCDM] Eqn. (12).

    Product of the two pulse-shaping windows; equals 1 only when both t and
    t - tau fall inside the same OCDM symbol's active part (not the guard
    interval), and 0 otherwise.
    """
    tau_p0 = params.tau_p0 if tau_p0 is None else tau_p0
    t = np.asarray(t, dtype=float)
    tau = np.asarray(tau, dtype=float)

    m = symbol_index(t, a, params, tau_p0)
    base = tau_p0 + m * params.T_sym
    u1 = (1.0 + a) * t - base
    u2 = (1.0 + a) * (t - tau) - base

    inside = lambda u: (u >= 0.0) & (u < params.T_is)
    return (inside(u1) & inside(u2)).astype(float)


def c_Y(t: np.ndarray, tau: np.ndarray, a: float, params: SystemParams,
        tau_p0: float | None = None,
        A_p0: complex | None = None) -> np.ndarray:
    """Complex analytical AF c_Y(t, tau) -- [OCDM] Eqn. (8)."""
    return (Pi(tau, a, params, A_p0)
            * np.exp(1j * Theta(t, tau, a, params, tau_p0))
            * Gamma(t, tau, a, params, tau_p0))


def af_magnitude(t: float, tau: np.ndarray, a: float, params: SystemParams,
                 tau_p0: float | None = None,
                 A_p0: complex | None = None) -> np.ndarray:
    """|c_Y(t, tau)| = |Pi(tau)| * Gamma(t, tau).

    This is the quantity the curve-fitting stage matches against the
    empirical AF magnitude.
    """
    t_arr = np.full(np.shape(tau), float(t))
    return (np.abs(Pi(tau, a, params, A_p0))
            * Gamma(t_arr, tau, a, params, tau_p0))


def first_null_lag(a: float, params: SystemParams) -> float:
    """Lag of the first null of Pi: tau = T_is / (N_is (1 + a))."""
    return params.T_is / (params.N_is * (1.0 + a))


def lag_support(a: float, params: SystemParams) -> float:
    """c_Y(t, tau) = 0 for |tau| >= T_is / (1 + a)."""
    return params.T_is / (1.0 + a)
