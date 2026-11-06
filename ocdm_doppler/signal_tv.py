"""
Received signal for a time-varying Doppler factor a(t).

Time-varying counterparts of OCDMSignal.rx / rx_nominal ([OCDM] Eqn. 3 with
P = 1). With A(t) = int_0^t a(s) ds (doppler_profile.py, closed form):

    u(t)     = t + A(t) - tau_p0                              (time warp)
    phase(t) = 2 pi ((f_Tx - f_Rx) t + f_Tx A(t)) + Delta_phi (carrier)

and the front factor A_p0 exp(-j 2 pi f_Tx tau_p0) / (2 sqrt(N_is)) as in
OCDMSignal.rx. For a constant a, A(t) = a t and these are the expressions
of OCDMSignal.rx, up to floating-point rounding ((1 + a) t vs t + a t).
A ConstantDoppler is therefore delegated to the existing OCDMSignal.rx, so
that constant-Doppler results are bit-identical to the old ones; any other
profile (including a constant one given as, e.g., RampDoppler(a, 0)) uses
the general formulas.

Nothing is copied from signal_model: the symbol decomposition, the chirp sum
and the noise generator of the passed signal object are reused
(sig._decompose, sig._chirp_sum, sig._noise), so these functions work for
OCDMSignal and loop.LongSignal alike. (LongSignal has no noise generator:
use noiseless=True there; the loop adds its own per-block noise.)
"""

from __future__ import annotations

import numpy as np

from .doppler_profile import ConstantDoppler, as_profile


def rx_profile(sig, t, profile, noiseless: bool = False,
               noise_index=None) -> np.ndarray:
    """Y(t) for a Doppler profile (or a constant, wrapped in ConstantDoppler).

    `noise_index` as in OCDMSignal.rx: integer T_s-grid indices for the noise
    lookup, default rint(t / T_s).
    """
    p = sig.p
    prof = as_profile(profile)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if not noiseless and p.noise_power > 0.0 and sig._noise is None:
        raise ValueError("this signal object has no noise generator "
                         "(e.g. loop.LongSignal): use noiseless=True")
    if isinstance(prof, ConstantDoppler):
        # constant Doppler exactly as today: the existing OCDMSignal.rx
        # ((1 + a) t rather than t + a t, bit-identical to all old results)
        return sig.rx(t, a=prof.a0, noiseless=noiseless,
                      noise_index=noise_index)
    At = prof.A(t)

    u = t + At - p.tau_p0
    m, r, active = sig._decompose(u)
    s = sig._chirp_sum(r, m, active)

    front = (p.A_p0 * np.exp(-2j * np.pi * p.f_Tx * p.tau_p0)
             / (2.0 * np.sqrt(p.N_is)))
    phase = 2.0 * np.pi * ((p.f_Tx - p.f_Rx) * t + p.f_Tx * At) + p.Delta_phi
    y = front * np.exp(1j * phase) * s

    if not noiseless and p.noise_power > 0.0:
        if noise_index is None:
            noise_index = np.rint(t / p.T_s).astype(np.int64)
        y = y + sig._noise.at(noise_index)
    return y


def rx_nominal_profile(sig, n: int, profile, noiseless: bool = False,
                       start_sample: int = 0) -> np.ndarray:
    """Y at the receiver's nominal rate, Y(k T_samp_ns) for
    k = start_sample .. start_sample + n - 1 (as OCDMSignal.rx_nominal: exact
    instants, noise indexed by the nearest T_s grid point)."""
    p = sig.p
    k = start_sample + np.arange(int(n), dtype=np.int64)
    t = k * p.T_samp_ns
    return rx_profile(sig, t, profile, noiseless=noiseless,
                      noise_index=sig.grid_index(t))
