"""
First-order Doppler loop for a time-varying Doppler profile a(t).

`run_loop_profile` is the time-varying counterpart of loop.run_loop: the same
algorithm, the same long signal (loop.LongSignal, same seed), the same
per-block noise stream ([seed, 0xA015E, block]) and the same detector call;
only the received block comes from signal_tv.rx_nominal_profile. For a
ConstantDoppler the block is produced by the existing OCDMSignal.rx, so the
trajectories equal run_loop's bit for bit (verify_time_varying.py).

Two generalisations over run_loop (approved for Step 5):
  * `mu` may be a sequence of loop gains: all loops run in lockstep on the
    same blocks (same data, same noise), each with its own a_hat; the
    per-loop arithmetic is identical to a single-mu run.
  * `a_hat0` may be an array, broadcast to (n_mu, n_snr).

Error convention: eps[..., i] = a_hat[..., i] - a_bar[i], the estimate used in
block i minus the true mean Doppler of block i. For a ramp with
beta = a_bar[i+1] - a_bar[i] per block the linear model gives a steady-state
lag eps = -beta / g (g = mu K_d).

The analytical helpers at the end give the loop's response to a sinusoid,
the ramp lag and RMSE, and the RMSE-optimal gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .params import SystemParams
from .doppler_profile import as_profile
from .loop import LongSignal
from .signal_tv import rx_nominal_profile
from . import cbsfs
from . import discriminator as disc


@dataclass
class ProfileLoopResult:
    a_hat: np.ndarray
    """a_hat[m, s, i] (or [s, i] for a scalar mu), i = 0 .. n_blocks."""
    e: np.ndarray
    """e[m, s, i] (or [s, i]), i = 0 .. n_blocks - 1."""
    a_bar: np.ndarray
    """True mean Doppler of block i."""
    a_start: np.ndarray
    """a(t) at the start of block i."""
    a_end: np.ndarray
    """a(t) at the end of block i."""
    t_block: np.ndarray
    """Start time of block i (s)."""
    snrs: tuple
    mu: np.ndarray
    profile: object

    @property
    def eps(self) -> np.ndarray:
        """a_hat[..., i] - a_bar[i], i = 0 .. n_blocks - 1."""
        return self.a_hat[..., :-1] - self.a_bar


def run_loop_profile(params: SystemParams, profile, mu, n_blocks: int,
                     snrs: Sequence[float] = (float("inf"),), a_hat0=0.0,
                     N_c: int = 24000, N_seg: int = 2048, k: int = 1,
                     delta: float | None = None, seed: int = 0,
                     start_block: int = 0) -> ProfileLoopResult:
    """Run the first-order loop over `n_blocks` consecutive blocks of one
    long received signal with Doppler `profile` (a DopplerProfile, or a
    number for a constant Doppler).

    `params` sets everything except the Doppler and the SNR (in particular
    the lag set, via params.with_(cbsfs_deltas=...)). Continuing a run with
    start_block = previous end and a_hat0 = the last a_hat reproduces one
    long run exactly.
    """
    prof = as_profile(profile)
    scalar_mu = np.ndim(mu) == 0
    mus = np.atleast_1d(np.asarray(mu, dtype=float))
    delta = 0.25 / N_seg if delta is None else float(delta)
    p = params.with_(a_true=float(prof.a(0.0)), snr_db=float("inf"))
    sig = LongSignal(p, seed)
    snrs = tuple(float(s) for s in snrs)
    amp = np.array([0.0 if np.isinf(s) else
                    np.sqrt(p.sigma_D2 / 10 ** (s / 10) / 2.0) for s in snrs])
    nm, ns = mus.size, len(snrs)
    a_hat = np.empty((nm, ns, n_blocks + 1))
    e = np.empty((nm, ns, n_blocks))
    a_hat[:, :, 0] = np.broadcast_to(np.asarray(a_hat0, dtype=float),
                                     (nm, ns))
    for i in range(n_blocks):
        b = start_block + i
        y0 = rx_nominal_profile(sig, N_c, prof, noiseless=True,
                                start_sample=b * N_c)
        if np.any(amp > 0):
            rng = np.random.default_rng([seed, 0xA015E, b])
            w = rng.standard_normal(N_c) + 1j * rng.standard_normal(N_c)
        for s in range(ns):
            y = y0 if amp[s] == 0 else y0 + amp[s] * w
            for m in range(nm):
                ap_ = disc.predicted_alpha(a_hat[m, s, i], p, k)
                c = cbsfs.cost_function(
                    y, np.array([ap_ + delta, ap_ - delta]), p, N_seg=N_seg)
                e[m, s, i] = disc.error_from_costs(c[0], c[1])
                a_hat[m, s, i + 1] = a_hat[m, s, i] + mus[m] * e[m, s, i]

    T_blk = N_c * p.T_samp_ns
    tb = (start_block + np.arange(n_blocks)) * T_blk
    res = ProfileLoopResult(
        a_hat=a_hat[0] if scalar_mu else a_hat,
        e=e[0] if scalar_mu else e,
        a_bar=prof.block_mean(tb, tb + T_blk),
        a_start=prof.a(tb), a_end=prof.a(tb + T_blk), t_block=tb,
        snrs=snrs, mu=mus, profile=prof)
    return res


# ---------------------------------------------------------------------------
# Analytical first-order-loop predictions (one update per block)
# ---------------------------------------------------------------------------
def H(freq, pole: float, T_blk: float):
    """Transfer a_bar -> a_hat, H(z) = g z^-1 / (1 - p z^-1), at freq (Hz)."""
    g = 1.0 - pole
    z1 = np.exp(-2j * np.pi * np.asarray(freq, dtype=float) * T_blk)
    return g * z1 / (1.0 - pole * z1)


def f_3db(pole: float, T_blk: float) -> float:
    """Exact 3 dB frequency (Hz) of H: |H|^2 = 1/2
    <=> cos w = (1 + p^2 - 2 g^2) / (2 p)."""
    g = 1.0 - pole
    c = (1.0 + pole ** 2 - 2.0 * g ** 2) / (2.0 * pole)
    if not -1.0 <= c <= 1.0:
        return float("nan")
    return float(np.arccos(c) / (2.0 * np.pi * T_blk))


def sigma_ss(s: float, g: float) -> float:
    """Steady-state noise RMSE (sigma_e/K_d) sqrt(g / (2 - g))."""
    return s * np.sqrt(g / (2.0 - g))


def ramp_lag(beta: float, g: float) -> float:
    """Steady-state eps = a_hat - a_bar for a ramp of beta per block."""
    return -beta / g


def ramp_rmse(beta, g, s):
    """sqrt((beta/g)^2 + s^2 g / (2 - g)) (lag and noise)."""
    g = np.asarray(g, dtype=float)
    return np.sqrt((beta / g) ** 2 + s ** 2 * g / (2.0 - g))


def optimal_g(beta: float, s: float) -> tuple[float, float]:
    """Gain g in (0, 1] minimising ramp_rmse; returns (g, rmse).

    Dense logarithmic search refined by golden section (the function is
    unimodal in g on (0, 1]). Small-g approximation: g ~ (4 beta^2/s^2)^(1/3).
    """
    gs = np.logspace(-6, 0, 4001)
    f = ramp_rmse(beta, gs, s)
    i = int(np.argmin(f))
    lo, hi = gs[max(i - 1, 0)], gs[min(i + 1, gs.size - 1)]
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(100):
        x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if ramp_rmse(beta, x1, s) < ramp_rmse(beta, x2, s):
            hi = x2
        else:
            lo = x1
    g = 0.5 * (lo + hi)
    return float(g), float(ramp_rmse(beta, g, s))


def sine_rmse(amp: float, freq: float, pole: float, T_blk: float,
              s: float) -> float:
    """Tracking RMSE for a sinusoid of amplitude amp at freq (Hz):
    sqrt(amp^2 |1 - H|^2 / 2 + s^2 g / (2 - g))."""
    g = 1.0 - pole
    h = H(freq, pole, T_blk)
    return float(np.sqrt(amp ** 2 * abs(1 - h) ** 2 / 2
                         + s ** 2 * g / (2.0 - g)))
