"""
Doppler tracking loop with arbitrary loop filters (loop_filters.py).

`run_loop_filters` is the counterpart of loop_tv.run_loop_profile with the
update rule delegated to filter objects: the same long signal
(loop.LongSignal, same seed), the same received block
(signal_tv.rx_nominal_profile), the same per-block noise stream
([seed, 0xA015E, block]), the same detector call and the same lockstep
design (all filters x SNRs on the same blocks). A FirstOrderFilter(mu)
therefore reproduces run_loop_profile(mu) bit for bit
(verify_second_order.py).

Checkpointing: the result carries the final filter states; passing them as
`states` with start_block = previous end continues the run exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .params import SystemParams
from .doppler_profile import as_profile
from .loop import LongSignal
from .signal_tv import rx_nominal_profile
from .loop_filters import LoopFilter
from . import cbsfs
from . import discriminator as disc


@dataclass
class FilterLoopResult:
    a_hat: np.ndarray
    """a_hat[m, s, i], i = 0 .. n_blocks (a_hat[..., i] is used in block i)."""
    b_hat: np.ndarray
    """b_hat[m, s, i] (NaN for first-order filters)."""
    e: np.ndarray
    """e[m, s, i], i = 0 .. n_blocks - 1."""
    a_bar: np.ndarray
    a_start: np.ndarray
    a_end: np.ndarray
    t_block: np.ndarray
    snrs: tuple
    states: list
    """states[m][s]: final filter state, for continuing the run."""
    labels: list

    @property
    def eps(self) -> np.ndarray:
        """a_hat[..., i] - a_bar[i], i = 0 .. n_blocks - 1."""
        return self.a_hat[..., :-1] - self.a_bar


def run_loop_filters(params: SystemParams, profile,
                     filters: Sequence[LoopFilter], n_blocks: int,
                     snrs: Sequence[float] = (float("inf"),),
                     N_c: int = 24000, N_seg: int = 2048, k: int = 1,
                     delta: float | None = None, seed: int = 0,
                     start_block: int = 0, states=None) -> FilterLoopResult:
    """Run every filter (copied once per SNR) on `n_blocks` consecutive
    blocks of one long received signal with Doppler `profile`.

    The filters' own a_hat / b_hat are the initial state, unless `states`
    (states[m][s], as returned in the result) is given.
    """
    prof = as_profile(profile)
    delta = 0.25 / N_seg if delta is None else float(delta)
    p = params.with_(a_true=float(prof.a(0.0)), snr_db=float("inf"))
    sig = LongSignal(p, seed)
    snrs = tuple(float(s) for s in snrs)
    amp = np.array([0.0 if np.isinf(s) else
                    np.sqrt(p.sigma_D2 / 10 ** (s / 10) / 2.0) for s in snrs])
    nm, ns = len(filters), len(snrs)
    F = [[f.copy() for _ in range(ns)] for f in filters]
    if states is not None:
        for m in range(nm):
            for s in range(ns):
                F[m][s].set_state(states[m][s])
    a_hat = np.empty((nm, ns, n_blocks + 1))
    b_hat = np.full((nm, ns, n_blocks + 1), np.nan)
    e = np.empty((nm, ns, n_blocks))
    for m in range(nm):
        for s in range(ns):
            a_hat[m, s, 0] = F[m][s].a_hat
            if F[m][s].order == 2:
                b_hat[m, s, 0] = F[m][s].b_hat

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
                f = F[m][s]
                ap_ = disc.predicted_alpha(f.a_hat, p, k)
                c = cbsfs.cost_function(
                    y, np.array([ap_ + delta, ap_ - delta]), p, N_seg=N_seg)
                e[m, s, i] = disc.error_from_costs(c[0], c[1])
                a_hat[m, s, i + 1] = f.update(e[m, s, i])
                if f.order == 2:
                    b_hat[m, s, i + 1] = f.b_hat

    T_blk = N_c * p.T_samp_ns
    tb = (start_block + np.arange(n_blocks)) * T_blk
    return FilterLoopResult(
        a_hat=a_hat, b_hat=b_hat, e=e,
        a_bar=prof.block_mean(tb, tb + T_blk), a_start=prof.a(tb),
        a_end=prof.a(tb + T_blk), t_block=tb, snrs=snrs,
        states=[[F[m][s].get_state() for s in range(ns)] for m in range(nm)],
        labels=[getattr(f, "label", "") for f in filters])
