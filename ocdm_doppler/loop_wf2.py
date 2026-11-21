"""
Loop runner and open-loop detector with the cost function as a parameter.

As loop_wf.run_loop_wf / open_loop_e (waveform + channel, harmonic k per
filter, Step 6 filter objects, same block generation and noise stream), but
each filter may also use its own cost function (`dets`): e.g.
cbsfs.cost_function (the current detector) and cbsfs_dc.cost_function_dc
(DC line removed) in lockstep on the same blocks. With
dets = cbsfs.cost_function it reproduces loop_wf bit for bit
(study_otfs.py --part D checks this before running).

The channel's total gain is set through params.A_p0 (channel_norm.py
scales it so that the mean received signal power equals that of the
single-path reference).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .doppler_profile import as_profile
from .channel import MultipathChannel, rx_block
from .loop_tv2 import FilterLoopResult
from .params import SystemParams
from .wf_params import detector_params
from .loop_wf import _noise_amp, _block_noise
from . import cbsfs
from . import discriminator as disc


def _per_filter(x, n):
    return [x] * n if (callable(x) or np.ndim(x) == 0) else list(x)


def run_loop_det(params: SystemParams, waveform, channel: MultipathChannel,
                 profile, filters, n_blocks: int,
                 snrs: Sequence[float] = (float("inf"),), N_c: int = 24000,
                 N_seg: int = 2048, ks=1, dets=cbsfs.cost_function,
                 delta: float | None = None, seed: int = 0,
                 start_block: int = 0, states=None) -> FilterLoopResult:
    """run_loop_wf with a cost function per filter (`dets`)."""
    prof = as_profile(profile)
    delta = 0.25 / N_seg if delta is None else float(delta)
    p = params.with_(a_true=float(prof.a(0.0)), snr_db=float("inf"))
    wsig = waveform.signal(p, seed)
    dp = detector_params(p, wsig.P_samp)
    snrs = tuple(float(s) for s in snrs)
    amp = _noise_amp(p, snrs)
    nm, ns = len(filters), len(snrs)
    ks = [int(k) for k in _per_filter(ks, nm)]
    dets = _per_filter(dets, nm)
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
        y0 = rx_block(wsig, channel, p, N_c, prof, start_sample=b * N_c)
        if np.any(amp > 0):
            w = _block_noise(seed, b, N_c)
        for s in range(ns):
            y = y0 if amp[s] == 0 else y0 + amp[s] * w
            for m in range(nm):
                f = F[m][s]
                ap_ = disc.predicted_alpha(f.a_hat, dp, ks[m])
                c = dets[m](y, np.array([ap_ + delta, ap_ - delta]), dp,
                            N_seg=N_seg)
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


def open_loop_det(params: SystemParams, waveform, channel: MultipathChannel,
                  profile, a_hats, n_blocks: int,
                  snrs: Sequence[float] = (float("inf"),), k: int = 1,
                  dets=(cbsfs.cost_function,), N_c: int = 24000,
                  N_seg: int = 2048, delta: float | None = None,
                  seed: int = 0, start_block: int = 0) -> np.ndarray:
    """e[det, snr, block, a_hat] on fixed blocks (no loop)."""
    prof = as_profile(profile)
    delta = 0.25 / N_seg if delta is None else float(delta)
    p = params.with_(a_true=float(prof.a(0.0)), snr_db=float("inf"))
    wsig = waveform.signal(p, seed)
    dp = detector_params(p, wsig.P_samp)
    snrs = tuple(float(s) for s in snrs)
    amp = _noise_amp(p, snrs)
    a_hats = np.atleast_1d(np.asarray(a_hats, dtype=float))
    ap = disc.predicted_alpha(a_hats, dp, k)
    alphas = np.concatenate([ap + delta, ap - delta])
    nj = a_hats.size
    e = np.empty((len(dets), len(snrs), n_blocks, nj))
    for i in range(n_blocks):
        b = start_block + i
        y0 = rx_block(wsig, channel, p, N_c, prof, start_sample=b * N_c)
        if np.any(amp > 0):
            w = _block_noise(seed, b, N_c)
        for s in range(len(snrs)):
            y = y0 if amp[s] == 0 else y0 + amp[s] * w
            for d, fn in enumerate(dets):
                c = fn(y, alphas, dp, N_seg=N_seg)
                e[d, s, i] = disc.error_from_costs(c[:nj], c[nj:])
    return e
