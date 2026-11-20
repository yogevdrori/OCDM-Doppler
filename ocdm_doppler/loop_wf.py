"""
Doppler tracking loop and open-loop detector for any waveform and channel.

`run_loop_wf` is loop_tv2.run_loop_filters with the received block taken
from a waveform (waveforms.py) through a channel (channel.py), and the
detector's cycle frequency k (1 + a_hat) / P_samp from the waveform's
mean-power period (wf_params.WaveformParams). Everything else is the same:
one long received signal, the per-block noise stream ([seed, 0xA015E,
block], amplitude sqrt(P_n / 2) per component), the detector call
(cbsfs.cost_function at alpha_p +/- delta, discriminator.error_from_costs),
the Step 6 filter objects, all filters x SNRs in lockstep, and `states` to
continue a run. Each filter may use its own harmonic k (`ks`), so a sweep
over k runs on the same blocks. With OCDMWaveform, channel.single_path and
k = 1 it reproduces run_loop_filters bit for bit (verify_step7.py).

`open_loop_e` evaluates the discriminator output at given estimates a_hat
on fixed blocks (no loop): one cost_function call per block and SNR, all
harmonics and estimates together. With one harmonic it is exactly
discriminator.early_late on the same block.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .doppler_profile import as_profile
from .channel import MultipathChannel, rx_block
from .loop_tv2 import FilterLoopResult
from .params import SystemParams
from .wf_params import detector_params
from . import cbsfs
from . import discriminator as disc


def _noise_amp(p, snrs):
    return np.array([0.0 if np.isinf(s) else
                     np.sqrt(p.sigma_D2 / 10 ** (s / 10) / 2.0) for s in snrs])


def _block_noise(seed, b, N_c):
    rng = np.random.default_rng([seed, 0xA015E, b])
    return rng.standard_normal(N_c) + 1j * rng.standard_normal(N_c)


def run_loop_wf(params: SystemParams, waveform, channel: MultipathChannel,
                profile, filters, n_blocks: int,
                snrs: Sequence[float] = (float("inf"),), N_c: int = 24000,
                N_seg: int = 2048, ks=1, delta: float | None = None,
                seed: int = 0, start_block: int = 0,
                states=None) -> FilterLoopResult:
    """Every filter (copied once per SNR) on `n_blocks` consecutive blocks.
    `ks`: one harmonic for all filters, or one per filter."""
    prof = as_profile(profile)
    delta = 0.25 / N_seg if delta is None else float(delta)
    p = params.with_(a_true=float(prof.a(0.0)), snr_db=float("inf"))
    wsig = waveform.signal(p, seed)
    dp = detector_params(p, wsig.P_samp)
    snrs = tuple(float(s) for s in snrs)
    amp = _noise_amp(p, snrs)
    nm, ns = len(filters), len(snrs)
    ks = [int(ks)] * nm if np.ndim(ks) == 0 else [int(k) for k in ks]
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
                c = cbsfs.cost_function(
                    y, np.array([ap_ + delta, ap_ - delta]), dp, N_seg=N_seg)
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


def open_loop_e(params: SystemParams, waveform, channel: MultipathChannel,
                profile, a_hats, n_blocks: int,
                snrs: Sequence[float] = (float("inf"),), ks=(1,),
                N_c: int = 24000, N_seg: int = 2048,
                delta: float | None = None, seed: int = 0,
                start_block: int = 0, return_costs: bool = False):
    """Discriminator output e[s, i, k_index, j] at a_hat = a_hats[j] for
    blocks i = 0 .. n_blocks - 1 (no loop). With return_costs, also the
    costs C+[s, i, k, j] and C-[...]."""
    prof = as_profile(profile)
    delta = 0.25 / N_seg if delta is None else float(delta)
    p = params.with_(a_true=float(prof.a(0.0)), snr_db=float("inf"))
    wsig = waveform.signal(p, seed)
    dp = detector_params(p, wsig.P_samp)
    snrs = tuple(float(s) for s in snrs)
    amp = _noise_amp(p, snrs)
    a_hats = np.atleast_1d(np.asarray(a_hats, dtype=float))
    ks = [int(k) for k in np.atleast_1d(ks)]
    nk, nj = len(ks), a_hats.size
    ap = np.concatenate([disc.predicted_alpha(a_hats, dp, k) for k in ks])
    alphas = np.concatenate([ap + delta, ap - delta])
    e = np.empty((len(snrs), n_blocks, nk, nj))
    cp = np.empty_like(e)
    cm = np.empty_like(e)
    for i in range(n_blocks):
        b = start_block + i
        y0 = rx_block(wsig, channel, p, N_c, prof, start_sample=b * N_c)
        if np.any(amp > 0):
            w = _block_noise(seed, b, N_c)
        for s in range(len(snrs)):
            y = y0 if amp[s] == 0 else y0 + amp[s] * w
            c = cbsfs.cost_function(y, alphas, dp, N_seg=N_seg)
            n2 = nk * nj
            cp[s, i] = c[:n2].reshape(nk, nj)
            cm[s, i] = c[n2:].reshape(nk, nj)
            e[s, i] = disc.error_from_costs(cp[s, i], cm[s, i])
    return (e, cp, cm) if return_costs else e
