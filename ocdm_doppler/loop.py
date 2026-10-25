"""
First-order Doppler tracking loop on the early-late discriminator.

Algorithm 1 of the Step-3 report: one update per block of N_c received
samples,

    e[i]       = early_late(block i, a_hat[i])        (discriminator.py)
    a_hat[i+1] = a_hat[i] + mu * e[i]

Near the truth e ~ K_d (a - a_hat) + noise, so with g = mu K_d the error
obeys a_hat - a <- (1 - g)(a_hat - a) + mu * noise: a first-order loop with
pole p = 1 - g, 95 % convergence after n95 = ln(0.05) / ln(p) blocks and a
steady-state variance (sigma_e / K_d)^2 g / (2 - g) for white detector noise.

The loop runs on consecutive, non-overlapping blocks of ONE long received
signal: block i is OCDMSignal.rx_nominal(N_c, start_sample = i N_c), so the
symbol timing and the Doppler stretch continue across blocks. Two things are
generated lazily so that a signal of ~10^6 OCDM symbols fits in memory:
  * the data symbols, in seeded chunks (LazySymbols), and
  * the additive noise, fresh per block from a seeded stream (CSCG, power
    P_n per nominal sample -- statistically identical to the T_s-grid noise
    of OCDMSignal, which would otherwise be cached for the whole signal).
Several SNRs can be run in lockstep on the same blocks (same data, the same
unit noise scaled by sqrt(P_n)): each SNR has its own a_hat trajectory.

TIME-VARYING DOPPLER: signal_model.rx takes a constant a, and evaluating it
with a different a per block would make the symbol timing jump at block
boundaries. A time-varying a therefore needs a change in signal_model.py
(an integrated Doppler, u(t) = t + int_0^t a - tau_p0), which the Step-1
modules do not allow without approval. `run_loop` accepts only a constant
a for now and raises NotImplementedError otherwise.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .params import SystemParams
from .signal_model import OCDMSignal, draw_symbols
from . import cbsfs
from . import discriminator as disc

_SYM_CHUNK = 4096


class LazySymbols:
    """D[m, k] for an unbounded symbol index m, generated in seeded chunks.

    Supports the indexing OCDMSignal._chirp_sum uses: D[m_array, :].
    """

    def __init__(self, params: SystemParams, seed: int, cache_chunks: int = 4):
        self.p = params
        self.seed = int(seed)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._max = cache_chunks

    def _chunk(self, c: int) -> np.ndarray:
        blk = self._cache.get(c)
        if blk is None:
            rng = np.random.default_rng([self.seed, 0x5D, c])
            blk = draw_symbols(self.p, _SYM_CHUNK, rng)
            self._cache[c] = blk
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(c)
        return blk

    def __getitem__(self, key):
        m, cols = key
        m = np.asarray(m, dtype=np.int64)
        out = np.empty(m.shape + (self.p.N_is,), dtype=complex)
        cid = m // _SYM_CHUNK
        for c in np.unique(cid):
            sel = cid == c
            out[sel] = self._chunk(int(c))[m[sel] - c * _SYM_CHUNK]
        return out[..., cols]


class LongSignal(OCDMSignal):
    """OCDMSignal with lazily generated data symbols (unbounded length).

    Only the noiseless waveform is used (`rx_nominal(..., noiseless=True)`);
    the loop adds its own per-block noise.
    """

    def __init__(self, params: SystemParams, seed: int):
        self.p = params
        self.n_symbols = 2 ** 60
        self.D = LazySymbols(params, seed)
        self._noise = None


@dataclass
class LoopResult:
    a_hat: np.ndarray
    """a_hat[s, i], i = 0 .. n_blocks (a_hat[s, 0] = a_hat0), per SNR s."""
    e: np.ndarray
    """e[s, i], i = 0 .. n_blocks - 1."""
    snrs: tuple
    a: float
    mu: float


def run_loop(params: SystemParams, a, mu: float, n_blocks: int,
             snrs: Sequence[float] = (float("inf"),), a_hat0: float = 0.0,
             N_c: int = 24000, N_seg: int = 2048, k: int = 1,
             delta: float | None = None, seed: int = 0,
             start_block: int = 0) -> LoopResult:
    """Run the first-order loop over `n_blocks` consecutive blocks.

    `params` sets everything except the Doppler and the SNR (in particular
    the lag set, via params.with_(cbsfs_deltas=...)). All SNRs in `snrs`
    see the same data and the same unit noise.
    """
    if callable(a) or np.ndim(a) != 0:
        raise NotImplementedError(
            "time-varying a needs a signal_model.py change (integrated "
            "Doppler); not allowed without approval -- see loop.py")
    a = float(a)
    delta = 0.25 / N_seg if delta is None else float(delta)
    p = params.with_(a_true=a, snr_db=float("inf"))
    sig = LongSignal(p, seed)
    snrs = tuple(float(s) for s in snrs)
    amp = np.array([0.0 if np.isinf(s) else
                    np.sqrt(p.sigma_D2 / 10 ** (s / 10) / 2.0) for s in snrs])
    a_hat = np.empty((len(snrs), n_blocks + 1))
    e = np.empty((len(snrs), n_blocks))
    a_hat[:, 0] = a_hat0
    for i in range(n_blocks):
        b = start_block + i
        y0 = sig.rx_nominal(N_c, start_sample=b * N_c, noiseless=True)
        if np.any(amp > 0):
            rng = np.random.default_rng([seed, 0xA015E, b])
            w = rng.standard_normal(N_c) + 1j * rng.standard_normal(N_c)
        for s in range(len(snrs)):
            y = y0 if amp[s] == 0 else y0 + amp[s] * w
            ap_ = disc.predicted_alpha(a_hat[s, i], p, k)
            c = cbsfs.cost_function(y, np.array([ap_ + delta, ap_ - delta]),
                                    p, N_seg=N_seg)
            e[s, i] = disc.error_from_costs(c[0], c[1])
            a_hat[s, i + 1] = a_hat[s, i] + mu * e[s, i]
    return LoopResult(a_hat, e, snrs, a, mu)


def n95(pole: float) -> float:
    """Blocks to 95 % convergence of a first-order loop with pole p."""
    return np.log(0.05) / np.log(pole)


def steady_state_var(sigma_e_over_Kd: float, g: float) -> float:
    """Linear prediction of var(a_hat) for white detector noise."""
    return sigma_e_over_Kd ** 2 * g / (2.0 - g)
