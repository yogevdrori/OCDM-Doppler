"""
Waveform layer: the transmitted continuous-time signal X(t) of different
multi-basis modulations, behind one interface.

A `Waveform` is a specification; `waveform.signal(params, seed)` returns a
per-realisation signal object (unbounded record, data generated lazily from
a seed) with

    tx(u)            X(u), exact at any time instants u [s] (0 for u < 0)
    P_samp           period of the mean power E|X|^2 in samples (sets the
                     cycle frequencies k (1 + a) / P_samp of the detector)
    mean_power(u)    E|X(u)|^2 (the on/off pattern; sigma_D^2 when active)
    autocorr(t1, t2) E[X(t1) X*(t2)] over the data (for multipath checks)

All waveforms have the same sample rate (params.f_samp_syn), bandwidth and
power per active sample (sigma_D^2), so the SNR sigma_D^2 / P_n means the
same for all of them.

OCDMWaveform  wraps the existing OCDM signal (loop.LongSignal: the OCDM
              transmitter of signal_model.py with lazily generated data), so
              X(t) is exactly OCDMSignal.tx.
OTFSWaveform  M x N delay-Doppler grid per frame, ISFFT to the time-frequency
              grid (unitary), Heisenberg (OFDM-type) modulator with a
              rectangular pulse and centred subcarriers:
                  x(t) = (1/sqrt(M)) sum_m X_tf[m, n] exp(j 2 pi (m - M/2) r / T_is)
              for 0 <= r < T_is in slot n (T_is = M T_samp). Guard
              "symbol": N_gi zeros after every slot (period M + N_gi, as
              OCDM); "frame": the N slots back to back, then N N_gi zeros
              (same overhead, period N (M + N_gi)).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .params import SystemParams
from .signal_model import draw_symbols
from .loop import LongSignal

_CHUNK = 1 << 16          # max samples evaluated at once
_FRAME_CHUNK = 256        # OTFS frames per lazily generated data chunk


class Waveform:
    name = "waveform"

    def signal(self, params: SystemParams, seed: int):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OCDM
# ---------------------------------------------------------------------------
class OCDMWaveform(Waveform):
    name = "OCDM"

    def signal(self, params, seed):
        return OCDMWaveSignal(params, seed)


class OCDMWaveSignal:
    """OCDM through the waveform interface: X(t) = LongSignal.tx (the
    existing transmitter, data from loop.LazySymbols with the same seed)."""

    def __init__(self, params: SystemParams, seed: int):
        self.p = params
        self.long = LongSignal(params, seed)
        self.P_samp = params.N_is + params.N_gi_eff

    def tx(self, u):
        return self.long.tx(u)

    def mean_power(self, u):
        _, _, active = self.long._decompose(np.atleast_1d(
            np.asarray(u, dtype=float)))
        return self.p.sigma_D2 * active

    def autocorr(self, t1, t2):
        p = self.p
        m1, r1, a1 = self.long._decompose(np.atleast_1d(np.asarray(t1, float)))
        m2, r2, a2 = self.long._decompose(np.atleast_1d(np.asarray(t2, float)))
        k_off = np.arange(p.N_is) * p.T_is / p.N_is
        coef = -1j * np.pi * p.N_is / p.T_is ** 2
        ph = (coef * ((r1[..., None] - k_off) ** 2
                      - (r2[..., None] - k_off) ** 2))
        val = p.sigma_D2 / p.N_is * np.exp(ph).sum(-1)
        return np.where(a1 & a2 & (m1 == m2), val, 0.0)


# ---------------------------------------------------------------------------
# OTFS
# ---------------------------------------------------------------------------
class OTFSWaveform(Waveform):
    def __init__(self, M: int = 16, N: int = 16, N_gi: int = 4,
                 guard: str = "symbol"):
        if guard not in ("symbol", "frame"):
            raise ValueError("guard must be 'symbol' or 'frame'")
        self.M, self.N, self.N_gi, self.guard = int(M), int(N), int(N_gi), guard
        self.name = f"OTFS-{guard}"

    @property
    def frame_len(self) -> int:
        """Frame length in samples (same for both guard structures)."""
        return self.N * (self.M + self.N_gi)

    @property
    def P_samp(self) -> int:
        return self.M + self.N_gi if self.guard == "symbol" else self.frame_len

    def signal(self, params, seed):
        return OTFSSignal(self, params, seed)


def isfft(X_dd: np.ndarray) -> np.ndarray:
    """(..., M, N) delay-Doppler -> time-frequency, unitary:
    X_tf[m, n] = (1/sqrt(MN)) sum_l sum_k X_dd[l, k] e^{j2pi(nk/N - ml/M)}."""
    return np.fft.ifft(np.fft.fft(X_dd, axis=-2, norm="ortho"), axis=-1,
                       norm="ortho")


def sfft(X_tf: np.ndarray) -> np.ndarray:
    """Inverse of isfft."""
    return np.fft.fft(np.fft.ifft(X_tf, axis=-2, norm="ortho"), axis=-1,
                      norm="ortho")


class OTFSSignal:
    """One OTFS realisation, unbounded, data in seeded chunks of frames."""

    def __init__(self, wf: OTFSWaveform, params: SystemParams, seed: int,
                 cache_chunks: int = 4):
        self.wf, self.p, self.seed = wf, params, int(seed)
        self.M, self.N, self.N_gi = wf.M, wf.N, wf.N_gi
        self.P_samp = wf.P_samp
        T = params.T_samp_syn
        self.T = T
        self.T_is = self.M * T
        self.T_slot = (self.M + self.N_gi) * T
        self.T_frame = wf.frame_len * T
        self._cache: OrderedDict[int, tuple] = OrderedDict()
        self._max = cache_chunks
        self._pdata = params.with_(N_is=self.M * self.N)

    # -- data ---------------------------------------------------------------
    def _chunk(self, c: int):
        blk = self._cache.get(c)
        if blk is None:
            rng = np.random.default_rng([self.seed, 0x07F5, c])
            X_dd = draw_symbols(self._pdata, _FRAME_CHUNK, rng).reshape(
                _FRAME_CHUNK, self.M, self.N)
            blk = (X_dd, isfft(X_dd))
            self._cache[c] = blk
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(c)
        return blk

    def data(self, frames) -> np.ndarray:
        """X_dd[frame, l, k] (delay l, Doppler k) for the given frames."""
        frames = np.atleast_1d(np.asarray(frames, dtype=np.int64))
        out = np.empty((frames.size, self.M, self.N), dtype=complex)
        cid = frames // _FRAME_CHUNK
        for c in np.unique(cid):
            sel = cid == c
            out[sel] = self._chunk(int(c))[0][frames[sel] - c * _FRAME_CHUNK]
        return out

    def _coef(self, f: np.ndarray, n: np.ndarray) -> np.ndarray:
        """X_tf[f, :, n] -> (len, M)."""
        out = np.empty((f.size, self.M), dtype=complex)
        cid = f // _FRAME_CHUNK
        for c in np.unique(cid):
            sel = cid == c
            X_tf = self._chunk(int(c))[1]
            out[sel] = X_tf[f[sel] - c * _FRAME_CHUNK, :, n[sel]]
        return out

    # -- timing -------------------------------------------------------------
    def _decompose(self, u):
        """u -> (frame f, slot n, time in slot r, active)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        f = np.floor(u / self.T_frame).astype(np.int64)
        w = u - f * self.T_frame
        if self.wf.guard == "symbol":
            n = np.floor(w / self.T_slot).astype(np.int64)
            r = w - n * self.T_slot
            active = (r >= 0.0) & (r < self.T_is)
        else:
            n = np.floor(w / self.T_is).astype(np.int64)
            r = w - n * self.T_is
            active = (r >= 0.0) & (r < self.T_is) & (n < self.N)
        active &= (f >= 0) & (n >= 0) & (n < self.N)
        return f, np.clip(n, 0, self.N - 1), r, active

    # -- interface ----------------------------------------------------------
    def tx(self, u):
        f, n, r, active = self._decompose(u)
        out = np.zeros(r.shape, dtype=complex)
        idx = np.flatnonzero(active)
        freqs = (np.arange(self.M) - self.M / 2.0) / self.T_is
        for s in range(0, idx.size, _CHUNK):
            sl = idx[s:s + _CHUNK]
            ph = np.exp(2j * np.pi * r[sl][:, None] * freqs[None, :])
            out[sl] = np.einsum("nm,nm->n", self._coef(f[sl], n[sl]), ph)
        return out / np.sqrt(self.M)

    def mean_power(self, u):
        return self.p.sigma_D2 * self._decompose(u)[3]

    def autocorr(self, t1, t2):
        f1, n1, r1, a1 = self._decompose(t1)
        f2, n2, r2, a2 = self._decompose(t2)
        freqs = (np.arange(self.M) - self.M / 2.0) / self.T_is
        val = self.p.sigma_D2 / self.M * np.exp(
            2j * np.pi * (r1 - r2)[..., None] * freqs).sum(-1)
        return np.where(a1 & a2 & (f1 == f2) & (n1 == n2), val, 0.0)

    # -- demodulation (checks) ----------------------------------------------
    def active_sample_offsets(self) -> np.ndarray:
        """(N, M) sample offsets of the active samples within a frame."""
        n = np.arange(self.N)[:, None]
        i = np.arange(self.M)[None, :]
        slot = self.M + self.N_gi if self.wf.guard == "symbol" else self.M
        return n * slot + i

    def demod(self, y: np.ndarray) -> np.ndarray:
        """y[frame, n, i] (active samples, timing/carrier/gain removed)
        -> X_dd estimate [frame, l, k]."""
        sign = (-1.0) ** np.arange(self.M)          # e^{j pi i}
        X_tf = np.fft.fft(y * sign, axis=-1, norm="ortho")   # [f, n, m]
        return sfft(np.swapaxes(X_tf, -1, -2))
