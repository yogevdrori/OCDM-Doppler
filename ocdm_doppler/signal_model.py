"""
OCDM transmitter and single-path Doppler receiver.

Implements, directly and without resampling:

  * the complex baseband OCDM signal X(t)          -- [OCDM] Sec. 2
  * the received baseband signal Y(t), P = 1       -- [OCDM] Eqn. (3)

Because [OCDM] gives a closed-form expression for Y(t), the received signal
can be evaluated at *any* time instant. We therefore never resample: we
evaluate Y exactly where the estimator asks for it. This is both faster and
free of interpolation error.

Noise is generated block-wise from a counter-based stream, so Y(n * T_s) is
reproducible and consistent no matter in which order, or how many times, a
given sample index is requested.
"""

from __future__ import annotations

import numpy as np

from .params import SystemParams

_BLOCK = 1 << 16
_CHUNK = 1 << 18  # max time samples processed at once in the chirp sum


class _BlockNoise:
    """Deterministic CSCG noise indexed by (possibly negative) sample index.

    Noise is generated in fixed-size blocks; each block is seeded from
    (seed, block_id), so values are stable across calls and across growth in
    either direction.
    """

    def __init__(self, power: float, seed: int):
        self.power = float(power)
        self.seed = int(seed)
        self._cache: dict[int, np.ndarray] = {}

    def _block(self, block_id: int) -> np.ndarray:
        blk = self._cache.get(block_id)
        if blk is None:
            rng = np.random.default_rng([self.seed, block_id & 0xFFFFFFFF,
                                         (block_id >> 32) & 0xFFFFFFFF])
            # CSCG with E{|n|^2} = power
            re = rng.standard_normal(_BLOCK)
            im = rng.standard_normal(_BLOCK)
            blk = np.sqrt(self.power / 2.0) * (re + 1j * im)
            self._cache[block_id] = blk
        return blk

    def at(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if self.power == 0.0:
            return np.zeros(idx.shape, dtype=complex)
        out = np.empty(idx.shape, dtype=complex)
        block_ids = np.floor_divide(idx, _BLOCK)
        offsets = idx - block_ids * _BLOCK
        for bid in np.unique(block_ids):
            sel = block_ids == bid
            out[sel] = self._block(int(bid))[offsets[sel]]
        return out


def draw_symbols(params: SystemParams, n_symbols: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Draw i.i.d. zero-mean constellation symbols D[m, k].

    Satisfies [OCDM] Eqn. (1)-(2): E{D} = 0, E{|D|^2} = sigma_D^2,
    E{D * D} = 0 (proper complex, for QPSK).
    """
    shape = (n_symbols, params.N_is)
    amp = np.sqrt(params.sigma_D2)
    if params.modulation == "bpsk":
        return amp * (2.0 * rng.integers(0, 2, shape) - 1.0).astype(complex)
    if params.modulation == "qpsk":
        bits = rng.integers(0, 4, shape)
        return amp * np.exp(1j * (np.pi / 4.0 + bits * np.pi / 2.0))
    raise ValueError(f"unknown modulation {params.modulation!r}")


class OCDMSignal:
    """Transmitted and received OCDM waveform for one realisation.

    Parameters
    ----------
    params
        System configuration.
    n_symbols
        Number of OCDM symbols to generate, i.e. the valid range of the
        symbol index m is [0, n_symbols). Requests outside this range yield
        zero (the waveform is treated as switched off there).
    seed
        Overrides `params.seed` when given (used by the Monte-Carlo runner).
    """

    def __init__(self, params: SystemParams, n_symbols: int,
                 seed: int | None = None):
        self.p = params
        self.n_symbols = int(n_symbols)
        s = params.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(s)
        self.D = draw_symbols(params, self.n_symbols, self._rng)
        self._noise = _BlockNoise(params.noise_power, seed=s ^ 0x5EED)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _chirp_sum(self, r: np.ndarray, m: np.ndarray,
                   active: np.ndarray) -> np.ndarray:
        """sum_k D[m, k] * exp(-j pi N_is / T_is^2 * (r - k T_is / N_is)^2).

        `r` is the within-symbol time (already reduced modulo T_sym) and
        `m` the symbol index; `active` marks the entries that lie inside the
        pulse-shaping support [0, T_is).
        """
        p = self.p
        out = np.zeros(r.shape, dtype=complex)
        if not np.any(active):
            return out

        idx_act = np.flatnonzero(active)
        k = np.arange(p.N_is)
        k_off = k * p.T_is / p.N_is
        coef = -1j * np.pi * p.N_is / (p.T_is ** 2)

        for start in range(0, idx_act.size, _CHUNK):
            sl = idx_act[start:start + _CHUNK]
            dr = r[sl][:, None] - k_off[None, :]          # (n, N_is)
            phase = np.exp(coef * dr ** 2)
            out[sl] = np.einsum("nk,nk->n", self.D[m[sl], :], phase)
        return out

    def _decompose(self, u: np.ndarray):
        """Map effective time u to (symbol index m, within-symbol time r, active)."""
        p = self.p
        m = np.floor(u / p.T_sym).astype(np.int64)
        r = u - m * p.T_sym
        active = (r >= 0.0) & (r < p.T_is) & (m >= 0) & (m < self.n_symbols)
        m_safe = np.clip(m, 0, max(self.n_symbols - 1, 0))
        return m_safe, r, active

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def tx(self, t: np.ndarray) -> np.ndarray:
        """Transmitted complex baseband signal X(t), [OCDM] Sec. 2."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        m, r, active = self._decompose(t)
        s = self._chirp_sum(r, m, active)
        return np.exp(1j * np.pi / 4.0) * s / np.sqrt(self.p.N_is)

    def rx(self, t: np.ndarray, a: float | None = None,
           noiseless: bool = False,
           noise_index: np.ndarray | None = None) -> np.ndarray:
        """Received complex baseband signal Y(t), [OCDM] Eqn. (3) with P = 1.

        Parameters
        ----------
        t
            Time instants [s].
        a
            Doppler scaling factor; defaults to `params.a_true_eff`.
        noiseless
            If True, omit the additive noise term.
        noise_index
            Integer sample indices used to look up the noise realisation.
            The estimator passes the T_s-grid index so that repeated
            requests for the same grid point see the same noise sample.
        """
        p = self.p
        a = p.a_true_eff if a is None else float(a)
        t = np.atleast_1d(np.asarray(t, dtype=float))

        u = (1.0 + a) * t - p.tau_p0
        m, r, active = self._decompose(u)
        s = self._chirp_sum(r, m, active)

        front = (p.A_p0
                 * np.exp(-2j * np.pi * p.f_Tx * p.tau_p0)
                 / (2.0 * np.sqrt(p.N_is)))
        carrier = np.exp(1j * (2.0 * np.pi * p.Delta_f_of_a(a) * t + p.Delta_phi))
        y = front * carrier * s

        if not noiseless and p.noise_power > 0.0:
            if noise_index is None:
                noise_index = np.rint(t / p.T_s).astype(np.int64)
            y = y + self._noise.at(noise_index)
        return y

    def rx_at_grid(self, idx: np.ndarray, a: float | None = None,
                   noiseless: bool = False) -> np.ndarray:
        """Y evaluated on the fine grid: Y(idx * T_s).

        This is the entry point the estimators use, because the algorithm
        in the results documents quantises every requested time instant to
        the T_s grid via  floor(t / T_s + 1/2).
        """
        idx = np.asarray(idx, dtype=np.int64)
        return self.rx(idx * self.p.T_s, a=a, noiseless=noiseless,
                       noise_index=idx)

    def grid_index(self, t: np.ndarray) -> np.ndarray:
        """Quantise time instants to the fine grid: round(t / T_s)."""
        return np.rint(np.asarray(t, dtype=float) / self.p.T_s).astype(np.int64)

    def rx_nominal(self, n: int, a: float | None = None,
                   noiseless: bool = False,
                   start_sample: int = 0) -> np.ndarray:
        """Y sampled at the receiver's nominal rate: Y(n * T_samp_ns).

        This is the discrete-time signal Y_DT[n] that CB-SFS operates on
        ([CBSFS] Sec. IV-B). The time instants are exact -- they are *not*
        quantised to the fine T_s grid, because that quantisation would
        introduce a spurious sampling-rate error of order 1/rho, which is the
        same order as the Doppler values under study.

        Noise is still indexed by the nearest T_s grid point so that it stays
        consistent with the fine-grid evaluations used for AF estimation.
        """
        p = self.p
        k = start_sample + np.arange(int(n), dtype=np.int64)
        t = k * p.T_samp_ns
        return self.rx(t, a=a, noiseless=noiseless,
                       noise_index=self.grid_index(t))

    def rx_uniform(self, n: int, start_index: int = 0,
                   step: int = 1, a: float | None = None,
                   noiseless: bool = False) -> np.ndarray:
        """`n` samples of Y on a uniform T_s-grid with the given step."""
        idx = start_index + step * np.arange(n, dtype=np.int64)
        return self.rx_at_grid(idx, a=a, noiseless=noiseless)


def chirp_matrix(N: int) -> np.ndarray:
    """IDFnT matrix whose columns are the orthonormal OCDM chirps.

    psi_k[n] = (1/sqrt(N)) e^{j pi/4} e^{-j pi/N (n - k)^2}

    Provided for verification (unitarity, chirp-FFT-chirp factorisation);
    the waveform generator above works in continuous time instead.
    """
    n = np.arange(N)[:, None]
    k = np.arange(N)[None, :]
    return (1.0 / np.sqrt(N)) * np.exp(1j * np.pi / 4.0) * np.exp(
        -1j * np.pi / N * (n - k) ** 2)
