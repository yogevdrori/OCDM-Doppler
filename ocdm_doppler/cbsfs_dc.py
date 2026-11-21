"""
CB-SFS cost function with the DC line removed (detector variant, Step 7).

Identical to cbsfs.cost_function ([CBSFS] Eqns. 27-29), except that the
block mean of each lag product z[k] = Y[k] Y*[k - Delta] (for Delta = 0:
|Y[k]|^2) is subtracted before the segmented CAF. The mean of z is the
alpha = 0 spectral line (the mean power, plus the noise power for
Delta = 0); its leakage through the rectangular N_seg window into the two
probes of the early-late detector is a candidate cause of the detector bias
(results/nseg_leakage.md). cbsfs.py is unchanged; with remove_dc=False
this function reproduces cbsfs.cost_function bit for bit
(check_dc_removal.py).
"""

from __future__ import annotations

import numpy as np

from .params import SystemParams
from .cbsfs import autocorrelated_caf, _ALPHA_CHUNK


def segmented_caf_z(z: np.ndarray, alphas: np.ndarray,
                    N_seg: int) -> np.ndarray:
    """cbsfs.segmented_caf for a given lag product z (same arithmetic)."""
    N_c = z.size
    n_pos = N_c - N_seg + 1
    if n_pos <= 0:
        raise ValueError(f"signal too short: N_c={N_c} < N_seg={N_seg}")
    k = np.arange(N_c)
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    out = np.empty((alphas.size, n_pos), dtype=complex)
    for s in range(0, alphas.size, _ALPHA_CHUNK):
        a_chunk = alphas[s:s + _ALPHA_CHUNK][:, None]
        w = z[None, :] * np.exp(-2j * np.pi * a_chunk * k[None, :])
        csum = np.concatenate(
            [np.zeros((w.shape[0], 1), dtype=complex), np.cumsum(w, axis=1)],
            axis=1)
        out[s:s + a_chunk.shape[0]] = (csum[:, N_seg:] - csum[:, :n_pos]) / N_seg
    return out


def cost_function_dc(y: np.ndarray, alphas: np.ndarray, params: SystemParams,
                     N_seg: int | None = None,
                     deltas: np.ndarray | None = None,
                     N_lag: int | None = None,
                     remove_dc: bool = True) -> np.ndarray:
    """|C(alpha)| as cbsfs.cost_function, with the block mean of each lag
    product subtracted (remove_dc=True)."""
    N_seg = params.N_seg if N_seg is None else int(N_seg)
    N_lag = N_seg if N_lag is None else int(N_lag)
    deltas = params.deltas if deltas is None else np.asarray(deltas, dtype=int)
    y = np.asarray(y)
    N_c = y.size
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    total = np.zeros(alphas.size, dtype=complex)
    for d in deltas:
        d = int(d)
        z = np.zeros(N_c, dtype=complex)
        z[d:] = y[d:] * np.conj(y[:N_c - d])
        if remove_dc:
            z[d:] = z[d:] - z[d:].mean()
        R = segmented_caf_z(z, alphas, N_seg)
        total += autocorrelated_caf(R, N_lag, N_c)
    return np.abs(total)
