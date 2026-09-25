"""
CB-SFS: cyclostationarity-based symbol-rate / Doppler estimation.

Implements [CBSFS] Eqns. (27)-(34):

    (27)  segmented CAF          R^alpha[n, Delta]_{N_seg}
    (28)  autocorrelated CAF     Rbar[alpha, Delta; N_lag]
    (29)  cost function          C(alpha) = sum_Delta Rbar[alpha, Delta]
    (31)  cycle-frequency peaks  local maxima above a threshold
    (33)  fundamental CF         alpha_1 = alpha_{N_last} / N_last
    (34)  sampling interval      T_samp^(sync) = (alpha_1 / alpha_1_hat) T_samp^(ns)

DESIGN NOTE (important for the closed-loop work that follows)
------------------------------------------------------------
`cost_function` accepts an arbitrary array of cycle frequencies, not just a
uniform grid. The planned early-late discriminator evaluates the cost at two
cycle frequencies placed symmetrically around the position predicted by the
current Doppler estimate, so it can call this function directly with two
alpha values and nothing here needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import SystemParams

_ALPHA_CHUNK = 128


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------
def segmented_caf(y: np.ndarray, alphas: np.ndarray, delta: int,
                  N_seg: int) -> np.ndarray:
    """Segmented CAF, [CBSFS] Eqn. (27).

    Returns an array of shape (n_alpha, n_positions) holding
    R^alpha[n, delta]_{N_seg} for n = 0 .. N_c - N_seg.
    """
    y = np.asarray(y)
    N_c = y.size
    n_pos = N_c - N_seg + 1
    if n_pos <= 0:
        raise ValueError(f"signal too short: N_c={N_c} < N_seg={N_seg}")

    # z[k] = Y[k] Y*[k - delta]; zero where k - delta < 0
    z = np.zeros(N_c, dtype=complex)
    z[delta:] = y[delta:] * np.conj(y[:N_c - delta])

    k = np.arange(N_c)
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    out = np.empty((alphas.size, n_pos), dtype=complex)

    for s in range(0, alphas.size, _ALPHA_CHUNK):
        a_chunk = alphas[s:s + _ALPHA_CHUNK][:, None]          # (A, 1)
        w = z[None, :] * np.exp(-2j * np.pi * a_chunk * k[None, :])
        csum = np.concatenate(
            [np.zeros((w.shape[0], 1), dtype=complex), np.cumsum(w, axis=1)],
            axis=1)
        out[s:s + a_chunk.shape[0]] = (csum[:, N_seg:] - csum[:, :n_pos]) / N_seg
    return out


def autocorrelated_caf(R: np.ndarray, N_lag: int, N_c: int) -> np.ndarray:
    """Autocorrelated CAF, [CBSFS] Eqn. (28).

    `R` has shape (n_alpha, n_positions) as returned by `segmented_caf`.
    Returns shape (n_alpha,).
    """
    n_pos = R.shape[1]
    if n_pos <= N_lag:
        raise ValueError("signal too short for the requested N_lag")
    prod = R[:, N_lag:] * np.conj(R[:, :n_pos - N_lag])
    return prod.sum(axis=1) / N_c


def cost_function(y: np.ndarray, alphas: np.ndarray, params: SystemParams,
                  N_seg: int | None = None,
                  deltas: np.ndarray | None = None,
                  N_lag: int | None = None) -> np.ndarray:
    """CB-SFS cost function C(alpha), [CBSFS] Eqn. (29).

    Evaluated at arbitrary cycle frequencies `alphas`. Peaks of |C| occur at
    the cycle frequencies of the sampled signal, i.e. at k (1 + a) / N_sym.

    Returns the real, non-negative magnitude |C(alpha)|.
    """
    N_seg = params.N_seg if N_seg is None else int(N_seg)
    N_lag = N_seg if N_lag is None else int(N_lag)
    deltas = params.deltas if deltas is None else np.asarray(deltas, dtype=int)

    y = np.asarray(y)
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    total = np.zeros(alphas.size, dtype=complex)
    for d in deltas:
        R = segmented_caf(y, alphas, int(d), N_seg)
        total += autocorrelated_caf(R, N_lag, y.size)
    return np.abs(total)


# ---------------------------------------------------------------------------
# Peak detection and Doppler estimation
# ---------------------------------------------------------------------------
def _parabolic_refine(alphas: np.ndarray, cost: np.ndarray,
                      i: int) -> float:
    """Sub-grid peak location by parabolic interpolation on three points."""
    if i <= 0 or i >= cost.size - 1:
        return float(alphas[i])
    y0, y1, y2 = cost[i - 1], cost[i], cost[i + 1]
    den = y0 - 2.0 * y1 + y2
    if den == 0.0:
        return float(alphas[i])
    frac = 0.5 * (y0 - y2) / den
    frac = float(np.clip(frac, -1.0, 1.0))
    step = float(alphas[1] - alphas[0])
    return float(alphas[i] + frac * step)


def find_cycle_frequencies(alphas: np.ndarray, cost: np.ndarray,
                           threshold_ratio: float = 0.10,
                           refine: bool = True) -> np.ndarray:
    """Cycle-frequency estimates, [CBSFS] Eqn. (31).

    Local maxima of the cost function above `threshold_ratio` times the
    global maximum, optionally refined to sub-grid resolution.

    IMPLEMENTATION NOTE: parabolic refinement is not part of [CBSFS], which
    searches a grid of N_seg points. Without it the accuracy is limited by
    the grid spacing, which is far coarser than the Doppler values of
    interest here, so it is enabled by default. Disable it to reproduce a
    strictly grid-limited search.
    """
    cost = np.asarray(cost, dtype=float)
    thr = threshold_ratio * cost.max()

    interior = np.arange(1, cost.size - 1)
    is_peak = (cost[interior] > cost[interior - 1]) & \
              (cost[interior] >= cost[interior + 1]) & \
              (cost[interior] > thr)
    idx = interior[is_peak]
    if idx.size == 0:
        return np.empty(0)
    if refine:
        return np.array([_parabolic_refine(alphas, cost, int(i)) for i in idx])
    return alphas[idx]


def locate_harmonic(y: np.ndarray, params: SystemParams, k: int,
                    n_alpha: int = 512, half_width_factor: float = 0.25,
                    refine: bool = True,
                    N_seg: int | None = None) -> tuple[float, np.ndarray,
                                                       np.ndarray]:
    """Locate the k-th cycle-frequency harmonic of the received signal.

    Searches the cost function on a window centred at k * alpha_1 and
    +/- `half_width_factor` * alpha_1 wide, so neighbouring harmonics and the
    alpha = 0 skirt cannot be mistaken for the peak of interest.

    This replaces the generic "k-th detected local maximum is harmonic k"
    rule, which is fragile: the Dirichlet sidelobes of one harmonic are
    easily picked up as separate peaks.

    Returns (alpha_k, alphas, cost).
    """
    centre = k * params.alpha_1_syn
    half = half_width_factor * params.alpha_1_syn
    alphas = np.linspace(centre - half, centre + half, int(n_alpha))
    cost = cost_function(y, alphas, params, N_seg=N_seg)
    i = int(np.argmax(cost))
    alpha_k = _parabolic_refine(alphas, cost, i) if refine else float(alphas[i])
    return alpha_k, alphas, cost


@dataclass
class CBSFSResult:
    a_hat: float
    """Doppler estimate  a_hat_sr."""
    alpha_1_hat: float
    """Estimated fundamental cycle frequency of the received signal."""
    N_last: int
    """Index of the highest identified harmonic."""
    peaks: np.ndarray
    alphas: np.ndarray
    cost: np.ndarray


def estimate_doppler(y: np.ndarray, params: SystemParams,
                     n_alpha: int | None = None,
                     threshold_ratio: float = 0.10,
                     refine: bool = True,
                     alpha_margin: float = 0.05,
                     alpha_lo_factor: float = 0.5) -> CBSFSResult:
    """Full CB-SFS Doppler estimate, [CBSFS] steps 1-5.

    The search range is A = (0, alpha_{N_T}], widened by `alpha_margin` so the
    Doppler-shifted harmonics stay inside it, and with its lower end raised to
    `alpha_lo_factor * alpha_1`.

    Raising the lower end matters: the cost function has a large peak at
    alpha = 0 (the stationary component, dominated by the Delta = 0 lag),
    whose skirt has width ~1/N_seg and would otherwise swamp the
    cycle-frequency peaks in any threshold test. [CBSFS] excludes alpha = 0
    from A for the same reason; here we exclude its immediate neighbourhood
    as well, which is legitimate because N_sym -- and hence the approximate
    location of alpha_1 -- is known a priori.

    Returns `a_hat` such that the received signal's fundamental cycle
    frequency is alpha_1 (1 + a_hat), matching
    a_hat_sr = T_samp^(syn) / T_hat_samp^(syn) - 1 of the results documents.
    """
    n_alpha = params.n_alpha_grid if n_alpha is None else int(n_alpha)
    lo = alpha_lo_factor * params.alpha_1_syn
    hi = params.alpha_NT * (1.0 + alpha_margin)
    alphas = np.linspace(lo, hi, n_alpha)

    cost = cost_function(y, alphas, params)
    peaks = find_cycle_frequencies(alphas, cost, threshold_ratio, refine)
    if peaks.size == 0:
        return CBSFSResult(float("nan"), float("nan"), 0, peaks, alphas, cost)

    # Harmonics are integer multiples of the fundamental; the k-th detected
    # peak (ordered in alpha) corresponds to harmonic k.  [CBSFS] Eqn. (33)
    N_last = int(peaks.size)
    alpha_1_hat = float(peaks[-1]) / N_last

    a_hat = alpha_1_hat / params.alpha_1_syn - 1.0
    return CBSFSResult(a_hat, alpha_1_hat, N_last, peaks, alphas, cost)


def estimate_doppler_harmonic(y: np.ndarray, params: SystemParams,
                              N_last: int | None = None,
                              n_alpha: int = 512,
                              refine: bool = True,
                              N_seg: int | None = None) -> CBSFSResult:
    """CB-SFS Doppler estimate using a windowed search for one harmonic.

    Locates harmonic `N_last` (default: params.N_T) directly and applies
    [CBSFS] Eqn. (33), alpha_1 = alpha_{N_last} / N_last.

    Using a high harmonic is what gives this estimator its leverage on small
    Doppler values: harmonic k is displaced by k * a * alpha_1, so the
    absolute displacement grows with k while the peak width stays ~1/N_seg.
    """
    N_last = params.N_T if N_last is None else int(N_last)
    alpha_k, alphas, cost = locate_harmonic(y, params, N_last,
                                            n_alpha=n_alpha, refine=refine,
                                            N_seg=N_seg)
    alpha_1_hat = alpha_k / N_last
    a_hat = alpha_1_hat / params.alpha_1_syn - 1.0
    return CBSFSResult(a_hat, alpha_1_hat, N_last,
                       np.array([alpha_k]), alphas, cost)


def expected_cycle_frequencies(a: float, params: SystemParams,
                               n_harmonics: int | None = None) -> np.ndarray:
    """Cycle frequencies of the received signal: k (1 + a) / N_sym."""
    n = params.N_T if n_harmonics is None else int(n_harmonics)
    k = np.arange(1, n + 1)
    return k * (1.0 + a) / params.N_sym
