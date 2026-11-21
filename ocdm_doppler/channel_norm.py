"""
Normalise a multipath channel to the received signal power of the
single-path reference (Step 7, Part D; Yogev's decision).

With fractional delay differences the paths add partly coherently
(verify_step7.py check 5), so sum |c_p|^2 = 1 does not keep the received
power. The exact mean received signal power, averaged over one period of
the mean power, is

    P_ch = < sum_pq h_p h_q* R_X(t - tau_p, t - tau_q) >_t,
    h_p  = A_p0 c_p exp(-j 2 pi f_Tx tau_p) / 2,

with R_X(t1, t2) = E[X(t1) X*(t2)] from the waveform (`autocorr`). The
reference is one path with c = 1 (P_ref = |A_p0|^2/4 * sigma_D^2 * duty).
The channel is scaled through params.A_p0 -> A_p0 sqrt(P_ref / P_ch), so the
mean received signal power, and hence the effective SNR, is the same for
every channel. Computed at a = 0 (the Doppler stretch over one period is
negligible); per realisation for Rayleigh gains. Paths with their own
Doppler profile (multiscale) drift apart, their cross terms rotate and
average out, so for them only the diagonal terms are used.
"""

from __future__ import annotations

import numpy as np

from .channel import MultipathChannel, Path, single_path
from .params import SystemParams


def mean_received_power(wsig, channel: MultipathChannel, params: SystemParams,
                        n_sub: int = 16, n_skip_periods: int = 10) -> float:
    """Exact mean received signal power over one period (fine time grid of
    n_sub points per sample, several periods into the record)."""
    p = params
    P = wsig.P_samp
    t = (n_skip_periods * P + (np.arange(P * n_sub) + 0.5) / n_sub) \
        * p.T_samp_syn
    hs = [p.A_p0 * complex(q.gain) * np.exp(-2j * np.pi * p.f_Tx * q.delay
                                            * p.T_samp_syn) / 2.0
          for q in channel.paths]
    taus = [q.delay * p.T_samp_syn for q in channel.paths]
    coh = all(q.profile is None for q in channel.paths)
    tot = np.zeros(t.size)
    for i, (hp, tp) in enumerate(zip(hs, taus)):
        for j, (hq, tq) in enumerate(zip(hs, taus)):
            if i != j and not coh:
                continue
            tot += np.real(hp * np.conj(hq) * wsig.autocorr(t - tp, t - tq))
    return float(tot.mean())


def power_factor(wsig, channel: MultipathChannel,
                 params: SystemParams) -> float:
    """G = P_ref / P_ch (1 exactly for a single path)."""
    if len(channel.paths) == 1:
        return 1.0
    ref = mean_received_power(wsig, single_path(params), params)
    return ref / mean_received_power(wsig, channel, params)


def normalised_params(wsig, channel: MultipathChannel,
                      params: SystemParams):
    """(params with A_p0 scaled by sqrt(G), G)."""
    G = power_factor(wsig, channel, params)
    if G == 1.0:
        return params, G
    return params.with_(A_p0=params.A_p0 * np.sqrt(G)), G
