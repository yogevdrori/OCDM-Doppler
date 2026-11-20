"""
Multipath channel with a (common or per-path) Doppler profile.

    Y(t) = sum_p h_p exp(j phi_p(t)) X(t + A_p(t) - tau_p)
    h_p      = A_p0 c_p exp(-j 2 pi f_Tx tau_p) / 2
    phi_p(t) = 2 pi ((f_Tx - f_Rx) t + f_Tx A_p(t)) + phi_Tx - phi_Rx

with A_p = int a_p (doppler_profile.py; the common profile unless the path
has its own), X the waveform (waveforms.py) and the front factor and carrier
of the existing single-path model ([OCDM] Eqn. 3: OCDMSignal.rx has
A_p0 e^{-j2pi f_Tx tau_p0} / (2 sqrt(N_is)) and Delta_phi = phi_Tx - phi_Rx
+ pi/4; the 1/sqrt(N_is) and e^{j pi/4} are inside OCDMSignal.tx). The
relative gains c_p are normalised to sum |c_p|^2 = 1 per realisation, so
the total gain is |A_p0|^2 as today and the SNR sigma_D^2 / P_n and the
working point keep their meaning.

The reference case -- OCDM, one path with c = 1 at tau = tau_p0 and the
common profile -- is delegated to the existing signal_tv.rx_nominal_profile
(which itself delegates a constant Doppler to OCDMSignal.rx), so it is
bit-identical to all earlier results. Every other case uses the formula
above (for OCDM single path it differs from the old code only by float
rounding; verify_step7.py reports by how much).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .doppler_profile import as_profile
from .params import SystemParams
from .signal_tv import rx_nominal_profile
from .waveforms import OCDMWaveSignal


@dataclass
class Path:
    gain: complex = 1.0
    """Relative complex gain c_p (the channel normalises sum |c_p|^2 = 1)."""
    delay: float = 2.0
    """Delay tau_p in units of T_samp_syn (fractional allowed)."""
    profile: object = None
    """Own Doppler profile (multiscale case); None: the common profile."""


@dataclass
class MultipathChannel:
    paths: list = field(default_factory=lambda: [Path()])
    label: str = ""

    def __post_init__(self):
        g = np.array([complex(q.gain) for q in self.paths])
        nrm = np.sqrt(np.sum(np.abs(g) ** 2))
        if nrm == 0:
            raise ValueError("all path gains are zero")
        if nrm != 1.0:
            for q, gi in zip(self.paths, g / nrm):
                q.gain = complex(gi)

    def is_reference(self, params: SystemParams) -> bool:
        """One path, c = 1, tau = tau_p0, common profile (the old model)."""
        if len(self.paths) != 1:
            return False
        q = self.paths[0]
        return (q.profile is None and complex(q.gain) == 1.0
                and float(q.delay) == float(params.tau_p0_in_samples))

    @property
    def delays(self):
        return np.array([q.delay for q in self.paths], dtype=float)

    @property
    def powers(self):
        return np.array([abs(q.gain) ** 2 for q in self.paths])

    def mean_delay(self):
        return float(np.sum(self.powers * self.delays))

    def rms_delay_spread(self):
        d = self.delays - self.mean_delay()
        return float(np.sqrt(np.sum(self.powers * d ** 2)))

    def max_delay_spread(self):
        return float(self.delays.max() - self.delays.min())


def single_path(params: SystemParams) -> MultipathChannel:
    return MultipathChannel([Path(1.0, params.tau_p0_in_samples)],
                            label="single path")


def exponential_pdp(D_max: float, P: int = 4, decay_db: float = 10.0,
                    rayleigh: bool = False, rng=None,
                    tau0: float = 2.0) -> MultipathChannel:
    """P paths at tau0 + D_max i / (P - 1) samples, powers decaying
    exponentially to -decay_db at the last path; fixed (real, positive) or
    Rayleigh gains (CN(0, power) per path, one draw per call), then
    normalised to sum |c|^2 = 1. D_max = 0 is a single path (fixed: c = 1;
    Rayleigh: a random phase)."""
    if D_max == 0:
        c = 1.0
        if rayleigh:
            c = np.exp(2j * np.pi * rng.random())
        return MultipathChannel([Path(c, tau0)], label="D_max=0")
    i = np.arange(P)
    pw = 10.0 ** (-decay_db / 10.0 * i / (P - 1))
    pw = pw / pw.sum()
    if rayleigh:
        c = np.sqrt(pw / 2) * (rng.standard_normal(P)
                               + 1j * rng.standard_normal(P))
    else:
        c = np.sqrt(pw).astype(complex)
    delays = tau0 + D_max * i / (P - 1)
    return MultipathChannel([Path(ci, di) for ci, di in zip(c, delays)],
                            label=f"exp PDP D_max={D_max:g}"
                                  f"{' Rayleigh' if rayleigh else ''}")


def received(wsig, channel: MultipathChannel, params: SystemParams, t,
             profile) -> np.ndarray:
    """Noiseless Y(t) by the multipath formula (no delegation)."""
    p = params
    t = np.atleast_1d(np.asarray(t, dtype=float))
    common = as_profile(profile)
    y = np.zeros(t.shape, dtype=complex)
    for q in channel.paths:
        prof = common if q.profile is None else as_profile(q.profile)
        At = prof.A(t)
        tau = q.delay * p.T_samp_syn
        h = p.A_p0 * complex(q.gain) * np.exp(-2j * np.pi * p.f_Tx * tau) / 2.0
        phase = (2.0 * np.pi * ((p.f_Tx - p.f_Rx) * t + p.f_Tx * At)
                 + p.phi_Tx - p.phi_Rx)
        y += h * np.exp(1j * phase) * wsig.tx(t + At - tau)
    return y


def rx_block(wsig, channel: MultipathChannel, params: SystemParams, n: int,
             profile, start_sample: int = 0) -> np.ndarray:
    """Noiseless received block at the nominal rate, samples
    start_sample .. start_sample + n - 1 (the reference case delegated to
    signal_tv.rx_nominal_profile)."""
    if isinstance(wsig, OCDMWaveSignal) and channel.is_reference(params):
        return rx_nominal_profile(wsig.long, n, profile, noiseless=True,
                                  start_sample=start_sample)
    k = start_sample + np.arange(int(n), dtype=np.int64)
    return received(wsig, channel, params, k * params.T_samp_ns, profile)
