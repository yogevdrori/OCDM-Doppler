"""
Detector parameters for a waveform with an arbitrary mean-power period.

The detector (discriminator.predicted_alpha, cbsfs.cost_function) reads the
cycle-frequency period from `SystemParams.N_sym` (= N_is + N_gi, the OCDM
symbol length). For another waveform the period of the mean power is
`P_samp` samples, so the harmonics lie at k (1 + a) / P_samp.

`WaveformParams` is a SystemParams whose `N_sym` property returns `P_samp`;
every quantity derived from N_sym (alpha_1_syn, alpha_NT, predicted_alpha)
follows. It is passed ONLY to the detector; the signal objects use the
OCDM timing fields (T_is, T_sym), which do not depend on N_sym. For OCDM
(P_samp = N_is + N_gi) it gives the identical integer, so all detector
outputs are bit-identical to those computed with the plain SystemParams.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .params import SystemParams


@dataclass(frozen=True)
class WaveformParams(SystemParams):
    P_samp: int = 0
    """Period of the mean power in samples (0: the OCDM N_is + N_gi)."""

    @property
    def N_sym(self) -> int:
        return int(self.P_samp) if self.P_samp else self.N_is + self.N_gi_eff


def detector_params(params: SystemParams, P_samp: int) -> WaveformParams:
    """`params` with the cycle-frequency period set to `P_samp` samples."""
    kw = {f.name: getattr(params, f.name) for f in fields(SystemParams)}
    return WaveformParams(**kw, P_samp=int(P_samp))
