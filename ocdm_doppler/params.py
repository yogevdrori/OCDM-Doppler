"""
System parameters for the OCDM Doppler-estimation study.

Notation follows, as closely as possible, the two source documents:

  [OCDM]  Z. Tan, "OCDM Signal Analysis" (the project manuscript).
  [CBSFS] M. Kumar and R. Dabora, "A Novel Sampling Frequency Offset
          Estimation Algorithm for OFDM Systems Based on Cyclostationary
          Properties", IEEE Access, vol. 7, 2019.

Everything downstream (signal generation, analytical AF, CB-SFS,
curve-fitting, and later the closed-loop detector) reads its configuration
from `SystemParams`, so an experiment is fully described by one object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

Modulation = Literal["bpsk", "qpsk"]


@dataclass(frozen=True)
class SystemParams:
    """Complete description of one OCDM Doppler-estimation experiment.

    Time quantities are in seconds; `T_samp_syn` is the transmitter's
    (synchronised) sampling interval, which sets the OCDM symbol timing.
    """

    # ---------------- OCDM symbol structure ----------------
    N_is: int = 16
    """Number of information symbols (chirps) per OCDM symbol."""

    N_gi: int | None = None
    """Guard-interval length in transmitter samples. Defaults to N_is // 4.

    NOTE: in [OCDM] the pulse-shaping function is g(t) = 1 only on
    [0, T_is), so the guard interval is a *zero gap* after each OCDM
    symbol -- NOT a cyclic prefix.
    """

    modulation: Modulation = "qpsk"
    sigma_D2: float = 1.0
    """Variance of the information symbols, sigma_D^2 in [OCDM] Eqn. (1)."""

    # ---------------- Front-end ----------------
    f_samp_syn: float = 4e6
    """Transmitter sampling frequency f_samp^(syn) [Hz]."""

    f_Tx: float = 1e9
    """Transmitter carrier frequency [Hz]."""

    cfo: float = 1e3
    """Carrier frequency offset: f_Rx = f_Tx + cfo [Hz]."""

    phi_Tx: float = 0.0
    phi_Rx: float = 0.0

    # ---------------- Single-path mobile channel ----------------
    A_p0: complex = 1.08 + 0.36j
    """Complex amplitude scaling factor of the single path."""

    tau_p0_in_samples: float = 2.0
    """Path delay tau_p0, expressed in units of T_samp_syn."""

    # ---------------- Doppler ----------------
    a_max: float = 1e-3
    a_min: float | None = None
    """Defaults to -a_max."""

    a_true: float | None = None
    """True Doppler scaling factor. Defaults to a_max / 2."""

    # ---------------- Receiver / estimation grid ----------------
    rho: int = 32
    """Oversampling factor rho: the fine receiver sampling interval is
    T_s = T_samp_syn / ((1 + a_ref) * rho); see `T_s`."""

    ts_reference: Literal["a_min", "a_max", "a_true"] = "a_min"
    """Which Doppler value appears in the denominator of T_s.

    The two results documents differ here: the 22-Jan document uses
    (1 + a_min), the 31-Jan (noiseless) document uses (1 + a_true).
    Kept configurable so both experiments can be reproduced exactly.
    """

    N_avg: int = 2 ** 12 + 1
    """Number of OCDM symbols averaged per AF value."""

    N_lag: int = 2 ** 6
    """Number of lag values placed across the lag interval."""

    N_a: int = 2 ** 18
    """Number of candidate Doppler values in the curve-fitting grid."""

    N_mc: int = 2 ** 7
    """Number of Monte-Carlo experiments."""

    # ---------------- Noise ----------------
    snr_db: float = float("inf")
    """sigma_D^2 / P_n in dB (as defined in the results documents).
    Use float('inf') for the noiseless case."""

    # ---------------- CB-SFS ----------------
    M_cbsfs: int = 512
    """Number of OCDM symbols observed by CB-SFS.

    NOTE: the results documents do not state this value (nor N_seg / N_T).
    It is exposed here so the sensitivity of CB-SFS to it can be studied.
    """

    N_seg: int = 2 ** 8
    """Segment length for the segmented CAF, [CBSFS] Eqn. (27)."""

    N_T: int = 2
    """Number of cycle-frequency harmonics spanned by the search range
    A = (0, alpha_{N_T}], with alpha_{N_T} = N_T / N_sym."""

    n_alpha_grid: int = 4096
    """Number of points on the cycle-frequency grid used for peak search."""

    cbsfs_deltas: tuple[int, ...] | None = None
    """Lag set N_Delta for the cost function, [CBSFS] Eqn. (29).
    Defaults to 1..N_is (the lags for which the CAF is non-zero)."""

    seed: int = 0

    # ---------------- Derived quantities ----------------
    @property
    def N_gi_eff(self) -> int:
        return self.N_is // 4 if self.N_gi is None else self.N_gi

    @property
    def N_sym(self) -> int:
        """N_sym = N_is + N_gi."""
        return self.N_is + self.N_gi_eff

    @property
    def T_samp_syn(self) -> float:
        return 1.0 / self.f_samp_syn

    @property
    def T_is(self) -> float:
        """T_is = N_is * T_samp^(syn)."""
        return self.N_is * self.T_samp_syn

    @property
    def T_gi(self) -> float:
        return self.N_gi_eff * self.T_samp_syn

    @property
    def T_sym(self) -> float:
        return self.T_is + self.T_gi

    @property
    def f_Rx(self) -> float:
        return self.f_Tx + self.cfo

    @property
    def tau_p0(self) -> float:
        return self.tau_p0_in_samples * self.T_samp_syn

    @property
    def a_min_eff(self) -> float:
        return -self.a_max if self.a_min is None else self.a_min

    @property
    def a_true_eff(self) -> float:
        return self.a_max / 2.0 if self.a_true is None else self.a_true

    @property
    def T_s(self) -> float:
        """Fine receiver sampling interval used to approximate CT."""
        ref = {
            "a_min": self.a_min_eff,
            "a_max": self.a_max,
            "a_true": self.a_true_eff,
        }[self.ts_reference]
        return self.T_samp_syn / ((1.0 + ref) * self.rho)

    @property
    def T_samp_ns(self) -> float:
        """Receiver's nominal (non-synchronised) sampling interval.

        CB-SFS operates on Y sampled at this rate, which is a *different and
        much coarser* grid than the fine T_s grid used for AF estimation.
        Here the receiver clock is assumed ideal, so the only scaling between
        transmitter and receiver time axes is the Doppler itself.
        """
        return self.T_samp_syn

    @property
    def Delta_f_a(self) -> float:
        """Delta_{f,a} = (1 + a) f_Tx - f_Rx  ([OCDM] Sec. 3)."""
        return (1.0 + self.a_true_eff) * self.f_Tx - self.f_Rx

    @property
    def Delta_phi(self) -> float:
        """Delta_phi = phi_Tx - phi_Rx + pi/4  ([OCDM] Sec. 3)."""
        return self.phi_Tx - self.phi_Rx + np.pi / 4.0

    def Delta_f_of_a(self, a: float) -> float:
        """Delta_{f,a} for an arbitrary Doppler value a."""
        return (1.0 + a) * self.f_Tx - self.f_Rx

    @property
    def noise_power(self) -> float:
        """P_n from sigma_D^2 / P_n [dB]."""
        if np.isinf(self.snr_db):
            return 0.0
        return self.sigma_D2 / (10.0 ** (self.snr_db / 10.0))

    @property
    def deltas(self) -> np.ndarray:
        """Lag set N_Delta used by the CB-SFS cost function.

        [CBSFS] defines N_Delta as the lags for which the CAF is non-zero for
        at least one time instant. For OCDM the AF vanishes for |Delta| >= N_is,
        so the default is 0 .. N_is - 1.

        Delta = 0 must be included: it carries the strongest cyclostationary
        signature, because the guard interval makes |Y[k]|^2 periodic with
        period N_sym.
        """
        if self.cbsfs_deltas is None:
            return np.arange(0, self.N_is)
        return np.asarray(self.cbsfs_deltas, dtype=int)

    @property
    def alpha_1_syn(self) -> float:
        """Fundamental cycle frequency for synchronised sampling, 1 / N_sym."""
        return 1.0 / self.N_sym

    @property
    def alpha_NT(self) -> float:
        """Upper end of the cycle-frequency search range, N_T / N_sym."""
        return self.N_T / self.N_sym

    def with_(self, **kwargs) -> "SystemParams":
        """Return a copy with fields replaced (params are frozen)."""
        return replace(self, **kwargs)

    def summary(self) -> str:
        lines = [
            "SystemParams",
            f"  N_is={self.N_is}  N_gi={self.N_gi_eff}  N_sym={self.N_sym}"
            f"  mod={self.modulation}",
            f"  T_samp_syn={self.T_samp_syn:.4g}s  T_is={self.T_is:.4g}s"
            f"  T_sym={self.T_sym:.4g}s",
            f"  f_Tx={self.f_Tx:.4g}Hz  cfo={self.cfo:.4g}Hz"
            f"  Delta_f_a={self.Delta_f_a:.6g}Hz",
            f"  A_p0={self.A_p0}  |A_p0|^2={abs(self.A_p0) ** 2:.4f}"
            f"  tau_p0={self.tau_p0:.4g}s",
            f"  a in ({self.a_min_eff:g}, {self.a_max:g})"
            f"  a_true={self.a_true_eff:g}",
            f"  rho={self.rho}  T_s={self.T_s:.6g}s"
            f"  (ref={self.ts_reference})",
            f"  N_avg={self.N_avg}  N_lag={self.N_lag}  N_a={self.N_a}"
            f"  N_mc={self.N_mc}",
            f"  SNR={self.snr_db}dB  P_n={self.noise_power:.4g}",
            f"  CB-SFS: M={self.M_cbsfs}  N_seg={self.N_seg}  N_T={self.N_T}"
            f"  alpha_1={self.alpha_1_syn:.6g}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Presets that mirror the two results documents exactly.
# ---------------------------------------------------------------------------

def preset_noiseless_31jan() -> SystemParams:
    """31-Jan-2026 document, Experiment 1 (noiseless, no sampling deviation).

    Reported there: normalized bias = -1.0138, NMSE = 4.9102.
    """
    return SystemParams(
        N_is=16,
        modulation="qpsk",
        a_max=1e-3,
        rho=2 ** 5,
        ts_reference="a_true",
        N_avg=2 ** 12 + 1,
        N_lag=2 ** 5 - 1,
        N_a=2 ** 18,
        N_mc=2 ** 7,
        snr_db=float("inf"),
    )


def preset_noisy_22jan(snr_db: float = 20.0) -> SystemParams:
    """22-Jan-2026 document, Experiment 1 (BPSK, noisy).

    Reported there (20 dB): CB-SFS Doppler NMSE = 0.0044046,
    AF-magnitude Doppler NMSE = 1.4881.
    """
    return SystemParams(
        N_is=16,
        modulation="bpsk",
        a_max=1e-3,
        rho=2 ** 5,
        ts_reference="a_min",
        N_avg=2 ** 10 + 1,
        N_lag=2 ** 6,
        N_a=2 ** 18,
        N_mc=2 ** 7,
        snr_db=snr_db,
    )
