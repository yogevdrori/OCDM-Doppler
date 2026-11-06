"""
Time-varying Doppler factor profiles a(t) for the time-varying receiver
(signal_tv.py) and loop (loop_tv.py).

A time-varying Doppler factor enters the received signal only through its
integral A(t) = int_0^t a(s) ds (time warp u(t) = t + A(t) - tau_p0, carrier
phase 2 pi f_Tx A(t)). Because the carrier phase multiplies A by f_Tx = 1 GHz,
an error dA becomes a phase error 2 pi 1e9 dA, so every profile here gives
A(t) in CLOSED FORM, written to avoid cancellation (see each class).

Profiles are vectorised over t (seconds). `block_mean(t0, t1)` is the mean
Doppler over [t0, t1], the quantity the loop estimates per block.

Generic use: PiecewiseLinearDoppler through any points, or CallableDoppler
with a user function and its closed-form integral. A plain float is wrapped
in ConstantDoppler by `as_profile`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


class DopplerProfile:
    """Base class: a(t), its integral A(t) from 0, and the block mean."""

    def a(self, t):
        raise NotImplementedError

    def A(self, t):
        raise NotImplementedError

    def block_mean(self, t0, t1):
        """(A(t1) - A(t0)) / (t1 - t0): mean Doppler over [t0, t1]."""
        t0 = np.asarray(t0, dtype=float)
        t1 = np.asarray(t1, dtype=float)
        return (self.A(t1) - self.A(t0)) / (t1 - t0)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                         if not k.startswith("_"))
        return f"{type(self).__name__}({args})"


class ConstantDoppler(DopplerProfile):
    """a(t) = a; A(t) = a t."""

    def __init__(self, a: float):
        self.a0 = float(a)

    def a(self, t):
        return np.full(np.shape(t), self.a0)

    def A(self, t):
        return self.a0 * np.asarray(t, dtype=float)


class RampDoppler(DopplerProfile):
    """a(t) = a0 + rate t (rate in 1/s); A(t) = a0 t + rate t^2 / 2."""

    def __init__(self, a0: float, rate: float):
        self.a0 = float(a0)
        self.rate = float(rate)

    def a(self, t):
        return self.a0 + self.rate * np.asarray(t, dtype=float)

    def A(self, t):
        t = np.asarray(t, dtype=float)
        return t * (self.a0 + 0.5 * self.rate * t)


class SinusoidalDoppler(DopplerProfile):
    """a(t) = a0 + amp sin(2 pi freq t + phase).

    A(t) = a0 t + amp/(2 pi f) (cos(phase) - cos(2 pi f t + phase))
         = a0 t + amp/(pi f) sin(pi f t) sin(pi f t + phase),
    the product form having no cancellation at small t.
    """

    def __init__(self, a0: float, amp: float, freq: float,
                 phase: float = 0.0):
        self.a0 = float(a0)
        self.amp = float(amp)
        self.freq = float(freq)
        self.phase = float(phase)

    def a(self, t):
        t = np.asarray(t, dtype=float)
        return self.a0 + self.amp * np.sin(2 * np.pi * self.freq * t
                                           + self.phase)

    def A(self, t):
        t = np.asarray(t, dtype=float)
        x = np.pi * self.freq * t
        return (self.a0 * t + self.amp / (np.pi * self.freq)
                * np.sin(x) * np.sin(x + self.phase))


class FlyByDoppler(DopplerProfile):
    """Source passing the receiver:
    a(t) = -a_max (t - t0) / sqrt((t - t0)^2 + T_c^2),
    from +a_max to -a_max, largest rate a_max / T_c at t = t0.

    A(t) = -a_max (s(t) - s(0)),  s(t) = sqrt((t - t0)^2 + T_c^2)
         = -a_max t (t - 2 t0) / (s(t) + s(0))   (no cancellation).
    """

    def __init__(self, a_max: float, t0: float, T_c: float):
        self.a_max = float(a_max)
        self.t0 = float(t0)
        self.T_c = float(T_c)

    def _s(self, t):
        return np.hypot(np.asarray(t, dtype=float) - self.t0, self.T_c)

    def a(self, t):
        t = np.asarray(t, dtype=float)
        return -self.a_max * (t - self.t0) / self._s(t)

    def A(self, t):
        t = np.asarray(t, dtype=float)
        return -self.a_max * t * (t - 2 * self.t0) / (self._s(t)
                                                      + self._s(0.0))


class PiecewiseLinearDoppler(DopplerProfile):
    """Linear interpolation through (times[k], values[k]); held constant
    before the first and after the last point. A(t) is exact
    (piecewise quadratic): the integral up to each knot is precomputed and
    the part within the segment is added in local time.
    """

    def __init__(self, times: Sequence[float], values: Sequence[float]):
        tk = np.asarray(times, dtype=float)
        vk = np.asarray(values, dtype=float)
        if tk.ndim != 1 or tk.size < 1 or tk.size != vk.size:
            raise ValueError("times and values: equal-length 1-D sequences")
        if np.any(np.diff(tk) <= 0):
            raise ValueError("times must be strictly increasing")
        self.times = tk
        self.values = vk
        self._slope = np.diff(vk) / np.diff(tk) if tk.size > 1 else np.zeros(0)
        # integral from times[0] to each knot
        seg = 0.5 * (vk[:-1] + vk[1:]) * np.diff(tk)
        self._Ak = np.concatenate([[0.0], np.cumsum(seg)])
        # integral from times[0] to t = 0, so that A(0) = 0
        self._A0 = float(self._F(np.array([0.0]))[0])

    def a(self, t):
        return np.interp(np.asarray(t, dtype=float), self.times, self.values)

    def _F(self, t):
        """int_{times[0]}^{t} a(s) ds (negative for t < times[0])."""
        t = np.asarray(t, dtype=float)
        tk, vk = self.times, self.values
        out = np.empty(t.shape)
        lo = t <= tk[0]
        hi = t >= tk[-1]
        mid = ~(lo | hi)
        out[lo] = vk[0] * (t[lo] - tk[0])
        out[hi] = self._Ak[-1] + vk[-1] * (t[hi] - tk[-1])
        if np.any(mid):
            k = np.searchsorted(tk, t[mid], side="right") - 1
            tau = t[mid] - tk[k]
            out[mid] = (self._Ak[k] + vk[k] * tau
                        + 0.5 * self._slope[k] * tau * tau)
        return out

    def A(self, t):
        t = np.asarray(t, dtype=float)
        return (self._F(np.atleast_1d(t)) - self._A0).reshape(t.shape)


class CallableDoppler(DopplerProfile):
    """User profile: a_fn(t) and its closed-form integral A_fn(t) with
    A_fn(0) = 0 (required: a numerical integral is not precise enough for
    the carrier phase)."""

    def __init__(self, a_fn: Callable, A_fn: Callable):
        if A_fn is None:
            raise ValueError("CallableDoppler needs the closed-form A_fn")
        self._a = a_fn
        self._A = A_fn

    def a(self, t):
        return np.asarray(self._a(np.asarray(t, dtype=float)), dtype=float)

    def A(self, t):
        return np.asarray(self._A(np.asarray(t, dtype=float)), dtype=float)


def as_profile(x) -> DopplerProfile:
    """A DopplerProfile as is; a number -> ConstantDoppler."""
    if isinstance(x, DopplerProfile):
        return x
    if np.ndim(x) == 0:
        return ConstantDoppler(float(x))
    raise TypeError(f"not a Doppler profile: {x!r}")
