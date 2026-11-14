"""
Loop filters for the Doppler tracking loop, and their linear analysis.

The detector (discriminator.py) gives e[i] per block; a loop filter turns the
error sequence into the next Doppler estimate. Two filters:

  FirstOrderFilter(mu):          a_hat <- a_hat + mu e
  SecondOrderFilter(mu1, mu2):   b_hat <- b_hat + mu2 e
                                 a_hat <- a_hat + b_hat + mu1 e
(proportional-plus-integral filter followed by an integrator; b_hat is the
estimated change of a per block). With mu2 = 0 and b_hat = 0 the second-order
filter is exactly the first-order one (adding +0.0 is exact in IEEE).

Linear model: e = K_d (a_bar - a_hat) + w, g1 = mu1 K_d, g2 = mu2 K_d.
Transfer a_bar -> a_hat:

    H2(z) = ((g1 + g2) z - g1) / (z^2 - (2 - g1 - g2) z + (1 - g1))

(g2 = 0: H(z) = g z^-1 / (1 - (1 - g) z^-1), the first-order loop).
Stable for 0 < g1 < 2, g2 > 0, 2 g1 + g2 < 4. Steady-state error
eps = a_hat - a_bar: constant a 0; ramp beta per block: 0 (first order:
-beta/g); constant second difference gamma per block^2: -gamma/g2.
Noise: Var(eps) = (sigma_e/K_d)^2 B_L, B_L = sum h_n^2 (first order:
g/(2 - g)), computed here exactly from the discrete Lyapunov equation.

Design: closed-loop poles r exp(+/- j theta): g1 = 1 - r^2,
g2 = 1 - 2 r cos(theta) + r^2; critically damped (double pole p):
g1 = 1 - p^2, g2 = (1 - p)^2; damping zeta and natural frequency w_n
(rad/block): r = exp(-zeta w_n), theta = w_n sqrt(1 - zeta^2);
`design_equal_BL` finds the design with a given B_L (e.g. that of a
first-order loop).

All gains here are the dimensionless g = mu K_d; the filters take mu = g/K_d0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Filters (stateful; one object per loop)
# ---------------------------------------------------------------------------
class LoopFilter:
    """Common interface: state (a_hat, b_hat), update(e) -> a_hat_next."""

    order = 0
    a_hat: float
    b_hat: float = float("nan")

    def update(self, e: float) -> float:
        raise NotImplementedError

    def get_state(self) -> tuple:
        raise NotImplementedError

    def set_state(self, state: tuple) -> None:
        raise NotImplementedError

    def copy(self) -> "LoopFilter":
        raise NotImplementedError

    def gains(self, K_d: float) -> tuple[float, float]:
        """(g1, g2) for a detector gain K_d."""
        raise NotImplementedError


class FirstOrderFilter(LoopFilter):
    """a_hat <- a_hat + mu e (the Step 4/5 loop)."""

    order = 1

    def __init__(self, mu: float, a_hat0: float = 0.0, label: str = ""):
        self.mu = float(mu)
        self.a_hat = float(a_hat0)
        self.label = label or f"1st, mu={self.mu:.3g}"

    def update(self, e):
        self.a_hat = self.a_hat + self.mu * e
        return self.a_hat

    def get_state(self):
        return (self.a_hat,)

    def set_state(self, state):
        self.a_hat = float(state[0])

    def copy(self):
        return FirstOrderFilter(self.mu, self.a_hat, self.label)

    def gains(self, K_d):
        return self.mu * K_d, 0.0


class SecondOrderFilter(LoopFilter):
    """b_hat <- b_hat + mu2 e;  a_hat <- a_hat + b_hat + mu1 e."""

    order = 2

    def __init__(self, mu1: float, mu2: float, a_hat0: float = 0.0,
                 b_hat0: float = 0.0, label: str = ""):
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.a_hat = float(a_hat0)
        self.b_hat = float(b_hat0)
        self.label = label or f"2nd, mu1={self.mu1:.3g}, mu2={self.mu2:.3g}"

    def update(self, e):
        self.b_hat = self.b_hat + self.mu2 * e
        self.a_hat = self.a_hat + self.b_hat + self.mu1 * e
        return self.a_hat

    def get_state(self):
        return (self.a_hat, self.b_hat)

    def set_state(self, state):
        self.a_hat, self.b_hat = float(state[0]), float(state[1])

    def copy(self):
        return SecondOrderFilter(self.mu1, self.mu2, self.a_hat, self.b_hat,
                                 self.label)

    def gains(self, K_d):
        return self.mu1 * K_d, self.mu2 * K_d


def make_filter(g1: float, g2: float, K_d0: float, a_hat0: float = 0.0,
                b_hat0: float = 0.0, label: str = "") -> LoopFilter:
    """FirstOrderFilter if g2 == 0, else SecondOrderFilter (mu = g / K_d0)."""
    if g2 == 0.0:
        return FirstOrderFilter(g1 / K_d0, a_hat0, label)
    return SecondOrderFilter(g1 / K_d0, g2 / K_d0, a_hat0, b_hat0, label)


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------
def gains_from_poles(r: float, theta: float) -> tuple[float, float]:
    """Closed-loop poles r exp(+/- j theta) -> (g1, g2)."""
    return 1.0 - r * r, 1.0 - 2.0 * r * math.cos(theta) + r * r


def gains_critically_damped(p: float) -> tuple[float, float]:
    """Double real pole p -> (g1, g2) = (1 - p^2, (1 - p)^2)."""
    return 1.0 - p * p, (1.0 - p) ** 2


def gains_from_zeta_wn(zeta: float, wn: float) -> tuple[float, float]:
    """Damping zeta, natural frequency wn (rad/block):
    r = exp(-zeta wn), theta = wn sqrt(1 - zeta^2) (0 for zeta >= 1)."""
    r = math.exp(-zeta * wn)
    theta = wn * math.sqrt(max(0.0, 1.0 - zeta * zeta))
    return gains_from_poles(r, theta)


def first_order_BL(g: float) -> float:
    return g / (2.0 - g)


def design_equal_BL(B_target: float, zeta: float = 1.0
                    ) -> tuple[float, float, float]:
    """(g1, g2, wn) of the (zeta, wn) design with noise bandwidth B_target.
    B_L increases monotonically with wn; solved by bisection."""
    lo, hi = 1e-7, 1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if B_L(*gains_from_zeta_wn(zeta, mid)) < B_target:
            lo = mid
        else:
            hi = mid
    wn = 0.5 * (lo + hi)
    g1, g2 = gains_from_zeta_wn(zeta, wn)
    return g1, g2, wn


# ---------------------------------------------------------------------------
# Linear analysis
# ---------------------------------------------------------------------------
def is_stable(g1: float, g2: float) -> bool:
    return 0 < g1 < 2 and g2 >= 0 and 2 * g1 + g2 < 4 and \
        np.all(np.abs(closed_loop_poles(g1, g2)) < 1)


def closed_loop_poles(g1: float, g2: float, k_ratio: float = 1.0
                      ) -> np.ndarray:
    """Poles of H2 with the gains scaled by k_ratio = K_d / K_d0."""
    a, b = g1 * k_ratio, g2 * k_ratio
    if b == 0.0:
        return np.array([1.0 - a])
    return np.roots([1.0, -(2.0 - a - b), 1.0 - a])


def H2(freq, g1: float, g2: float, T_blk: float):
    """H2 at freq (Hz), one update per block of T_blk seconds."""
    z = np.exp(2j * np.pi * np.asarray(freq, dtype=float) * T_blk)
    if g2 == 0.0:
        return g1 / (z - (1.0 - g1))
    return ((g1 + g2) * z - g1) / (z * z - (2.0 - g1 - g2) * z + (1.0 - g1))


def f_3db(g1: float, g2: float, T_blk: float) -> float:
    """Highest frequency where |H2| = 1/sqrt(2) (allows peaking above 1):
    dense scan up to Nyquist, then bisection on the last crossing."""
    fn = 0.5 / T_blk
    f = np.linspace(0, fn, 200001)[1:]
    m = np.abs(H2(f, g1, g2, T_blk)) - 1 / math.sqrt(2)
    idx = np.flatnonzero((m[:-1] >= 0) & (m[1:] < 0))
    if idx.size == 0:
        return float("nan")
    lo, hi = f[idx[-1]], f[idx[-1] + 1]
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if abs(H2(mid, g1, g2, T_blk)) >= 1 / math.sqrt(2):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def B_L(g1: float, g2: float) -> float:
    """Noise bandwidth sum_n h_n^2 of H2, exact (discrete Lyapunov)."""
    if g2 == 0.0:
        return first_order_BL(g1)
    a1, a0 = -(2.0 - g1 - g2), 1.0 - g1
    A = np.array([[0.0, 1.0], [-a0, -a1]])
    Bv = np.array([[0.0], [1.0]])
    C = np.array([[-g1, g1 + g2]])
    # P = A P A^T + B B^T   <=>   (I - A (x) A) vec(P) = vec(B B^T)
    P = np.linalg.solve(np.eye(4) - np.kron(A, A),
                        (Bv @ Bv.T).reshape(-1)).reshape(2, 2)
    return float((C @ P @ C.T)[0, 0])


def linear_response(g1: float, g2: float, a_bar: np.ndarray,
                    a_hat0: float = 0.0, b_hat0: float = 0.0) -> np.ndarray:
    """Noise-free loop: x = a_bar - a_hat, b += g2 x, a_hat += b + g1 x.
    Returns a_hat[i], i = 0 .. len(a_bar) (a_hat[i] is used in block i)."""
    a_bar = np.asarray(a_bar, dtype=float)
    out = np.empty(a_bar.size + 1)
    a, b = float(a_hat0), float(b_hat0)
    out[0] = a
    for i, ab in enumerate(a_bar):
        x = ab - a
        b = b + g2 * x
        a = a + b + g1 * x
        out[i + 1] = a
    return out


def impulse(g1: float, g2: float, n: int) -> np.ndarray:
    """h_k, k = 0 .. n-1 (response a_hat to a_bar = delta[k])."""
    x = np.zeros(n)
    x[0] = 1.0
    return linear_response(g1, g2, x)[:n]


def step_response(g1: float, g2: float, n: int) -> np.ndarray:
    """a_hat[k] for a_bar = 1 from block 0 (a_hat0 = b_hat0 = 0)."""
    return linear_response(g1, g2, np.ones(n))[:n]


def _horizon(g1: float, g2: float) -> int:
    """Blocks enough for the step response to settle (30 time constants of
    the slowest pole)."""
    rmax = float(np.max(np.abs(closed_loop_poles(g1, g2))))
    return int(30.0 / max(1.0 - rmax, 1e-6)) + 100


def n95(g1: float, g2: float, n_max: int | None = None) -> int:
    """First block k from which |a_hat - 1| <= 0.05 for good (step)."""
    s = step_response(g1, g2, n_max or _horizon(g1, g2))
    bad = np.flatnonzero(np.abs(s - 1.0) > 0.05)
    return int(bad[-1] + 1) if bad.size else 0


def overshoot(g1: float, g2: float, n_max: int | None = None) -> float:
    """max(a_hat) - 1 for a unit step (0 if none)."""
    s = step_response(g1, g2, n_max or _horizon(g1, g2))
    return max(0.0, float(s.max() - 1.0))


def ss_error_ramp(beta: float, g1: float, g2: float) -> float:
    """Steady-state eps for a ramp of beta per block: -beta/g1 (first
    order), 0 (second order)."""
    return -beta / g1 if g2 == 0.0 else 0.0


def ss_error_parabola(gamma: float, g1: float, g2: float) -> float:
    """Steady-state eps for a constant second difference gamma per block^2
    (second order): -gamma / g2 (diverges for the first-order loop)."""
    return -gamma / g2 if g2 > 0 else float("-inf") * np.sign(gamma)


@dataclass
class Design:
    """A loop design: name, gains (g1, g2) and derived quantities."""
    name: str
    g1: float
    g2: float

    @property
    def order(self):
        return 1 if self.g2 == 0.0 else 2

    @property
    def BL(self):
        return B_L(self.g1, self.g2)

    def filter(self, K_d0: float, a_hat0: float = 0.0,
               b_hat0: float = 0.0) -> LoopFilter:
        return make_filter(self.g1, self.g2, K_d0, a_hat0, b_hat0, self.name)


def first_order_design(p: float) -> Design:
    return Design(f"1st p={p:g}", 1.0 - p, 0.0)


def equal_BL_design(p1: float, zeta: float) -> Design:
    """Second-order (zeta) design with the noise bandwidth of the
    first-order loop with pole p1."""
    g1, g2, wn = design_equal_BL(first_order_BL(1.0 - p1), zeta)
    tag = "cd" if zeta == 1.0 else f"z{zeta:g}"
    return Design(f"2nd {tag} ~p1={p1:g}", g1, g2)


def critically_damped_design(p: float) -> Design:
    g1, g2 = gains_critically_damped(p)
    return Design(f"2nd cd p={p:g}", g1, g2)
