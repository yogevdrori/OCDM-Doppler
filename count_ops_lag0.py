"""
Independent check of the complexity count for the Delta = {0} detector.

Re-implements one loop update (two probe frequencies, segmented CAF with
running sums, autocorrelated CAF, magnitudes, ratio) sample by sample, with
explicit counters for every real multiplication, and checks that its output
equals discriminator.early_late on the same block.
"""

import math
import sys

import numpy as np

from ocdm_doppler import preset_noisy_22jan
from ocdm_doppler import discriminator as disc
from ocdm_doppler.loop import LongSignal


class Ops:
    rm = 0      # real multiplications
    div = 0
    sqrt = 0


def cmul(a, b):                     # complex x complex: 4 RM
    Ops.rm += 4
    return complex(a.real * b.real - a.imag * b.imag,
                   a.real * b.imag + a.imag * b.real)


def rmul(r, b):                     # real x complex: 2 RM
    Ops.rm += 2
    return complex(r * b.real, r * b.imag)


def update_counted(y, a_hat, p, N_seg, k=1):
    N_c = y.size
    delta = 0.25 / N_seg
    alpha_p = k * (1 + a_hat) / p.N_sym
    alphas = (alpha_p + delta, alpha_p - delta)

    # lag-0 products |Y|^2: 2 RM per sample (shared by both probes)
    z = np.empty(N_c)
    for n in range(N_c):
        Ops.rm += 2
        z[n] = y[n].real ** 2 + y[n].imag ** 2

    costs = []
    n_pos = N_c - N_seg + 1
    for a in alphas:
        w = complex(math.cos(-2 * math.pi * a), math.sin(-2 * math.pi * a))
        ph = 1 + 0j
        mod = np.empty(N_c, dtype=complex)
        for n in range(N_c):
            mod[n] = rmul(z[n], ph)          # z e^{-j2 pi a n}: 2 RM
            ph = cmul(ph, w)                 # recursive phasor: 4 RM
        # running window sums: additions only
        R = np.empty(n_pos, dtype=complex)
        s = mod[:N_seg].sum()
        R[0] = s
        for n in range(1, n_pos):
            s += mod[n + N_seg - 1] - mod[n - 1]
            R[n] = s
        # autocorrelated CAF, N_lag = N_seg: one CM per term
        acc = 0j
        for n in range(n_pos - N_seg):
            acc += cmul(R[n + N_seg], R[n].conjugate())
        Ops.rm += 2                          # |acc|^2
        Ops.sqrt += 1
        costs.append(math.sqrt(acc.real ** 2 + acc.imag ** 2))
    Ops.div += 1                             # (C+ - C-)/(C+ + C-)
    return (costs[0] - costs[1]) / (costs[0] + costs[1])


def main():
    N_c, N_seg = 24000, 2048
    p = preset_noisy_22jan(4.0).with_(cbsfs_deltas=(0,), a_true=5.1542e-4)
    sig = LongSignal(p, seed=1000)
    y = sig.rx_nominal(N_c, noiseless=True)
    rng = np.random.default_rng(7)
    y = y + math.sqrt(p.noise_power / 2) * (rng.standard_normal(N_c)
                                            + 1j * rng.standard_normal(N_c))
    a_hat = 3e-4
    e_ref = disc.early_late(y, a_hat, p, N_seg=N_seg)
    e_cnt = update_counted(y, a_hat, p, N_seg)
    pred = 2 + 8 + 4 + 8 * (N_c - 2 * N_seg + 1) / N_c
    print(f"e (repo detector)    = {e_ref:+.12f}")
    print(f"e (counted re-impl.) = {e_cnt:+.12f}   |diff| = {abs(e_ref - e_cnt):.1e}")
    print(f"real multiplications per update = {Ops.rm:,}  (+{Ops.sqrt} sqrt, {Ops.div} division)")
    print(f"per received sample             = {Ops.rm / N_c:.3f} RM  (document: {pred:.3f})")
    print(f"per second at 4 MHz             = {Ops.rm / N_c * 4e6 / 1e6:.1f} M RM/s")
    sys.exit(0 if abs(e_ref - e_cnt) < 1e-9 else 1)


if __name__ == "__main__":
    main()
