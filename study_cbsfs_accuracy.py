"""
How accurate is CB-SFS here, and what drives its bias?

CB-SFS provides the initial Doppler estimate a_hat_sr for the Section-4
algorithm, and it is also the natural home for the discriminator of a
closed-loop detector. Its accuracy therefore sets the floor for everything
downstream, so it is worth characterising before reproducing the baseline.

Sweeps the record length M, the segment length N_seg and the harmonic index
used for the estimate.
"""

import time

import numpy as np

from ocdm_doppler import SystemParams, OCDMSignal, cbsfs


def run(a_true, M, N_seg, N_last, seed=0, n_alpha=384):
    p = SystemParams(N_is=16, modulation="qpsk", a_max=1e-3, a_true=a_true,
                     snr_db=float("inf"), rho=32, ts_reference="a_true",
                     N_seg=N_seg, N_T=N_last)
    sig = OCDMSignal(p, n_symbols=M + 16, seed=seed)
    y = sig.rx_nominal(M * p.N_sym, a=a_true, noiseless=True)
    res = cbsfs.estimate_doppler_harmonic(y, p, N_last=N_last,
                                          n_alpha=n_alpha)
    return res.a_hat


A_TRUE = 5e-4

print("=" * 74)
print("A. effect of record length M      (N_seg=256, harmonic=2)")
print("=" * 74)
print(f"{'M':>6} {'N_c':>8} {'a_hat':>14} {'error':>13} {'rel.err':>10}")
for M in (300, 600, 1200, 2400):
    ah = run(A_TRUE, M, 256, 2)
    print(f"{M:6d} {M*20:8d} {ah:14.6e} {ah-A_TRUE:+13.3e} "
          f"{(ah-A_TRUE)/A_TRUE:9.1%}")

print()
print("=" * 74)
print("B. effect of segment length N_seg  (M=1200, harmonic=2)")
print("=" * 74)
print(f"{'N_seg':>6} {'peak width 1/N_seg':>20} {'a_hat':>14} {'error':>13}"
      f" {'rel.err':>10}")
for N_seg in (128, 256, 512, 1024, 2048):
    ah = run(A_TRUE, 1200, N_seg, 2)
    print(f"{N_seg:6d} {1.0/N_seg:20.6f} {ah:14.6e} {ah-A_TRUE:+13.3e} "
          f"{(ah-A_TRUE)/A_TRUE:9.1%}")

print()
print("=" * 74)
print("C. effect of harmonic index  (M=1200, N_seg=1024)  <- the leverage")
print("=" * 74)
print(f"{'k':>3} {'alpha_k':>12} {'shift k*a*alpha_1':>20} {'a_hat':>14}"
      f" {'error':>13} {'rel.err':>10}")
for k in (1, 2, 3, 4, 6, 8):
    ah = run(A_TRUE, 1200, 1024, k)
    shift = k * A_TRUE / 20.0
    print(f"{k:3d} {k/20.0:12.5f} {shift:20.3e} {ah:14.6e} "
          f"{ah-A_TRUE:+13.3e} {(ah-A_TRUE)/A_TRUE:9.1%}")

print()
print("=" * 74)
print("D. is the error a bias or noise?  (M=1200, N_seg=1024, k=4)")
print("=" * 74)
for a_t in (0.0, 2.5e-4, 5e-4, 7.5e-4, 1e-3):
    ests = [run(a_t, 1200, 1024, 4, seed=s) for s in range(4)]
    ests = np.array(ests)
    print(f"  a_true={a_t:+.3e}   mean(a_hat)={ests.mean():+.4e}   "
          f"std={ests.std():.2e}   mean error={ests.mean()-a_t:+.3e}")
