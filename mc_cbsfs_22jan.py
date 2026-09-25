"""
Monte-Carlo NMSE of the CB-SFS Doppler estimate at the 22-Jan settings.

Comparable to the "CBSFS Doppler NMSE" column of the 22-Jan results doc:
  a_max = 1e-3 (a = 5e-4)  -> Settings 1, 2, 5
  a_max = 1e-2 (a = 5e-3)  -> Settings 3, 4
CB-SFS does not depend on T_s (rho), so Settings 1/2 (and 3/4) share numbers.

Estimator: identical to make_figures.initial_estimate
  N_c = 24000 samples at the nominal rate, N_seg = 2048, harmonic k = 1,
  parabolic sub-grid refinement.  The search window around alpha_1 is
  +/-0.05 alpha_1 with 128 points (make_figures uses +/-0.25 alpha_1 with
  512 points); the peak and its refinement are the same, it is only faster.
"""

import sys
import time
from multiprocessing import Pool

import numpy as np

from ocdm_doppler import OCDMSignal, cbsfs, preset_noisy_22jan

N_C, N_SEG, K = 24000, 2048, 1
N_MC = int(sys.argv[1]) if len(sys.argv) > 1 else 128


def one(args):
    snr, a_max, seed = args
    p = preset_noisy_22jan(snr).with_(a_max=a_max)
    M = N_C // p.N_sym
    sig = OCDMSignal(p, n_symbols=M + 2, seed=seed)
    y = sig.rx_nominal(M * p.N_sym)
    ak, _, _ = cbsfs.locate_harmonic(y, p, K, n_alpha=128,
                                     half_width_factor=0.05, N_seg=N_SEG)
    return ak / K / p.alpha_1_syn - 1.0


if __name__ == "__main__":
    cases = [(snr, am) for am in (1e-3, 1e-2) for snr in (10.0, 20.0)]
    t0 = time.time()
    with Pool(2) as pool:
        for snr, am in cases:
            a = am / 2
            est = np.array(pool.map(one, [(snr, am, 1000 + s)
                                          for s in range(N_MC)]))
            err = est - a
            print(f"a_max={am:g} a={a:g} SNR={snr:g}dB  N_mc={N_MC}  "
                  f"NMSE={np.mean(err**2)/a**2:.4g}  "
                  f"norm.bias={err.mean()/a:+.4g}  "
                  f"RMSE={np.sqrt(np.mean(err**2)):.3e}  "
                  f"std={err.std():.3e}   [{time.time()-t0:.0f}s]",
                  flush=True)
