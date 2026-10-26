"""
Step 4 check: is the detector offset at a = 0 real, and does it depend on a?

Part B found the open-loop characteristic offset by ~ +1.1e-5 at a = 0
exactly (a_hat = 0), but not at neighbouring a. This script measures
mean(e) / K_d at a_hat = a (where an unbiased detector gives 0) for
a in {0, 2.5e-4, 5e-4, 1e-3} and delta in {0.25, 0.5} / N_seg, on 256 fresh
noiseless blocks (seeds 5000..5255, independent of the 1000.. seeds used
elsewhere). Lag set Delta = {0}, N_c = 24000, N_seg = 2048, 22-Jan preset.

Result (README, Step 4 B): +1.10e-5 (11 sigma) at a = 0 for both deltas;
<= 1.5 sigma at every a != 0. At a = 0 the receiver samples synchronously
(N_sym = 20 samples per symbol); the mechanism is not yet understood.

Usage:  python check_detector_offset.py      (~30 s, 4 processes)
"""

from multiprocessing import Pool

import numpy as np

from ocdm_doppler import OCDMSignal, preset_noisy_22jan
from ocdm_doppler import discriminator as disc

N_C, N_SEG = 24000, 2048
# noiseless K_d for Delta = {0}: 175.5 at delta = 0.25/N_seg
# (results/characteristic.md); 368.9 at 0.5/N_seg (Step 3, results/scurve.md)
KD = {0.25: 175.5, 0.5: 368.9}


def one(args):
    a, seed, dmul = args
    p = preset_noisy_22jan(float("inf")).with_(a_true=a, cbsfs_deltas=(0,))
    M = N_C // p.N_sym
    y = OCDMSignal(p, n_symbols=M + 2, seed=seed).rx_nominal(
        M * p.N_sym, noiseless=True)
    return disc.early_late(y, a, p, delta=dmul / N_SEG, N_seg=N_SEG)


def main():
    seeds = range(5000, 5256)
    with Pool(4) as pool:
        for dmul in (0.25, 0.5):
            for a in (0.0, 2.5e-4, 5e-4, 1e-3):
                e = np.array(pool.map(one, [(a, s, dmul) for s in seeds]))
                b = e.mean() / KD[dmul]
                se = e.std() / KD[dmul] / np.sqrt(e.size)
                print(f"delta={dmul}/N_seg  a={a:.2e}: mean e/K_d = "
                      f"{b:+.2e} +/- {se:.1e}  ({b / se:+.1f} sigma)",
                      flush=True)


if __name__ == "__main__":
    main()
