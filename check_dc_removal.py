"""
Step 7: does removing the DC line of |y|^2 remove the detector bias?

Detector variant ocdm_doppler/cbsfs_dc.cost_function_dc: the block mean of
the lag product (Delta = 0: |y|^2) is subtracted before the segmented CAF.
Compared, open loop and on the same blocks (same data and noise), with the
current detector (cbsfs.cost_function, unchanged):

1. OCDM, N_seg = 2048 and 1920, a = 5.1542e-4 and -4.8458e-4, SNR inf, 4
   (WP), 0, -5 dB: e at a_hat = a and a +/- 2e-4 on the first 1500 blocks of
   the N_seg check (check_nseg_leakage.py; its current-detector values are
   reproduced exactly). K_d from the +/-2e-4 secant per SNR, bias =
   mean e(a_hat = a) / K_d, sigma_e / K_d; DC-removed minus current paired.
2. OTFS-frame, N_seg = 1920, k = 2 and 3, noiseless and WP: open-loop
   characteristic (a_hat = 0, the Part C grid, 32 blocks): offset (lock-
   point bias), linearity, K_d, sigma_e / K_d.
0. cost_function_dc(remove_dc=False) == cbsfs.cost_function (bit for bit).

Output: results/dc_removal.md, raw results/dc_raw/.
Usage: python check_dc_removal.py [--quick] [--resume] [--procs 3]
"""

from __future__ import annotations

import argparse
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ocdm_doppler import preset_noisy_22jan
from ocdm_doppler import cbsfs
from ocdm_doppler import discriminator as disc
from ocdm_doppler.cbsfs_dc import cost_function_dc
from ocdm_doppler.channel import single_path, rx_block
from ocdm_doppler.doppler_profile import ConstantDoppler
from ocdm_doppler.waveforms import OCDMWaveform, OTFSWaveform
from ocdm_doppler.wf_params import detector_params
from study_otfs import char_grid, char_metrics

ROOT = Path(__file__).parent
N_C = 24000
A_OCDM = (5.1542e-4, -4.8458e-4)
D_OFF = 2e-4
SNRS_O = (float("inf"), 4.0, 0.0, -5.0)
SNRS_F = (float("inf"), 4.0)
NSEGS = (2048, 1920)
DETS = ("current", "DC removed")


def base():
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,))


def snr_s(x):
    return "∞" if np.isinf(x) else (f"{x:g} (WP)" if x == 4.0 else f"{x:g}")


def _amp(p, s):
    return 0.0 if np.isinf(s) else math.sqrt(p.sigma_D2 / 10 ** (s / 10) / 2)


def blocks_e(wf, a, a_hats, ks, snrs, nsegs, start, n, seed=1000):
    """e[det, nseg, snr, block, k, a_hat] on blocks start .. start + n - 1
    (the same received blocks and noise as loop_wf.open_loop_e)."""
    p = base().with_(a_true=a)
    ws = wf.signal(p, seed)
    dp = detector_params(p, ws.P_samp)
    prof = ConstantDoppler(a)
    a_hats = np.atleast_1d(a_hats)
    out = np.empty((2, len(nsegs), len(snrs), n, len(ks), a_hats.size))
    for i in range(n):
        b = start + i
        y0 = rx_block(ws, single_path(p), p, N_C, prof, start_sample=b * N_C)
        rng = np.random.default_rng([seed, 0xA015E, b])
        w = rng.standard_normal(N_C) + 1j * rng.standard_normal(N_C)
        for s, snr in enumerate(snrs):
            y = y0 if np.isinf(snr) else y0 + _amp(p, snr) * w
            for ni, N in enumerate(nsegs):
                delta = 0.25 / N
                ap = np.concatenate([disc.predicted_alpha(a_hats, dp, k)
                                     for k in ks])
                al = np.concatenate([ap + delta, ap - delta])
                m = ap.size
                for di, fn in enumerate((cbsfs.cost_function,
                                         cost_function_dc)):
                    c = fn(y, al, dp, N_seg=N)
                    out[di, ni, s, i] = disc.error_from_costs(
                        c[:m], c[m:]).reshape(len(ks), a_hats.size)
    return out


def _ocdm_task(args):
    f, j, start, n = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    a = A_OCDM[j]
    e = blocks_e(OCDMWaveform(), a, a + np.array([-D_OFF, 0.0, D_OFF]), (1,),
                 SNRS_O, NSEGS, start, n)
    np.savez(f, e=e[:, :, :, :, 0, :])          # [det, nseg, snr, i, off]
    return f, time.time() - t0


def _frame_task(args):
    f, i, a, n = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    e = blocks_e(OTFSWaveform(guard="frame"), a, [0.0], (2, 3), SNRS_F,
                 (1920,), 0, n)
    np.savez(f, e=e[:, 0, :, :, :, 0])          # [det, snr, i, k]
    return f, time.time() - t0


def run(fn, tasks, procs, label, t0, budget):
    todo = [t for t in tasks if not Path(t[0]).exists()]
    if not todo:
        return True
    print(f"  {label}: {len(todo)} tasks", flush=True)
    longest = 0.0
    with Pool(procs) as pool:
        for i, (f, dur) in enumerate(pool.imap_unordered(fn, todo)):
            longest = max(longest, dur)
            print(f"    {i + 1}/{len(todo)} [{dur:.0f}s, total "
                  f"{time.time() - t0:.0f}s]", flush=True)
            if i + 1 < len(todo) and time.time() - t0 + 1.1 * longest > budget:
                print("  budget reached; rerun with --resume", flush=True)
                pool.terminate()
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--budget", type=float, default=540.0)
    args = ap.parse_args()
    t0 = time.time()
    q = args.quick
    out = ROOT / "results" / ("quick" if q else "")
    raw = out / "dc_raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for f in raw.glob("*.npz"):
            f.unlink()
    n_ol, chunk = (40, 20) if q else (1500, 125)
    a_idx = (0, 5, 10) if q else tuple(range(11))
    a_grid = char_grid()[list(a_idx)]
    n_char = 4 if q else 32

    # 0. bit identity of the variant without DC removal
    p = base().with_(a_true=A_OCDM[0])
    ws = OCDMWaveform().signal(p, 1000)
    y = rx_block(ws, single_path(p), p, N_C, ConstantDoppler(A_OCDM[0]))
    al = np.array([0.04, 0.05, 0.0500257, 0.06])
    d0 = max(float(np.max(np.abs(cost_function_dc(y, al, p, N_seg=N,
                                                  remove_dc=False)
                                 - cbsfs.cost_function(y, al, p, N_seg=N))))
             for N in NSEGS)

    if not run(_frame_task, [(str(raw / f"frame_{i}.npz"), i, float(a),
                              n_char) for i, a in zip(a_idx, a_grid)],
               args.procs, "OTFS-frame", t0, args.budget):
        return
    if not run(_ocdm_task, [(str(raw / f"ocdm_{j}_{s}.npz"), j, s,
                             min(chunk, n_ol - s))
                            for j in range(2) for s in range(0, n_ol, chunk)],
               args.procs, "OCDM", t0, args.budget):
        return

    E = {j: np.concatenate([np.load(raw / f"ocdm_{j}_{s}.npz")["e"]
                            for s in range(0, n_ol, chunk)], axis=3)
         for j in range(2)}                   # [det, nseg, snr, i, off]
    F = np.array([np.load(raw / f"frame_{i}.npz")["e"] for i in a_idx])
    # [a, det, snr, i, k]
    rep = "not checked (--quick)"
    if not q:
        nr = ROOT / "results" / "nseg_raw"
        ok = True
        for j in range(2):
            old = {N: np.concatenate([np.load(nr / f"ocdm_ol_{j}_{s}.npz")
                                      [f"e{N}"] for s in range(0, 3000, 250)],
                                     axis=1)[:, :n_ol] for N in NSEGS}
            for ni, N in enumerate(NSEGS):
                ok &= np.array_equal(E[j][0, ni, :3], old[N])
        oldf = np.array([np.load(nr / f"frame_{i}.npz")["e1920"]
                         for i in a_idx])     # [a, s, i, k]
        ok &= np.array_equal(F[:, 0], oldf)
        rep = "yes" if ok else "NO"

    Lr = ["# DC-line removal in the detector: open-loop check", "",
          "Generated by `check_dc_removal.py`" + (" --quick" if q else "")
          + ". Variant `ocdm_doppler/cbsfs_dc.py`: the block mean of |y|² "
          "is subtracted before the segmented CAF; everything else as the "
          "current detector (δ = 0.25/N_seg). Both detectors on the same "
          "blocks.", "",
          f"- cost_function_dc(remove_dc=False) vs cbsfs.cost_function: max "
          f"|diff| = {d0:.1e}",
          f"- current-detector values identical to the N_seg check "
          f"(results/nseg_leakage.md, same blocks): {rep}", "",
          f"## 1. OCDM, open loop near lock ({n_ol} blocks per a)", "",
          "K_d from the ±2e-4 secant at that SNR; bias = mean e(â = a)/K_d "
          "± std err; DC − cur = paired difference of the biases on the same "
          "blocks.", "",
          "| a | N_seg | SNR | K_d cur / DC | bias cur | bias DC removed | "
          "DC − cur | σ_e/K_d cur / DC |", "|---|---|---|---|---|---|---|---|"]
    for j, a in enumerate(A_OCDM):
        for ni, N in enumerate(NSEGS):
            for s, snr in enumerate(SNRS_O):
                K, b = [], []
                for di in range(2):
                    e = E[j][di, ni, s]                     # [i, off]
                    Kd = (e[:, 0] - e[:, 2]).mean() / (2 * D_OFF)
                    K.append(Kd)
                    b.append(e[:, 1] / Kd)
                dd = b[1] - b[0]
                se = [x.std(ddof=1) / math.sqrt(x.size) for x in (*b, dd)]
                Lr.append(
                    f"| {a:+.4e} | {N} | {snr_s(snr)} | {K[0]:.1f} / "
                    f"{K[1]:.1f} | {b[0].mean():+.2e} ± {se[0]:.1e} | "
                    f"{b[1].mean():+.2e} ± {se[1]:.1e} | {dd.mean():+.2e} ± "
                    f"{se[2]:.1e} | {b[0].std(ddof=1):.2e} / "
                    f"{b[1].std(ddof=1):.2e} |")
    Lr += ["", "Does N_seg still matter once DC is removed? Paired 1920 − "
           "2048 bias difference on the same blocks, per detector:", "",
           "| a | SNR | current: 1920 − 2048 | DC removed: 1920 − 2048 |",
           "|---|---|---|---|"]
    for j, a in enumerate(A_OCDM):
        for s, snr in enumerate(SNRS_O):
            cells = []
            for di in range(2):
                bb = []
                for ni in range(2):
                    e = E[j][di, ni, s]
                    bb.append(e[:, 1] / ((e[:, 0] - e[:, 2]).mean()
                                         / (2 * D_OFF)))
                dd = bb[1] - bb[0]
                cells.append(f"{dd.mean():+.2e} ± "
                             f"{dd.std(ddof=1) / math.sqrt(dd.size):.1e}")
            Lr.append(f"| {a:+.4e} | {snr_s(snr)} | {cells[0]} | {cells[1]} |")
    Lr += ["", "## 2. OTFS-frame, N_seg = 1920, open-loop characteristic "
           f"(â = 0, true a on {len(a_grid)} values × {n_char} blocks)", "",
           "offset = lock-point bias ± std err of the fit intercept; "
           "linearity = max |residual of the mean e| / (K_d·a_max).", "",
           "DC − cur = paired offset difference: intercept of a linear fit "
           "of e_DC/K_d,DC − e_cur/K_d,cur on the same blocks.", "",
           "| k | SNR | detector | K_d | offset ± se | linearity | σ_e/K_d |",
           "|---|---|---|---|---|---|---|"]
    paired = []
    for kk, k in enumerate((2, 3)):
        for s, snr in enumerate(SNRS_F):
            Kc = [char_metrics(a_grid, F[:, di, s, :, kk][:, None, :])[0]
                  ["K_d"] for di in range(2)]
            dE = F[:, 1, s, :, kk] / Kc[1] - F[:, 0, s, :, kk] / Kc[0]
            x = np.repeat(a_grid, dE.shape[1])
            X = np.vstack([x, np.ones_like(x)]).T
            cf, *_ = np.linalg.lstsq(X, dE.ravel(), rcond=None)
            s2 = float(np.sum((dE.ravel() - X @ cf) ** 2) / (x.size - 2))
            paired.append((k, snr, cf[1],
                           math.sqrt((s2 * np.linalg.inv(X.T @ X))[1, 1])))
            for di, dname in enumerate(DETS):
                Ea = F[:, di, s, :, kk]                     # [a, i]
                m = char_metrics(a_grid, Ea[:, None, :])[0]
                x = np.repeat(a_grid, Ea.shape[1])
                X = np.vstack([x, np.ones_like(x)]).T
                coef, *_ = np.linalg.lstsq(X, Ea.ravel(), rcond=None)
                s2 = float(np.sum((Ea.ravel() - X @ coef) ** 2)
                           / (x.size - 2))
                se_off = math.sqrt((s2 * np.linalg.inv(X.T @ X))[1, 1]) / abs(
                    coef[0])
                Lr.append(f"| {k} | {snr_s(snr)} | {dname} | {m['K_d']:.2f} "
                          f"| {m['offset_a']:+.2e} ± {se_off:.1e} | "
                          f"{m['lin']:.1%} | {m['s_ek']:.2e} |")
    Lr += ["", "| k | SNR | offset DC − cur (paired) |", "|---|---|---|"]
    for k, snr, d_, se_ in paired:
        # e/K_d = (a - a_hat) + offset: the intercept of the difference is
        # offset_DC - offset_cur
        Lr.append(f"| {k} | {snr_s(snr)} | {d_:+.2e} ± {se_:.1e} |")
    Lr += ["", f"Runtime {time.time() - t0:.0f} s.", ""]
    (out / "dc_removal.md").write_text("\n".join(Lr), encoding="utf-8")
    print("\n".join(Lr))


if __name__ == "__main__":
    main()
