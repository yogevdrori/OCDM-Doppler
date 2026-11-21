"""
Step 7: is the detector bias leakage from neighbouring spectral lines?
N_seg = 2048 vs 1920 (= 6 * 320 = 96 * 20, a multiple of both mean-power
periods).

Hypothesis. The segmented CAF ([CBSFS] 27) uses a rectangular window of
N_seg samples, so a spectral line of |y|^2 at alpha_0 leaks into
C(alpha) with a Dirichlet kernel whose nulls are at alpha - alpha_0 =
m / N_seg. The early / late probes sit at k (1 + a_hat)/P_samp +/- delta;
the DC line (mean power, enlarged by the noise power) and the neighbouring
harmonics are (k N_seg / P_samp) bins away. With a non-integer number of
bins (2048/320 = 6.4, 2048/20 = 102.4) the leakage into the two probes is
unequal -> a bias, growing as the noise raises the DC line. With
N_seg = 1920 the distances are integers (6, 96 bins), the leakage into
the two probes is nearly symmetric, and the bias should shrink.

1. OTFS-frame, k = 2 and 3: open-loop characteristic (a_hat = 0, true a on
   the Part C grid, 32 blocks each), noiseless and WP, N_seg 2048 vs 1920.
   The 2048 values reproduce Part C (same blocks; asserted).
2. OCDM, a = 5.1542e-4 and -4.8458e-4:
   (a) open loop near lock: e at a_hat = a and a +/- 2e-4 on the same
       blocks for both N_seg; K_d from the +/-2e-4 secant (re-measured for
       1920); bias = mean e(a_hat = a) / K_d;
   (b) short closed loop, first order, pole 0.99, mu = 0.01 / K_d(N_seg,
       noiseless), R = 2 x L = 1500 after 900 blocks;
   SNR inf, 4 (WP), 0 dB in lockstep. Both N_seg see the same data and
   noise, so their difference is a paired estimate.
Standard errors: open loop from the block-to-block scatter (independent
blocks); closed loop from batch means (batches of 300 blocks = 3 loop time
constants).

Output: results/nseg_leakage.md, raw results/nseg_raw/.
Usage: python check_nseg_leakage.py [--quick] [--resume] [--procs 3]
"""

from __future__ import annotations

import argparse
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ocdm_doppler import preset_noisy_22jan
from ocdm_doppler import loop_filters as lf
from ocdm_doppler.doppler_profile import ConstantDoppler
from ocdm_doppler.waveforms import OCDMWaveform, OTFSWaveform
from ocdm_doppler.channel import single_path
from ocdm_doppler.loop_wf import run_loop_wf, open_loop_e
from study_otfs import char_grid, char_metrics

ROOT = Path(__file__).parent
NSEGS = (2048, 1920)
SNRS = (float("inf"), 4.0, 0.0)
SNRS_FRAME = (float("inf"), 4.0)
A_OCDM = (5.1542e-4, -4.8458e-4)
D_OFF = 2e-4
BATCH = 300


def base():
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,))


def snr_s(x):
    return "∞" if np.isinf(x) else (f"{x:g} (WP)" if x == 4.0 else f"{x:g}")


# ---------------------------------------------------------------------------
def _frame_task(args):
    f, a, n_blocks = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    p = base().with_(a_true=a)
    out = {}
    for N in NSEGS:
        out[f"e{N}"] = open_loop_e(p, OTFSWaveform(guard="frame"),
                                   single_path(p), ConstantDoppler(a), [0.0],
                                   n_blocks, snrs=SNRS_FRAME, ks=(2, 3),
                                   N_seg=N, seed=1000)[..., 0]  # [s, i, k]
    np.savez(f, **out)
    return f, time.time() - t0


def _ocdm_ol_task(args):
    f, a, start, n = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    p = base().with_(a_true=a)
    out = {}
    for N in NSEGS:
        out[f"e{N}"] = open_loop_e(p, OCDMWaveform(), single_path(p),
                                   ConstantDoppler(a),
                                   a + np.array([-D_OFF, 0.0, D_OFF]), n,
                                   snrs=SNRS, ks=(1,), N_seg=N, seed=1000,
                                   start_block=start)[:, :, 0, :]
    np.savez(f, **out)                                  # [s, i, offset]
    return f, time.time() - t0


def _ocdm_loop_task(args):
    f, a, N, mu, seed, n = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    p = base().with_(a_true=a)
    filt = [lf.make_filter(mu, 0.0, 1.0, a_hat0=a)]       # mu = g1 / 1
    r = run_loop_wf(p, OCDMWaveform(), single_path(p), ConstantDoppler(a),
                    filt, n, snrs=SNRS, N_seg=N, seed=seed)
    np.savez(f, a_hat=r.a_hat[0])                       # [s, i]
    return f, time.time() - t0


BUDGET = [540.0]


def run(fn, tasks, procs, label, t0):
    """Run the unfinished tasks (each saves its own file). Stops (returns
    False) when the next wave of tasks would not finish within the budget;
    tasks still running are then lost and redone on --resume."""
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
            if (i + 1 < len(todo)
                    and time.time() - t0 + 1.1 * longest > BUDGET[0]):
                print("  budget reached; rerun with --resume", flush=True)
                pool.terminate()
                return False
    return True


def batch_se(x, b):
    """x[R, L]: std err of the mean from batch means of length b."""
    nb = x.shape[1] // b
    m = x[:, :nb * b].reshape(x.shape[0], nb, b).mean(-1).ravel()
    return float(m.std(ddof=1) / math.sqrt(m.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--budget", type=float, default=540.0)
    args = ap.parse_args()
    BUDGET[0] = args.budget
    t0 = time.time()
    q = args.quick
    out = ROOT / "results" / ("quick" if q else "")
    raw = out / "nseg_raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for f in raw.glob("*.npz"):
            f.unlink()
    a_idx = (0, 5, 10) if q else tuple(range(11))
    a_grid = char_grid()[list(a_idx)]
    n_char = 4 if q else 32
    n_ol, chunk = (60, 30) if q else (3000, 250)
    D, L, R = (60, 120, 2) if q else (900, 1500, 2)

    # 1. OTFS-frame characteristic
    if not run(_frame_task, [(str(raw / f"frame_{i}.npz"), float(a), n_char)
                             for i, a in zip(a_idx, a_grid)], args.procs,
               "OTFS-frame characteristic", t0):
        return
    # 2a. OCDM open loop near lock
    if not run(_ocdm_ol_task, [(str(raw / f"ocdm_ol_{j}_{s}.npz"), a, s,
                                min(chunk, n_ol - s))
                               for j, a in enumerate(A_OCDM)
                               for s in range(0, n_ol, chunk)], args.procs,
               "OCDM open loop", t0):
        return
    ol = {}
    for j, a in enumerate(A_OCDM):
        for N in NSEGS:
            ol[j, N] = np.concatenate(
                [np.load(raw / f"ocdm_ol_{j}_{s}.npz")[f"e{N}"]
                 for s in range(0, n_ol, chunk)], axis=1)  # [s, i, off]
    # K_d per N_seg (noiseless, pooled over both a): e(a - d) - e(a + d)
    Kd = {N: float(np.mean([(ol[j, N][0, :, 0] - ol[j, N][0, :, 2]).mean()
                            / (2 * D_OFF) for j in range(2)]))
          for N in NSEGS}
    # 2b. OCDM closed loop
    if not run(_ocdm_loop_task, [(str(raw / f"ocdm_loop_{j}_{N}_{r}.npz"),
                                  a, N, 0.01 / Kd[N], 1000 + r, D + L)
                                 for j, a in enumerate(A_OCDM) for N in NSEGS
                                 for r in range(R)], args.procs,
               "OCDM closed loop", t0):
        return

    # ------------------------------------------------------------------ report
    Lr = ["# Is the detector bias leakage? N_seg = 2048 vs 1920", "",
          "Generated by `check_nseg_leakage.py`" + (" --quick" if q else "")
          + ". Hypothesis and method in the script docstring. δ = 0.25/N_seg "
          "in both cases (the default). 1920 = 6·320 = 96·20: the DC line "
          "and the neighbouring harmonics are an integer number of CAF bins "
          "(6 for OTFS-frame, 96 for OCDM) from the harmonic, instead of "
          "6.4 / 102.4 with 2048.", ""]
    # part 1
    Ef = {N: np.array([np.load(raw / f"frame_{i}.npz")[f"e{N}"]
                       for i in a_idx]) for N in NSEGS}   # [a, s, i, k]
    same = "not checked (--quick)"
    if not q:
        pc = np.array([np.load(ROOT / "results" / "otfs_raw" /
                               f"C_char_OTFS-frame_{i}.npz")["e"]
                       for i in a_idx])                   # [a, s5, i, k5]
        same = ("yes" if np.array_equal(pc[:, [0, 2]][..., [1, 2]], Ef[2048])
                else "NO")
    Lr += ["## 1. OTFS-frame, open-loop characteristic (â = 0, true a on "
           f"{len(a_grid)} values × {n_char} blocks)", "",
           "offset = the a at which the mean e is zero, i.e. the lock-point "
           "bias (Part C: closed-loop bias ≈ offset); ± std err of the fit "
           f"intercept. N_seg = 2048 values identical to Part C: {same}.", "",
           "| k | SNR | N_seg | K_d | offset (in a) ± se | linearity | "
           "σ_e/K_d |", "|---|---|---|---|---|---|---|"]
    for kk, k in enumerate((2, 3)):
        for s, snr in enumerate(SNRS_FRAME):
            for N in NSEGS:
                E = Ef[N][:, s, :, kk]                    # [a, i]
                m = char_metrics(a_grid, E[:, None, :])[0]
                x = np.repeat(a_grid, E.shape[1])
                y = E.ravel()
                X = np.vstack([x, np.ones_like(x)]).T
                coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
                s2 = float(np.sum((y - X @ coef) ** 2) / (y.size - 2))
                cov = s2 * np.linalg.inv(X.T @ X)
                se_off = math.sqrt(cov[1, 1]) / abs(coef[0])
                Lr.append(f"| {k} | {snr_s(snr)} | {N} | {m['K_d']:.2f} | "
                          f"{m['offset_a']:+.2e} ± {se_off:.1e} | "
                          f"{m['lin']:.1%} | {m['s_ek']:.2e} |")
    # part 2a
    Lr += ["", f"## 2a. OCDM, open loop near lock ({n_ol} independent "
           "blocks per a, the same blocks for both N_seg)", "",
           f"K_d (noiseless, ±{D_OFF:g} secant): N_seg = 2048: "
           f"{Kd[2048]:.1f}; 1920: {Kd[1920]:.1f}. Bias = mean e(â = a) / "
           "K_d(N_seg, that SNR); paired difference 1920 − 2048 on the same "
           "blocks.", "",
           "| a | SNR | K_d 2048 / 1920 | bias 2048 ± se | bias 1920 ± se | "
           "1920 − 2048 ± se |", "|---|---|---|---|---|---|"]
    for j, a in enumerate(A_OCDM):
        for s, snr in enumerate(SNRS):
            row, b = [], {}
            for N in NSEGS:
                e = ol[j, N][s]
                K = (e[:, 0] - e[:, 2]).mean() / (2 * D_OFF)
                b[N] = e[:, 1] / K
                row.append(K)
            dd = b[1920] - b[2048]
            Lr.append(
                f"| {a:+.4e} | {snr_s(snr)} | {row[0]:.1f} / {row[1]:.1f} | "
                f"{b[2048].mean():+.2e} ± "
                f"{b[2048].std(ddof=1) / math.sqrt(b[2048].size):.1e} | "
                f"{b[1920].mean():+.2e} ± "
                f"{b[1920].std(ddof=1) / math.sqrt(b[1920].size):.1e} | "
                f"{dd.mean():+.2e} ± {dd.std(ddof=1) / math.sqrt(dd.size):.1e}"
                " |")
    # part 2b
    Lr += ["", f"## 2b. OCDM, closed loop (pole 0.99, μ = 0.01/K_d(N_seg); "
           f"R = {R} × L = {L} after {D} blocks; the same data and noise for "
           "both N_seg)", "",
           f"Std err from batch means ({BATCH} blocks).", "",
           "| a | SNR | bias 2048 ± se | bias 1920 ± se | 1920 − 2048 ± se | "
           "std 2048 / 1920 |", "|---|---|---|---|---|---|"]
    bsz = min(BATCH, L // 4)
    for j, a in enumerate(A_OCDM):
        A = {N: np.array([np.load(raw / f"ocdm_loop_{j}_{N}_{r}.npz")["a_hat"]
                          for r in range(R)]) for N in NSEGS}  # [R, s, i]
        for s, snr in enumerate(SNRS):
            eps = {N: A[N][:, s, D:D + L] - a for N in NSEGS}
            dd = eps[1920] - eps[2048]
            Lr.append(
                f"| {a:+.4e} | {snr_s(snr)} | {eps[2048].mean():+.2e} ± "
                f"{batch_se(eps[2048], bsz):.1e} | {eps[1920].mean():+.2e} ± "
                f"{batch_se(eps[1920], bsz):.1e} | {dd.mean():+.2e} ± "
                f"{batch_se(dd, bsz):.1e} | {eps[2048].std():.2e} / "
                f"{eps[1920].std():.2e} |")
    Lr += ["", f"Runtime {time.time() - t0:.0f} s.", ""]
    (out / "nseg_leakage.md").write_text("\n".join(Lr), encoding="utf-8")
    print("\n".join(Lr))


if __name__ == "__main__":
    main()
