"""
Step 7: the detector on other multi-basis waveforms (OTFS) and in multipath
(ocdm_doppler/waveforms.py, channel.py, loop_wf.py).

Part B  OCDM vs OTFS with a zero guard after every sub-symbol (P_samp = 20):
        open-loop characteristic (a_hat = 0, true a swept over the prior
        range, SNR inf / 20 / 4 (WP) / 0 / -5 dB): K_d (linear fit),
        linearity, sigma_e / K_d; noiseless S-curve for the pull-in range;
        closed loop (first order, pole 0.99, mu = 0.01 / K_d measured at the
        WP for that waveform) at a = 5.1542e-4 and -4.8458e-4, noiseless and
        WP in lockstep: std / bias / RMSE vs the linear prediction.
Part C  OTFS with one guard per frame, harmonic k = 1..5   -- added later
Part D  multipath                                          -- added later

Settings as Steps 4-6: 22-Jan preset (BPSK), Delta = {0}, N_c = 24000,
N_seg = 2048, delta = 0.25 / N_seg, seeds 1000 + index, non-integer shift
a N_c per block.

Raw data in results/otfs_raw/ (one file per task / realisation, saved as
it completes); --resume continues, --replot redoes tables and figures,
--budget stops between rounds, --quick for timing.

Usage:
    python study_otfs.py --part B --quick
    python study_otfs.py --part B [--resume] [--procs 3] [--budget 540]
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocdm_doppler import preset_noisy_22jan
from ocdm_doppler import loop_filters as lf
from ocdm_doppler.doppler_profile import ConstantDoppler
from ocdm_doppler.waveforms import OCDMWaveform, OTFSWaveform
from ocdm_doppler.channel import single_path
from ocdm_doppler.loop_wf import run_loop_wf, open_loop_e

ROOT = Path(__file__).parent
N_C, N_SEG = 24000, 2048
WP_DB = 4.0
SNRS_CHAR = (float("inf"), 20.0, WP_DB, 0.0, -5.0)
SNRS_LOOP = (float("inf"), WP_DB)
POLE = 0.99
SEG = 300


def base():
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,))


def make_wf(name):
    return {"OCDM": OCDMWaveform(),
            "OTFS-symbol": OTFSWaveform(guard="symbol"),
            "OTFS-frame": OTFSWaveform(guard="frame")}[name]


def char_grid():
    """11 true values over the prior range with a non-integer shift a N_c."""
    # 4.5 samples per step: fractional shifts 0.37 / 0.13 alternately
    a = -9.8458e-4 + 1.875e-4 * np.arange(11)     # ... +8.9042e-4
    frac = np.abs(a * N_C - np.round(a * N_C))
    assert frac.min() > 0.05, frac
    return a


def snr_s(x):
    return "∞" if np.isinf(x) else (f"{x:g} (WP)" if x == WP_DB else f"{x:g}")


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def _char_task(args):
    f, wf_name, a, n_blocks, seed, ks = args
    if f.exists():
        return str(f), 0.0
    t0 = time.time()
    p = base().with_(a_true=a)
    e = open_loop_e(p, make_wf(wf_name), single_path(p), ConstantDoppler(a),
                    [0.0], n_blocks, snrs=SNRS_CHAR, ks=ks, seed=seed)
    np.savez(f, e=e[..., 0])                      # [snr, block, k]
    # (Part B files hold one k; load_char returns [a, snr, block, k])
    return str(f), time.time() - t0


def _scurve_task(args):
    f, wf_name, a, offsets, n_blocks, seed, ks = args
    if f.exists():
        return str(f), 0.0
    t0 = time.time()
    p = base().with_(a_true=a)
    e = open_loop_e(p, make_wf(wf_name), single_path(p), ConstantDoppler(a),
                    a + offsets, n_blocks, snrs=(float("inf"),), ks=ks,
                    seed=seed)
    np.savez(f, e=e[0], offsets=offsets)          # [block, k, offset]
    return str(f), time.time() - t0


def _loop_segment(args):
    wf_name, a, mus, ks, seed, start, n, last = args
    p = base().with_(a_true=a)
    # make_filter(g1, 0, K_d0 = 1) is a FirstOrderFilter with mu = g1
    filters = [lf.make_filter(mu, 0.0, 1.0, a_hat0=a) for mu in mus]
    states = None
    if last is not None:
        states = [[(last[m, s],) for s in range(len(SNRS_LOOP))]
                  for m in range(len(filters))]
    r = run_loop_wf(p, make_wf(wf_name), single_path(p), ConstantDoppler(a),
                    filters, n, snrs=SNRS_LOOP, ks=ks, seed=seed,
                    start_block=start, states=states)
    return r.a_hat[..., 1:], r.e


def run_tasks(fn, tasks, procs, label, t0, budget):
    todo = [t for t in tasks if not Path(t[0]).exists()]
    if not todo:
        return True
    print(f"  {label}: {len(todo)} tasks", flush=True)
    with Pool(procs) as pool:
        for i, (f, dur) in enumerate(pool.imap_unordered(fn, todo)):
            if (i + 1) % max(1, len(todo) // 8) == 0 or i + 1 == len(todo):
                print(f"    {i + 1}/{len(todo)} [{dur:.0f}s last, total "
                      f"{time.time() - t0:.0f}s]", flush=True)
            if time.time() - t0 > budget:
                print("  budget reached; rerun with --resume", flush=True)
                pool.terminate()
                return False
    return True


def run_loops(cases, raw, procs, t0, budget):
    """cases: name -> dict(wf, a, mus, ks, R, n_blocks). Rounds of SEG
    blocks over all unfinished realisations; one file per realisation."""
    state = {}
    for name, c in cases.items():
        for r in range(c["R"]):
            f = raw / f"{name}__r{r}.npz"
            if f.exists():
                st = np.load(f)
                state[name, r] = (st["a_hat"], st["e"])
            else:
                a0 = np.full((len(c["mus"]), len(SNRS_LOOP), 1), c["a"])
                state[name, r] = (a0, np.zeros(a0.shape[:2] + (0,)))
    last_dur = 0.0
    with Pool(procs) as pool:
        while True:
            todo = [(nm, r) for (nm, r), (A, _) in state.items()
                    if A.shape[-1] - 1 < cases[nm]["n_blocks"]]
            if not todo:
                return True
            if time.time() - t0 + 1.1 * last_dur > budget:
                print(f"  budget reached; rerun with --resume "
                      f"[{time.time() - t0:.0f}s]", flush=True)
                return False
            ts = time.time()
            tasks = []
            for nm, r in todo:
                A, _ = state[nm, r]
                c = cases[nm]
                done = A.shape[-1] - 1
                n = min(SEG, c["n_blocks"] - done)
                tasks.append((c["wf"], c["a"], c["mus"], c["ks"], 1000 + r,
                              done, n, None if done == 0 else A[..., -1]))
            for (nm, r), (sa, se) in zip(todo, pool.map(_loop_segment, tasks,
                                                        chunksize=1)):
                A, E = state[nm, r]
                A, E = np.concatenate([A, sa], -1), np.concatenate([E, se], -1)
                state[nm, r] = (A, E)
                np.savez(raw / f"{nm}__r{r}.npz", a_hat=A, e=E)
            last_dur = time.time() - ts
            nb = sum(t[6] for t in tasks)
            print(f"  round: {len(tasks)} realisations, {nb} blocks "
                  f"[{last_dur:.0f}s = {last_dur * procs / nb * 1e3:.0f} "
                  f"ms/block/worker, total {time.time() - t0:.0f}s]",
                  flush=True)


# ---------------------------------------------------------------------------
# characteristic analysis (shared by Parts B and C)
# ---------------------------------------------------------------------------
def char_metrics(a_grid, E):
    """E[a_index, snr, block] -> per SNR: K_d (fit over the prior range),
    offset, linearity, sigma_e, sigma_e / K_d."""
    out = []
    for s in range(E.shape[1]):
        me = E[:, s].mean(1)
        K, c = np.polyfit(a_grid, me, 1)
        res = me - (K * a_grid + c)
        sig = float(np.sqrt(np.mean(E[:, s].var(1, ddof=1))))
        out.append(dict(K_d=float(K), offset=float(c), offset_a=float(c / K),
                        lin=float(np.max(np.abs(res)) / abs(K * a_grid[-1])),
                        sigma_e=sig, s_ek=sig / abs(K)))
    return out


def pull_in(offsets, me):
    """First sign changes of the mean S-curve away from the lock point."""
    pos = offsets > 0
    neg = offsets < 0
    # e has the sign of -(offset): e > 0 for a_hat < a (offset < 0)
    up = offsets[pos][np.flatnonzero(me[pos] >= 0)]
    dn = offsets[neg][np.flatnonzero(me[neg] <= 0)]
    return (float(dn.max()) if dn.size else float("nan"),
            float(up.min()) if up.size else float("nan"))


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
WF_B = ("OCDM", "OTFS-symbol")
A_LOOP = (5.1542e-4, -4.8458e-4)


def design_b(quick):
    if quick:
        return dict(n_char=4, a_idx=(0, 5, 10), n_s=2, n_off=21, D=30, L=60,
                    R=1)
    return dict(n_char=32, a_idx=tuple(range(11)), n_s=8, n_off=121, D=900,
                L=1500, R=2)


def part_b(args, out, raw, figd, t0):
    d = design_b(args.quick)
    a_grid = char_grid()[list(d["a_idx"])]
    pin = 20 / N_SEG                                   # P_samp / (k N_seg)
    offsets = np.linspace(-1.5 * pin, 1.5 * pin, d["n_off"])
    if not args.replot:
        ct = [(raw / f"B_char_{w}_{i}.npz", w, float(a), d["n_char"], 1000,
               (1,)) for w in WF_B for i, a in zip(d["a_idx"], a_grid)]
        st = [(raw / f"B_scurve_{w}.npz", w, A_LOOP[0], offsets, d["n_s"],
               1001, (1,)) for w in WF_B]
        if not run_tasks(_char_task, ct + [], args.procs,
                         "characteristic", t0, args.budget):
            return
        if not run_tasks(_scurve_task, st, args.procs, "S-curve", t0,
                         args.budget):
            return
    # K_d per waveform (at the WP) -> loop gains
    char = {}
    for w in WF_B:
        E = np.array([np.load(raw / f"B_char_{w}_{i}.npz")["e"][:, :, 0]
                      for i in d["a_idx"]])           # [a, snr, block]
        char[w] = char_metrics(a_grid, E)
    s_wp = SNRS_CHAR.index(WP_DB)
    cases = {}
    for w in WF_B:
        mu = (1 - POLE) / char[w][s_wp]["K_d"]
        for a in A_LOOP:
            cases[f"B_loop_{w}_a{a:+.4e}"] = dict(
                wf=w, a=a, mus=[mu], ks=1, R=d["R"], n_blocks=d["D"] + d["L"])
    if not args.replot and not run_loops(cases, raw, args.procs, t0,
                                         args.budget):
        return
    analyse_b(d, a_grid, offsets, char, cases, out, raw, figd)


def analyse_b(d, a_grid, offsets, char, cases, out, raw, figd):
    rows = []
    L_ = ["# Step 7, Part B: OCDM vs OTFS with a guard after every symbol",
          "",
          "Generated by `study_otfs.py --part B`. Open-loop characteristic: "
          f"â = 0, true a on {len(a_grid)} non-integer-shift values over "
          f"[{a_grid[0]:.4e}, {a_grid[-1]:.4e}], {d['n_char']} blocks each "
          "(the same data and noise for all SNRs); K_d = slope of a linear "
          "fit of the mean e over that range, linearity = max |residual| / "
          "(K_d·a_max), σ_e pooled over a. Pull-in from the noiseless "
          f"S-curve at a = {A_LOOP[0]:.4e} ({d['n_s']} blocks).", "",
          "## Open-loop characteristic", "",
          "| waveform | SNR | K_d | offset (in a) | linearity | σ_e | σ_e/K_d |",
          "|---|---|---|---|---|---|---|"]
    for w in WF_B:
        for s, snr in enumerate(SNRS_CHAR):
            m = char[w][s]
            L_.append(f"| {w} | {snr_s(snr)} | {m['K_d']:.1f} | "
                      f"{m['offset_a']:+.2e} | {m['lin']:.1%} | "
                      f"{m['sigma_e']:.2e} | {m['s_ek']:.2e} |")
            rows.append(dict(part="B", kind="char", waveform=w, snr_db=snr,
                             **m))
    L_ += ["", "Ratio OTFS-symbol / OCDM:", "",
           "| SNR | K_d | σ_e/K_d |", "|---|---|---|"]
    for s, snr in enumerate(SNRS_CHAR):
        o, t = char["OCDM"][s], char["OTFS-symbol"][s]
        L_.append(f"| {snr_s(snr)} | {t['K_d'] / o['K_d']:.3f} | "
                  f"{t['s_ek'] / o['s_ek']:.3f} |")
    L_ += ["", "## Pull-in (noiseless S-curve)", "",
           f"Predicted ±P_samp/(k N_seg) = ±{20 / N_SEG:.2e} (main-lobe "
           "nulls); measured: first sign change of the mean e on each side.",
           "", "| waveform | pull-in, a_hat < a | a_hat > a |", "|---|---|---|"]
    scurves = {}
    for w in WF_B:
        st = np.load(raw / f"B_scurve_{w}.npz")
        me = st["e"][:, 0, :].mean(0)
        scurves[w] = me
        lo, hi = pull_in(offsets, me)
        L_.append(f"| {w} | {lo:+.2e} | {hi:+.2e} |")
        rows.append(dict(part="B", kind="pull-in", waveform=w, lo=lo, hi=hi))

    L_ += ["", f"## Closed loop (first order, pole {POLE}, μ = 0.01 / K_d "
           f"at the WP per waveform; R = {d['R']} × L = {d['L']} after "
           f"{d['D']} blocks)", "",
           "Prediction: std = (σ_e/K_d)·√(g/(2−g)) with the measured "
           "σ_e/K_d and g = μ K_d = 0.01. RMSE includes the bias.", "",
           "| waveform | a | SNR | bias ± se | std | pred. std | std / pred | "
           "RMSE |", "|---|---|---|---|---|---|---|---|"]
    g = 1 - POLE
    loop_traj = {}
    for nm, c in cases.items():
        A = np.array([np.load(raw / f"{nm}__r{r}.npz")["a_hat"]
                      for r in range(c["R"])])     # [R, 1, snr, i]
        loop_traj[nm] = A
        for s, snr in enumerate(SNRS_LOOP):
            eps = A[:, 0, s, d["D"]:d["D"] + d["L"]] - c["a"]
            bias = float(eps.mean())
            se = (float(eps.mean(1).std(ddof=1) / math.sqrt(eps.shape[0]))
                  if eps.shape[0] > 1 else float("nan"))
            std = float(eps.std())
            pred = char[c["wf"]][SNRS_CHAR.index(snr)]["s_ek"] * math.sqrt(
                g / (2 - g))
            rmse = float(np.sqrt(np.mean(eps ** 2)))
            L_.append(f"| {c['wf']} | {c['a']:+.4e} | {snr_s(snr)} | "
                      f"{bias:+.2e} ± {se:.1e} | {std:.2e} | {pred:.2e} | "
                      f"{std / pred:.2f} | {rmse:.2e} |")
            rows.append(dict(part="B", kind="loop", waveform=c["wf"],
                             a=c["a"], snr_db=snr, bias=bias, bias_se=se,
                             std=std, std_pred=pred, rmse=rmse))
    L_.append("")
    (out / "otfs_B.md").write_text("\n".join(L_), encoding="utf-8")
    keys = sorted({k for r in rows for k in r})
    with open(out / "otfs_B.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("\n".join(L_))
    plot_b(a_grid, offsets, char, scurves, raw, d, figd)


def plot_b(a_grid, offsets, char, scurves, raw, d, figd):
    cols = {"OCDM": "#1f77b4", "OTFS-symbol": "#d62728"}
    with plt.rc_context({"font.size": 11}):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        x = [s if np.isfinite(s) else 30.0 for s in SNRS_CHAR]
        for w in WF_B:
            ax.semilogy(x, [m["s_ek"] for m in char[w]], "o-" if w == "OCDM"
                        else "s--", color=cols[w], label=w)
        ax.set_xticks(x)
        ax.set_xticklabels([("∞" if not np.isfinite(s) else f"{s:g}")
                            for s in SNRS_CHAR])
        ax.set_xlabel("σ_D²/P_n [dB] (WP = 4 dB)")
        ax.set_ylabel("σ_e / K_d")
        ax.set_title("Open-loop detector noise per waveform (Δ = {0}, k = 1)",
                     fontsize=10.5)
        ax.grid(True, which="both", color="0.9", lw=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figd / "fig_otfs_sigma_vs_snr.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for w in WF_B:
            E = np.array([np.load(raw / f"B_char_{w}_{i}.npz")["e"][:, :, 0]
                          for i in d["a_idx"]])
            axes[0].plot(a_grid, E[:, 0].mean(1), "o-" if w == "OCDM"
                         else "s--", color=cols[w], label=f"{w}, noiseless")
            axes[0].plot(a_grid, E[:, SNRS_CHAR.index(WP_DB)].mean(1),
                         ".:", color=cols[w], label=f"{w}, WP")
            axes[1].plot(offsets, scurves[w], "-" if w == "OCDM" else "--",
                         color=cols[w], label=w)
        axes[0].set_xlabel("true a (â = 0)")
        axes[0].set_ylabel("mean e")
        axes[0].set_title("open-loop characteristic", fontsize=10.5)
        axes[1].set_xlabel("â − a")
        axes[1].set_ylabel("mean e (noiseless)")
        axes[1].set_title("S-curve, a = 5.1542e-4", fontsize=10.5)
        for ax in axes:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.axhline(0, color="0.6", lw=0.6)
            ax.grid(True, color="0.9", lw=0.5)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figd / "fig_otfs_char_B.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------
KS_C = (1, 2, 3, 4, 5)


def fourier_pred(P, n_on, k):
    """|c_k| of a 0/1 pattern with n_on ones per period P (k >= 1)."""
    d = n_on / P
    return abs(math.sin(math.pi * k * d)) / (math.pi * k)


def load_char(raw, prefix, w, a_idx):
    return np.array([np.load(raw / f"{prefix}_char_{w}_{i}.npz")["e"]
                     for i in a_idx])                # [a, snr, block, k]


def part_c(args, out, raw, figd, t0):
    d = design_b(args.quick)
    a_grid = char_grid()[list(d["a_idx"])]
    w = "OTFS-frame"
    P = OTFSWaveform(guard="frame").P_samp
    offs = {k: np.linspace(-1.5, 1.5, d["n_off"]) * P / (k * N_SEG)
            for k in KS_C}
    if not args.replot:
        ct = [(raw / f"C_char_{w}_{i}.npz", w, float(a), d["n_char"], 1000,
               KS_C) for i, a in zip(d["a_idx"], a_grid)]
        st = [(raw / f"C_scurve_k{k}.npz", w, A_LOOP[0], offs[k], d["n_s"],
               1001, (k,)) for k in KS_C]
        if not run_tasks(_char_task, ct, args.procs, "characteristic", t0,
                         args.budget):
            return
        if not run_tasks(_scurve_task, st, args.procs, "S-curve", t0,
                         args.budget):
            return
    E = load_char(raw, "C", w, d["a_idx"])
    char = {k: char_metrics(a_grid, E[..., j]) for j, k in enumerate(KS_C)}
    # loops for the harmonics the prediction says exist (c_5 = 0); mu from
    # the noiseless K_d (the WP slope is too noisy for OTFS-frame)
    ks_loop = [k for k in KS_C if fourier_pred(P, 256, k) > 1e-6]
    mus = [(1 - POLE) / char[k][0]["K_d"] for k in ks_loop]
    cases = {f"C_loop_{w}_a{a:+.4e}": dict(wf=w, a=a, mus=mus, ks=ks_loop,
                                           R=d["R"], n_blocks=d["D"] + d["L"])
             for a in A_LOOP}
    if not args.replot and not run_loops(cases, raw, args.procs, t0,
                                         args.budget):
        return
    # Part B metrics for the comparison (same blocks / design)
    charB = {}
    for wb in WF_B:
        if (raw / f"B_char_{wb}_{d['a_idx'][0]}.npz").exists():
            charB[wb] = char_metrics(a_grid, load_char(raw, "B", wb,
                                                       d["a_idx"])[..., 0])
    analyse_c(d, a_grid, offs, char, charB, ks_loop, cases, P, out, raw,
              figd)


def analyse_c(d, a_grid, offs, char, charB, ks_loop, cases, P, out, raw,
              figd):
    n_on = 256
    rows = []
    fom = {k: k * fourier_pred(P, n_on, k) for k in KS_C}
    L_ = ["# Step 7, Part C: OTFS with one guard per frame, harmonic k", "",
          "Generated by `study_otfs.py --part C`. OTFS-frame: M = N = 16, "
          f"one zero guard of N·N_gi = 64 samples per frame, P_samp = {P} "
          f"({n_on} on, {P - n_on} off). Same design as Part B "
          f"(characteristic over {len(a_grid)} values of a × {d['n_char']} "
          "blocks, noiseless S-curve per k, closed loop pole 0.99 with "
          "μ = 0.01 / K_d(k, noiseless)); all k on the same blocks.", "",
          "## Prediction from the Fourier coefficients of the on/off pattern",
          "",
          "|c_k| = |sin(π k d)|/(π k), d = 256/320 = 0.8. With Δ = {0} the "
          "cost at the harmonic is ∝ |c_k|²; δ is fixed in α, so K_d (in a) "
          "∝ k/P_samp and the pull-in ±P_samp/(k N_seg). If the detector "
          "noise is set by the CAF estimation noise, σ_e/K_d ∝ P_samp/(k "
          "|c_k|) where the signal dominates (high SNR) and ∝ P_samp/(k "
          "|c_k|²) where the noise dominates.", "",
          "| k | \\|c_k\\| | k\\|c_k\\| (high SNR) | k\\|c_k\\|² (low SNR) | "
          "pull-in ±P/(k N_seg) |", "|---|---|---|---|---|"]
    for k in KS_C:
        c = fourier_pred(P, n_on, k)
        L_.append(f"| {k} | {c:.4f} | {k * c:.4f} | {k * c * c:.5f} | "
                  f"{P / (k * N_SEG):.4f} |")
    c1 = fourier_pred(20, 16, 1)
    L_ += ["", f"OCDM / OTFS-symbol (P = 20, k = 1): |c_1| = {c1:.4f}; the "
           f"high-SNR figure of merit k|c_k|/P_samp is {c1 / 20:.2e} vs "
           f"{max(fom.values()) / P:.2e} for OTFS-frame at its best k, i.e. "
           f"σ_e/K_d about {(c1 / 20) / (max(fom.values()) / P):.1f}× higher "
           "for OTFS-frame (same reasoning; low SNR: "
           f"{(c1 * c1 / 20) / max(k * fourier_pred(P, n_on, k) ** 2 / P for k in KS_C):.1f}×).",
           "", "## Measured open-loop characteristic", "",
           "| k | SNR | K_d | offset (in a) | linearity | σ_e/K_d |",
           "|---|---|---|---|---|---|"]
    for k in KS_C:
        for s, snr in enumerate(SNRS_CHAR):
            m = char[k][s]
            L_.append(f"| {k} | {snr_s(snr)} | {m['K_d']:.2f} | "
                      f"{m['offset_a']:+.2e} | {m['lin']:.1%} | "
                      f"{m['s_ek']:.2e} |")
            rows.append(dict(part="C", kind="char", k=k, snr_db=snr, **m))
    L_ += ["", "σ_e/K_d of OTFS-frame at each k relative to OCDM and "
           "OTFS-symbol (Part B):", "",
           "| SNR | OCDM | OTFS-symbol | " + " | ".join(
               f"frame k={k}" for k in KS_C) + " |",
           "|---|---|---|" + "---|" * len(KS_C)]
    for s, snr in enumerate(SNRS_CHAR):
        o = charB.get("OCDM", [None] * 5)[s]
        t = charB.get("OTFS-symbol", [None] * 5)[s]
        L_.append(f"| {snr_s(snr)} | {o['s_ek']:.2e} | {t['s_ek']:.2e} | "
                  + " | ".join(f"{char[k][s]['s_ek']:.2e} "
                               f"({char[k][s]['s_ek'] / o['s_ek']:.1f}×)"
                               for k in KS_C) + " |"
                  if o and t else f"| {snr_s(snr)} | – | – | – |")
    L_ += ["", "## Pull-in (noiseless S-curve per k)", "",
           "| k | predicted ±P/(k N_seg) | measured, a_hat < a | a_hat > a |",
           "|---|---|---|---|"]
    sc = {}
    for k in KS_C:
        st = np.load(raw / f"C_scurve_k{k}.npz")
        me = st["e"][:, 0, :].mean(0)
        sc[k] = me
        lo, hi = pull_in(offs[k], me)
        L_.append(f"| {k} | {P / (k * N_SEG):.4f} | {lo:+.4f} | {hi:+.4f} |")
        rows.append(dict(part="C", kind="pull-in", k=k, lo=lo, hi=hi))
    L_ += ["", f"## Closed loop (k ∈ {ks_loop}; k = 5 has no harmonic; "
           f"R = {d['R']} × L = {d['L']} after {d['D']} blocks)", "",
           "μ = 0.01 / K_d(k, noiseless); prediction std = (σ_e/K_d)·"
           "√(g/(2−g)) with g = μ K_d(k, SNR) (the loop gain at that SNR).",
           "", "| k | a | SNR | g | bias ± se | std | pred. std | std / pred "
           "| RMSE |", "|---|---|---|---|---|---|---|---|---|"]
    for nm, c in cases.items():
        A = np.array([np.load(raw / f"{nm}__r{r}.npz")["a_hat"]
                      for r in range(c["R"])])
        for m, k in enumerate(ks_loop):
            for s, snr in enumerate(SNRS_LOOP):
                eps = A[:, m, s, d["D"]:d["D"] + d["L"]] - c["a"]
                bias = float(eps.mean())
                se = (float(eps.mean(1).std(ddof=1) / math.sqrt(eps.shape[0]))
                      if eps.shape[0] > 1 else float("nan"))
                std = float(eps.std())
                cs = char[k][SNRS_CHAR.index(snr)]
                g = c["mus"][m] * cs["K_d"]
                pred = (cs["s_ek"] * math.sqrt(g / (2 - g)) if 0 < g < 2
                        else float("nan"))
                rmse = float(np.sqrt(np.mean(eps ** 2)))
                L_.append(f"| {k} | {c['a']:+.4e} | {snr_s(snr)} | {g:.4f} | "
                          f"{bias:+.2e} ± {se:.1e} | {std:.2e} | {pred:.2e} | "
                          f"{std / pred:.2f} | {rmse:.2e} |")
                rows.append(dict(part="C", kind="loop", k=k, a=c["a"],
                                 snr_db=snr, g=g, bias=bias, bias_se=se,
                                 std=std, std_pred=pred, rmse=rmse))
    L_.append("")
    (out / "otfs_C.md").write_text("\n".join(L_), encoding="utf-8")
    keys = sorted({k for r in rows for k in r})
    with open(out / "otfs_C.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("\n".join(L_))
    plot_c(char, charB, sc, offs, P, figd)


def plot_c(char, charB, sc, offs, P, figd):
    with plt.rc_context({"font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        ax = axes[0]
        for s, snr, mk in ((0, "noiseless", "o-"),
                           (SNRS_CHAR.index(WP_DB), "WP 4 dB", "s--"),
                           (SNRS_CHAR.index(0.0), "0 dB", "^:")):
            y = [char[k][s]["s_ek"] if char[k][s]["K_d"] > 0 else np.nan
                 for k in KS_C]
            ax.semilogy(KS_C, y, mk, label=f"OTFS-frame, {snr}")
            if "OCDM" in charB:
                ax.axhline(charB["OCDM"][s]["s_ek"], color=ax.lines[-1]
                           .get_color(), lw=0.8, ls="-.", alpha=0.7)
        ax.set_xticks(KS_C)
        ax.set_xlabel("harmonic k")
        ax.set_ylabel("σ_e / K_d")
        ax.set_title("OTFS-frame vs harmonic (dash-dot: OCDM, k = 1)",
                     fontsize=10.5)
        ax.grid(True, which="both", color="0.9", lw=0.5)
        ax.legend(fontsize=8)
        ax = axes[1]
        for k in KS_C:
            ax.plot(offs[k] * k * N_SEG / P, sc[k], label=f"k = {k}")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel("(â − a) / (P_samp / (k N_seg))")
        ax.set_ylabel("mean e (noiseless)")
        ax.set_title("S-curves, offset scaled by the predicted pull-in",
                     fontsize=10.5)
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figd / "fig_otfs_frame_vs_k.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Part D: multipath
# ---------------------------------------------------------------------------
from ocdm_doppler import cbsfs                                  # noqa: E402
from ocdm_doppler.cbsfs_dc import cost_function_dc              # noqa: E402
from ocdm_doppler.channel import (exponential_pdp, MultipathChannel,  # noqa
                                  Path as CPath)
from ocdm_doppler.channel_norm import normalised_params         # noqa: E402
from ocdm_doppler.loop_wf2 import run_loop_det, open_loop_det   # noqa: E402

DETS = {"dc": cost_function_dc, "cur": cbsfs.cost_function}
DET_NAME = {"dc": "DC removed", "cur": "current"}
# Yogev's Part D settings
WAVE_D = {
    "OCDM": dict(N_seg=2048, k=1, dets=("dc", "cur"), spreads=(0, 2, 4, 8),
                 d_off=2e-4),
    "OTFS-frame": dict(N_seg=1920, k=2, dets=("dc",),
                       spreads=(0, 2, 4, 8, 32, 64), d_off=5e-4),
}
A_D = 5.1542e-4
SNRS_D = (float("inf"), WP_DB)
MS_A1 = 5.1542e-4
MS_DA = 1e-4


def design_d(quick):
    if quick:
        return dict(n_ol=4, D=10, L=20, R_fix=1, R_ray=1, n_ms=20,
                    ms_snap=(2, 18))
    # option B (Yogev): fixed gains R = 1, Rayleigh R = 2, 200 + 600 blocks
    # (the first design, R = 2 + 3 and 300 + 1000 blocks, was ~85 min)
    return dict(n_ol=100, D=200, L=600, R_fix=1, R_ray=2, n_ms=500,
                ms_snap=(10, 490))


def d_channel(spread, gain, r, sidx):
    """The channel of one Part D realisation (Rayleigh: one draw per r)."""
    if gain == "fixed":
        return exponential_pdp(spread, 4)
    rng = np.random.default_rng([2024, sidx, r])
    return exponential_pdp(spread, 4, rayleigh=True, rng=rng)


def d_cases(d):
    """(wf, spread index, spread, gain, r, seed) for every realisation."""
    out = []
    for w, c in WAVE_D.items():
        for si, D_max in enumerate(c["spreads"]):
            gains = ("fixed",) if D_max == 0 else ("fixed", "rayleigh")
            for g in gains:
                R = d["R_fix"] if g == "fixed" else d["R_ray"]
                for r in range(R):
                    out.append((w, si, D_max, g, r, 1000 + r))
    return out


def _d_name(w, si, g, r):
    return f"{w}_s{si}_{g}_r{r}"


def _d_setup(w, si, D_max, g, r, seed):
    c = WAVE_D[w]
    p = base().with_(a_true=A_D)
    ch = d_channel(D_max, g, r, si)
    ws = make_wf(w).signal(p, seed)
    pn, G = normalised_params(ws, ch, p)
    return c, ch, pn, G


def _d_ol_task(args):
    f, w, si, D_max, g, r, seed, n = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    c, ch, pn, G = _d_setup(w, si, D_max, g, r, seed)
    e = open_loop_det(pn, make_wf(w), ch, ConstantDoppler(A_D),
                      A_D + np.array([-c["d_off"], 0.0, c["d_off"]]), n,
                      snrs=SNRS_D, k=c["k"],
                      dets=[DETS[x] for x in c["dets"]], N_seg=c["N_seg"],
                      seed=seed)
    np.savez(f, e=e, G=G, rms=ch.rms_delay_spread(),
             mx=ch.max_delay_spread())
    return f, time.time() - t0


def _d_loop_task(args):
    f, w, si, D_max, g, r, seed, n, mus = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    c, ch, pn, G = _d_setup(w, si, D_max, g, r, seed)
    filt = [lf.make_filter(mu, 0.0, 1.0, a_hat0=A_D) for mu in mus]
    res = run_loop_det(pn, make_wf(w), ch, ConstantDoppler(A_D), filt, n,
                       snrs=SNRS_D, N_seg=c["N_seg"], ks=c["k"],
                       dets=[DETS[x] for x in c["dets"]], seed=seed)
    np.savez(f, a_hat=res.a_hat, G=G)
    return f, time.time() - t0


def ms_channel(p2):
    """Two paths, delays 2 and 4 samples, powers (1 - p2, p2), Doppler
    a1 = MS_A1 and a2 = MS_A1 + MS_DA (per path)."""
    return MultipathChannel([
        CPath(np.sqrt(1 - p2), 2.0, ConstantDoppler(MS_A1)),
        CPath(np.sqrt(p2), 4.0, ConstantDoppler(MS_A1 + MS_DA))],
        label=f"multiscale p2={p2:g}")


def _d_ms_task(args):
    f, p2, n, mus, snaps = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    p = base().with_(a_true=MS_A1)
    ch = ms_channel(p2)
    wf = make_wf("OCDM")
    filt = [lf.make_filter(mu, 0.0, 1.0, a_hat0=MS_A1) for mu in mus]
    dets = [DETS[x] for x in WAVE_D["OCDM"]["dets"]]
    res = run_loop_det(p, wf, ch, ConstantDoppler(MS_A1), filt, n,
                       snrs=SNRS_D, N_seg=2048, ks=1, dets=dets, seed=1000)
    # CAF around the two expected peaks at two time points (noiseless)
    from ocdm_doppler.channel import rx_block
    from ocdm_doppler.wf_params import detector_params
    ws = wf.signal(p, 1000)
    dp = detector_params(p, ws.P_samp)
    a_grid = np.linspace(MS_A1 - 3 * MS_DA, MS_A1 + 4 * MS_DA, 141)
    caf = []
    for b in snaps:
        y = rx_block(ws, ch, p, N_C, ConstantDoppler(MS_A1),
                     start_sample=b * N_C)
        caf.append([fn(y, (1 + a_grid) / ws.P_samp, dp, N_seg=2048)
                    for fn in dets])
    np.savez(f, a_hat=res.a_hat, caf=np.array(caf), a_grid=a_grid,
             snaps=np.array(snaps))
    return f, time.time() - t0


def d_identity_check():
    """run_loop_det / open_loop_det with the current detector == the
    loop_wf versions (bit for bit), OCDM single path and OTFS multipath."""
    from ocdm_doppler.loop_wf import run_loop_wf, open_loop_e
    p = base().with_(a_true=A_D)
    worst = 0.0
    for w, ch in (("OCDM", single_path(p)),
                  ("OTFS-frame", exponential_pdp(4, 4))):
        filt = [lf.make_filter(5.7e-5, 0.0, 1.0, a_hat0=A_D)]
        r1 = run_loop_det(p, make_wf(w), ch, ConstantDoppler(A_D), filt, 8,
                          snrs=SNRS_D, N_seg=1920, ks=2,
                          dets=cbsfs.cost_function, seed=1000)
        r2 = run_loop_wf(p, make_wf(w), ch, ConstantDoppler(A_D), filt, 8,
                         snrs=SNRS_D, N_seg=1920, ks=2, seed=1000)
        worst = max(worst, float(np.max(np.abs(r1.a_hat - r2.a_hat))))
        e1 = open_loop_det(p, make_wf(w), ch, ConstantDoppler(A_D),
                           A_D + np.array([-1e-4, 0, 1e-4]), 3, snrs=SNRS_D,
                           k=2, dets=(cbsfs.cost_function,), N_seg=1920,
                           seed=1000)
        e2 = open_loop_e(p, make_wf(w), ch, ConstantDoppler(A_D),
                         A_D + np.array([-1e-4, 0, 1e-4]), 3, snrs=SNRS_D,
                         ks=(2,), N_seg=1920, seed=1000)
        worst = max(worst, float(np.max(np.abs(e1[0] - e2[:, :, 0, :]))))
    return worst


def part_d(args, out, raw, figd, t0):
    d = design_d(args.quick)
    cases = d_cases(d)
    ident = d_identity_check() if not args.replot else float("nan")
    print(f"  runner identity check: max |diff| = {ident:.1e}", flush=True)
    if not args.replot:
        ol = [(str(raw / f"D_ol_{_d_name(w, si, g, r)}.npz"), w, si, D_max,
               g, r, seed, d["n_ol"]) for w, si, D_max, g, r, seed in cases]
        if not run_tasks(_d_ol_task, ol, args.procs, "open loop", t0,
                         args.budget):
            return
    # K_d0 per waveform and detector: single path, noiseless
    Kd0 = {}
    for w, c in WAVE_D.items():
        E = np.array([np.load(raw / f"D_ol_{_d_name(w, 0, 'fixed', r)}.npz")
                      ["e"] for r in range(d["R_fix"])])  # [R, det, s, i, o]
        for di, x in enumerate(c["dets"]):
            Kd0[w, x] = float((E[:, di, 0, :, 0] - E[:, di, 0, :, 2]).mean()
                              / (2 * c["d_off"]))
    if not args.replot:
        lt = [(str(raw / f"D_loop_{_d_name(w, si, g, r)}.npz"), w, si, D_max,
               g, r, seed, d["D"] + d["L"],
               [(1 - POLE) / Kd0[w, x] for x in WAVE_D[w]["dets"]])
              for w, si, D_max, g, r, seed in cases]
        mus_ms = [(1 - POLE) / Kd0["OCDM", x] for x in WAVE_D["OCDM"]["dets"]]
        ms = [(str(raw / f"D_ms_p{p2:g}.npz"), p2, d["n_ms"], mus_ms,
               d["ms_snap"]) for p2 in (0.5, 1 / 3)]
        if not run_tasks(_d_loop_task, lt + [], args.procs, "closed loop",
                         t0, args.budget):
            return
        if not run_tasks(_d_ms_task, ms, args.procs, "multiscale", t0,
                         args.budget):
            return
    analyse_d(d, cases, Kd0, ident, out, raw, figd)


def analyse_d(d, cases, Kd0, ident, out, raw, figd):
    rows = []
    L_ = ["# Step 7, Part D: multipath", "",
          "Generated by `study_otfs.py --part D`. Exponential power-delay "
          "profile, 4 paths uniformly over [0, D_max] samples (fractional "
          "delays), last path −10 dB, common Doppler a = 5.1542e-4; fixed "
          f"gains ({d['R_fix']} data seeds) and Rayleigh gains ({d['R_ray']} "
          "channel draws, one per realisation). Every channel is scaled to "
          "the mean received signal power of the single-path reference "
          "(exact expression incl. the cross terms; G = P_ref / P_ch is the "
          "power factor applied). OCDM: N_seg = 2048, k = 1, DC-removed "
          "detector (main) and the current detector on the same blocks; "
          "OTFS-frame: N_seg = 1920, k = 2, DC removed. Open loop: "
          f"{d['n_ol']} blocks at â = a, a ± d (d = 2e-4 OCDM, 5e-4 "
          "OTFS-frame); K_d from the secant. Closed loop: first order, pole "
          "0.99, μ = 0.01 / K_d0 (single path, noiseless, per detector: the "
          "receiver does not know the channel), "
          f"{d['L']} blocks after {d['D']}; g_eff = μ K_d(channel, WP).",
          "", "**Normalisation factor G.** With h_p = A_p0 c_p "
          "e^{−j2π f_Tx τ_p}/2 (Σ|c_p|² = 1), the exact mean received signal "
          "power over one period is P_ch = ⟨Σ_pq h_p h_q* R_X(t − τ_p, "
          "t − τ_q)⟩_t, R_X(t1, t2) = E[X(t1) X*(t2)]; P_ref is the same for "
          "one path with c = 1. G = P_ref / P_ch, and the channel's amplitude "
          "A_p0 is multiplied by √G (the received power by G), so every "
          "channel has the single-path received power and the same effective "
          "SNR. Without cross terms (paths uncorrelated: integer delay "
          "differences, or random phases on average) P_ch = P_ref and G = 1; "
          "G > 1 means the paths' cross terms are negative on average (partial "
          "cancellation: the received power fell and had to be raised), "
          "G < 1 that they add constructively.", "",
          "Captured: max |â − a| over the last 200 loop blocks < 1e-3 (far "
          "from the false-lock points), per realisation.", "",
          f"Runner check: run_loop_det / open_loop_det with the current "
          f"detector vs loop_wf: max |diff| = {ident:.1e}.", "",
          "K_d0 (single path, noiseless): " + ", ".join(
              f"{w} {DET_NAME[x]} {v:.1f}" for (w, x), v in Kd0.items()), ""]
    groups = {}
    for w, si, D_max, g, r, seed in cases:
        groups.setdefault((w, si, D_max, g), []).append(r)
    bsz = 150
    for w, c in WAVE_D.items():
        for di, x in enumerate(c["dets"]):
            L_ += [f"## {w}, {DET_NAME[x]} detector", "",
                   "| D_max | gains | RMS spread | G (per realisation) | SNR "
                   "| K_d | K_d / K_d0 | σ_e/K_d | σ_e/K_d rel. to D_max = 0 "
                   "| g_eff | bias ± se | std | RMSE | captured |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
                   "---|"]
            sek0 = {}
            for (w2, si, D_max, g), rs in groups.items():
                if w2 != w:
                    continue
                ols = [np.load(raw / f"D_ol_{_d_name(w, si, g, r)}.npz")
                       for r in rs]
                lps = [np.load(raw / f"D_loop_{_d_name(w, si, g, r)}.npz")
                       for r in rs]
                Gs = [float(o["G"]) for o in ols]
                rms = float(np.mean([o["rms"] for o in ols]))
                for s, snr in enumerate(SNRS_D):
                    E = np.array([o["e"][di, s] for o in ols])  # [R, i, off]
                    K = float((E[..., 0] - E[..., 2]).mean()
                              / (2 * c["d_off"]))
                    sek = float((E[..., 1] / K).std(ddof=1))
                    eps = np.array([lp["a_hat"][di, s, d["D"]:d["D"] + d["L"]]
                                    for lp in lps]) - A_D
                    nb = eps.shape[1] // bsz
                    bm = eps[:, :nb * bsz].reshape(len(rs), nb, bsz).mean(-1)
                    bias = float(eps.mean())
                    se = float(bm.std(ddof=1) / math.sqrt(bm.size))
                    g_eff = (1 - POLE) / Kd0[w, x] * K
                    if D_max == 0:
                        sek0[s] = sek
                    rel = sek / sek0[s] if s in sek0 else float("nan")
                    capt = int(np.sum(np.abs(eps[:, -200:]).max(1) < 1e-3))
                    L_.append(
                        f"| {D_max:g} | {g} | {rms:.2f} | "
                        f"{', '.join(f'{v:.3f}' for v in Gs)} | "
                        f"{snr_s(snr)} | {K:.1f} | {K / Kd0[w, x]:.3f} | "
                        f"{sek:.2e} | {rel:.2f} | "
                        f"{g_eff:.4f} | {bias:+.2e} ± {se:.1e} | "
                        f"{eps.std():.2e} | "
                        f"{float(np.sqrt(np.mean(eps ** 2))):.2e} | "
                        f"{capt}/{len(rs)} |")
                    rows.append(dict(part="D", waveform=w, detector=x,
                                     D_max=D_max, gains=g, rms=rms,
                                     G=";".join(f"{v:.4f}" for v in Gs),
                                     snr_db=snr, K_d=K, K_rel=K / Kd0[w, x],
                                     s_ek=sek, s_ek_rel=rel, g_eff=g_eff,
                                     bias=bias, bias_se=se,
                                     std=float(eps.std()),
                                     rmse=float(np.sqrt(np.mean(eps ** 2))),
                                     captured=f"{capt}/{len(rs)}"))
            L_.append("")
    # multiscale
    L_ += ["## Multiscale (OCDM, two paths with their own Doppler factors)",
           "",
           f"Paths at delays 2 and 4 samples, a1 = {MS_A1:.4e}, a2 = a1 + "
           f"{MS_DA:g}; equal powers and 2:1 (p2 = 1/3); â0 = a1; "
           f"{d['n_ms']} blocks ({d['n_ms'] * N_C / 4e6:.1f} s). Because "
           "each path has its own Doppler factor the paths drift apart "
           f"(relative delay changes by {MS_DA:g}·t, ≈ "
           f"{MS_DA * d['n_ms'] * N_C:.0f} samples over the run) and their "
           "cross terms rotate at f_Tx·Δa: this case shows which component "
           "the loop locks to, not steady-state accuracy. Position in units "
           "of a2 − a1 (0 = path 1, 1 = path 2; power-weighted mean = p2).",
           "", "| p2 | detector | SNR | position, blocks 100–200 | "
           "last 100 blocks | power-weighted mean |", "|---|---|---|---|---|"
           "---|"]
    ms_data = {}
    for p2 in (0.5, 1 / 3):
        z = np.load(raw / f"D_ms_p{p2:g}.npz")
        ms_data[p2] = z
        for di, x in enumerate(WAVE_D["OCDM"]["dets"]):
            for s, snr in enumerate(SNRS_D):
                ah = (z["a_hat"][di, s] - MS_A1) / MS_DA
                n = ah.size - 1
                mid = ah[min(100, n // 3):min(200, 2 * n // 3)].mean()
                L_.append(f"| {p2:.3g} | {DET_NAME[x]} | {snr_s(snr)} | "
                          f"{mid:+.3f} | {ah[-100:].mean():+.3f} | "
                          f"{p2:.3f} |")
    L_.append("")
    (out / "otfs_D.md").write_text("\n".join(L_), encoding="utf-8")
    keys = sorted({k for r in rows for k in r})
    with open(out / "otfs_D.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("\n".join(L_))
    plot_d(rows, ms_data, d, figd)


def plot_d(rows, ms_data, d, figd):
    cols = {"OCDM": "#1f77b4", "OTFS-frame": "#d62728"}
    with plt.rc_context({"font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        for (w, x), ls in ((("OCDM", "dc"), "-"), (("OCDM", "cur"), ":"),
                           (("OTFS-frame", "dc"), "-")):
            for g, mk in (("fixed", "o"), ("rayleigh", "s")):
                sel = [r for r in rows if r["waveform"] == w and
                       r["detector"] == x and r["snr_db"] == WP_DB and
                       (r["gains"] == g or r["D_max"] == 0)]
                sel.sort(key=lambda r: r["D_max"])
                if not sel:
                    continue
                xs = [max(r["D_max"], 0.5) for r in sel]
                lab = f"{w}, {DET_NAME[x]}, {g}"
                axes[0].plot(xs, [r["K_rel"] for r in sel], ls + mk,
                             color=cols[w], label=lab, mfc="none"
                             if g == "rayleigh" else None)
                axes[1].plot(xs, [r["rmse"] for r in sel], ls + mk,
                             color=cols[w], label=lab, mfc="none"
                             if g == "rayleigh" else None)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("max delay spread D_max [samples] (0 plotted at "
                          "0.5)")
            ax.grid(True, which="both", color="0.9", lw=0.5)
            ax.axvline(4, color="0.6", ls="--", lw=0.8)
            ax.axvline(64, color="0.6", ls="--", lw=0.8)
        axes[0].set_ylabel("K_d / K_d0 (WP)")
        axes[1].set_ylabel("closed-loop RMSE (WP)")
        axes[1].set_yscale("log")
        axes[0].set_title("detector gain vs delay spread (dashed: guard "
                          "lengths 4, 64)", fontsize=10)
        axes[1].set_title("closed-loop RMSE at the WP", fontsize=10)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figd / "fig_otfs_multipath.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for p2, ls in ((0.5, "-"), (1 / 3, "--")):
            z = ms_data[p2]
            for di, x in enumerate(WAVE_D["OCDM"]["dets"]):
                ah = (z["a_hat"][di, 1] - MS_A1) / MS_DA
                axes[0].plot(np.arange(ah.size) * N_C / 4e6, ah, ls,
                             lw=1.0 if x == "dc" else 0.7,
                             label=f"p2 = {p2:.2g}, {DET_NAME[x]}, WP")
            axes[0].axhline(p2, color="0.5", ls=ls, lw=0.6)
        axes[0].axhline(0, color="k", lw=0.6)
        axes[0].axhline(1, color="k", lw=0.6)
        axes[0].set_xlabel("time [s]")
        axes[0].set_ylabel("(â − a1) / (a2 − a1)")
        axes[0].set_title("which path does the loop follow? (grey: power-"
                          "weighted mean)", fontsize=10)
        axes[0].legend(fontsize=7)
        z = ms_data[0.5]
        for j, ax in enumerate(axes[1:]):
            for di, x in enumerate(WAVE_D["OCDM"]["dets"]):
                cf = z["caf"][j, di]
                ax.plot((z["a_grid"] - MS_A1) / MS_DA, cf / cf.max(),
                        lw=1.2 if x == "dc" else 0.8,
                        label=DET_NAME[x])
            ax.axvline(0, color="k", ls=":", lw=0.8)
            ax.axvline(1, color="k", ls=":", lw=0.8)
            ax.set_xlabel("(a − a1) / (a2 − a1)  [α = (1 + a)/20]")
            ax.set_ylabel("C(α) / max")
            ax.set_title(f"cost around the two peaks, block "
                         f"{int(z['snaps'][j])} (noiseless, p2 = 0.5)",
                         fontsize=10)
            ax.legend(fontsize=8)
        for ax in axes:
            ax.grid(True, color="0.9", lw=0.5)
        fig.tight_layout()
        fig.savefig(figd / "fig_otfs_multiscale.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--part", choices=("B", "C", "D"), required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--budget", type=float, default=540.0)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seg", type=int, default=300)
    args = ap.parse_args()
    global SEG
    SEG = args.seg
    t0 = time.time()
    out = ROOT / "results" / ("quick" if args.quick else "")
    raw = out / "otfs_raw"
    figd = out if args.quick else ROOT / "figures"
    for q in (out, raw, figd):
        q.mkdir(parents=True, exist_ok=True)
    if not (args.resume or args.replot):
        for f in raw.glob(f"{args.part}_*.npz"):
            f.unlink()
    if args.part == "B":
        part_b(args, out, raw, figd, t0)
    elif args.part == "C":
        part_c(args, out, raw, figd, t0)
    elif args.part == "D":
        part_d(args, out, raw, figd, t0)
    else:
        raise SystemExit(f"Part {args.part}: not implemented yet")
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
