"""
Accuracy vs computational cost of the Delta = {0} detector (Yogev, 22.11).

Cost per second = RM/sample x f_s / M (M = duty cycle: one block in M is
processed), so shortening N_c alone does not reduce it; the levers are
N_seg and M.

(a) Open loop, OCDM single path, DC-removed detector (cbsfs_dc), noiseless
    and the WP (4 dB): K_d (secant at a_hat = a +/- d, d = 0.02 P/N_seg) and
    sigma_e/K_d (spread of e(a_hat = a)/K_d) for N_c in {6000, 12000, 24000,
    48000} x N_seg in {512, 1024, 2048, 4096} with N_c >= 4 N_seg; ~9.6e6
    samples per pair. a = 5.1542e-4 (a N_c = 3.09, 6.19, 12.37, 24.74: never
    an integer). Capture half-width: predicted P_samp/N_seg and measured
    (first sign change of the noiseless S-curve, N_c = 24000). Exact
    RM/sample: 22 N_c - 16 N_seg + 12 (+1 for the DC removal) per update
    (the count of count_ops_lag0.py; the closed form is checked against its
    counter for two pairs).
(b) Prediction at a FIXED physical noise bandwidth: B = B_L,blk / T_blk with
    the WP design (N_c = 24000, g = 0.01); for an update interval
    T_u = M N_c / f_s the loop uses B_L,blk = B T_u (g = 2B'/(1 + B')),
    and RMSE = (sigma_e/K_d) sqrt(B_L,blk) (white detector noise, linear
    loop, bias not included) -> figures/fig_cost_accuracy.png.
(c) Short closed-loop checks of 4 predicted points (R = 3, 30 tau after 3 tau,
    noiseless and WP in lockstep; duty cycle by processing blocks 0, M,
    2M, ...).

Output: results/cost_accuracy.md/.csv, raw results/cost_raw/.
Usage: .venv\\Scripts\\python.exe study_cost_accuracy.py [--quick]
       [--resume] [--procs 3] [--budget 540]
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocdm_doppler import preset_noisy_22jan
from ocdm_doppler import loop_filters as lf
from ocdm_doppler.cbsfs_dc import cost_function_dc
from ocdm_doppler.channel import single_path
from ocdm_doppler.doppler_profile import ConstantDoppler
from ocdm_doppler.loop_wf2 import open_loop_det, run_loop_det
from ocdm_doppler.waveforms import OCDMWaveform
from study_otfs import run_tasks

ROOT = Path(__file__).parent
FS = 4e6
P = 20
A = 5.1542e-4
SNRS = (float("inf"), 4.0)
NCS = (6000, 12000, 24000, 48000)
NSEGS = (512, 1024, 2048, 4096)
G0, NC0 = 0.01, 24000                      # WP design that fixes B
CHECKS = ((24000, 2048, 1), (6000, 512, 1), (48000, 4096, 1),
          (24000, 2048, 4))


def pairs():
    return [(nc, ns) for nc in NCS for ns in NSEGS if nc >= 4 * ns]


def rm_update(nc, ns, dc=True):
    return 22 * nc - 16 * ns + 12 + (1 if dc else 0)


def base():
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,),
                                                  a_true=A)


def d_off(ns):
    return 0.02 * P / ns


# ---------------------------------------------------------------------------
def _ol_task(args):
    f, nc, ns, start, n = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    e = open_loop_det(base(), OCDMWaveform(), single_path(base()),
                      ConstantDoppler(A), A + np.array([-d_off(ns), 0.0,
                                                        d_off(ns)]), n,
                      snrs=SNRS, k=1, dets=(cost_function_dc,), N_c=nc,
                      N_seg=ns, seed=1000, start_block=start)
    np.savez(f, e=e[0])                            # [snr, i, off]
    return f, time.time() - t0


def _scurve_task(args):
    f, ns, n = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    offs = np.linspace(-1.5, 1.5, 121) * P / ns
    e = open_loop_det(base(), OCDMWaveform(), single_path(base()),
                      ConstantDoppler(A), A + offs, n,
                      snrs=(float("inf"),), k=1, dets=(cost_function_dc,),
                      N_c=24000, N_seg=ns, seed=1001)
    np.savez(f, e=e[0, 0].mean(0), offs=offs)
    return f, time.time() - t0


def _loop_task(args):
    f, nc, ns, M, mu, n, seed = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    filt = [lf.make_filter(mu, 0.0, 1.0, a_hat0=A)]
    if M == 1:
        r = run_loop_det(base(), OCDMWaveform(), single_path(base()),
                         ConstantDoppler(A), filt, n, snrs=SNRS, N_c=nc,
                         N_seg=ns, dets=cost_function_dc, seed=seed)
        a_hat = r.a_hat[0]
    else:
        a_hat = np.empty((len(SNRS), n + 1))
        a_hat[:, 0] = A
        states = None
        for i in range(n):
            r = run_loop_det(base(), OCDMWaveform(), single_path(base()),
                             ConstantDoppler(A), filt, 1, snrs=SNRS, N_c=nc,
                             N_seg=ns, dets=cost_function_dc, seed=seed,
                             start_block=i * M, states=states)
            states = r.states
            a_hat[:, i + 1] = r.a_hat[0, :, 1]
    np.savez(f, a_hat=a_hat)                         # [snr, update]
    return f, time.time() - t0


# ---------------------------------------------------------------------------
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
    raw = out / "cost_raw"
    figd = out if q else ROOT / "figures"
    for d_ in (out, raw, figd):
        d_.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for f in raw.glob("*.npz"):
            f.unlink()
    samples = 4.8e5 if q else 9.6e6
    chunk = 2.4e5 if q else 1.2e6

    # RM closed form vs Yogev's counter (two small pairs)
    import count_ops_lag0 as co
    from ocdm_doppler.loop import LongSignal
    cnt = []
    for nc, ns in ((6000, 512), (12000, 1024)):
        p = base().with_(snr_db=4.0)
        y = LongSignal(p, 1000).rx_nominal(nc, noiseless=True)
        co.Ops.rm = 0
        co.update_counted(y, 3e-4, p, ns)
        cnt.append((nc, ns, co.Ops.rm, rm_update(nc, ns, dc=False)))

    # (a) open loop + S-curves
    tasks = []
    for nc, ns in pairs():
        n_tot = int(samples / nc)
        n_ch = max(1, int(chunk / nc))
        for s in range(0, n_tot, n_ch):
            tasks.append((str(raw / f"ol_{nc}_{ns}_{s}.npz"), nc, ns, s,
                          min(n_ch, n_tot - s)))
    sc = [(str(raw / f"sc_{ns}.npz"), ns, 2 if q else 8) for ns in NSEGS]
    if not run_tasks(_ol_task, tasks + [], args.procs, "(a) open loop", t0,
                     args.budget):
        return
    if not run_tasks(_scurve_task, sc, args.procs, "(a) S-curves", t0,
                     args.budget):
        return
    meas = {}
    for nc, ns in pairs():
        E = np.concatenate([np.load(t[0])["e"] for t in tasks
                            if t[1] == nc and t[2] == ns], axis=1)
        for s, snr in enumerate(SNRS):
            e = E[s]
            K = float((e[:, 0] - e[:, 2]).mean() / (2 * d_off(ns)))
            b = e[:, 1] / K
            meas[nc, ns, snr] = dict(K=K, sek=float(b.std(ddof=1)),
                                     n=e.shape[0], bias=float(b.mean()),
                                     bias_se=float(b.std(ddof=1)
                                                   / math.sqrt(b.size)))
    pull = {}
    for f, ns, _ in sc:
        z = np.load(f)
        offs, me = z["offs"], z["e"]
        up = offs[(offs > 0) & (me >= 0)]
        dn = offs[(offs < 0) & (me <= 0)]
        pull[ns] = (float(dn.max()) if dn.size else np.nan,
                    float(up.min()) if up.size else np.nan)

    # (b) prediction at fixed physical noise bandwidth
    T0 = NC0 / FS
    B = (G0 / (2 - G0)) / T0                      # Hz (B_L per second)

    def design(nc, M):
        Bb = B * M * nc / FS                      # B_L per update
        g = 2 * Bb / (1 + Bb)
        return Bb, g
    pred = []
    for nc, ns in pairs():
        for M in (1, 2, 4):
            Bb, g = design(nc, M)
            for snr in SNRS:
                m = meas[nc, ns, snr]
                pred.append(dict(N_c=nc, N_seg=ns, M=M, snr_db=snr,
                                 RM_per_sample=rm_update(nc, ns) / nc,
                                 RM_per_s=rm_update(nc, ns) / nc * FS / M,
                                 K_d=m["K"], s_ek=m["sek"], B_L_upd=Bb, g=g,
                                 rmse_pred=m["sek"] * math.sqrt(Bb)))

    # (c) closed-loop checks
    lt = []
    for nc, ns, M in CHECKS:
        Bb, g = design(nc, M)
        mu = g / meas[nc, ns, float("inf")]["K"]
        tau = 1 / g
        n = int((6 if q else 33) * tau)
        for r in range(1 if q else 3):
            lt.append((str(raw / f"loop_{nc}_{ns}_M{M}_r{r}.npz"), nc, ns,
                       M, mu, n, 1000 + r))
    if not run_tasks(_loop_task, lt, args.procs, "(c) closed loop", t0,
                     args.budget):
        return
    checks = []
    for nc, ns, M in CHECKS:
        Bb, g = design(nc, M)
        tau = 1 / g
        D = int(3 * tau)
        A_ = np.array([np.load(t[0])["a_hat"] for t in lt
                       if (t[1], t[2], t[3]) == (nc, ns, M)])  # [R, s, i]
        for s, snr in enumerate(SNRS):
            eps = A_[:, s, D + 1:] - A
            pr = [x for x in pred if (x["N_c"], x["N_seg"], x["M"],
                                      x["snr_db"]) == (nc, ns, M, snr)][0]
            # se of the std estimate: tau_int = 1/B_L independent samples
            n_eff = eps.size * Bb
            checks.append(dict(N_c=nc, N_seg=ns, M=M, snr_db=snr,
                               R=eps.shape[0], L=eps.shape[1], g=g,
                               std=float(eps.std()),
                               rmse=float(np.sqrt(np.mean(eps ** 2))),
                               bias=float(eps.mean()),
                               rmse_pred=pr["rmse_pred"],
                               rel_se=1 / math.sqrt(2 * n_eff),
                               RM_per_s=pr["RM_per_s"]))
    report(cnt, meas, pull, pred, checks, B, out, figd, q, t0)


def snr_s(x):
    return "∞" if np.isinf(x) else f"{x:g} (WP)"


def report(cnt, meas, pull, pred, checks, B, out, figd, q, t0):
    L = ["# Accuracy vs computational cost (Δ = {0}, DC-removed detector, "
         "OCDM)", "",
         "Generated by `study_cost_accuracy.py`" + (" --quick" if q else "")
         + ". Method in the script docstring.", "",
         "## Operation count", "",
         "RM per update = 22 N_c − 16 N_seg + 12 (+1 for the DC removal: one "
         "multiplication by 1/N_c for the block mean; the subtraction is "
         "additions only). Closed form vs `count_ops_lag0.update_counted` "
         "(without DC removal):", "",
         "| N_c | N_seg | counter | closed form |", "|---|---|---|---|"]
    for nc, ns, c, f in cnt:
        L.append(f"| {nc} | {ns} | {c:,} | {f:,} |")
    L += ["", "## (a) Open loop: K_d, σ_e/K_d, capture half-width, cost", "",
          "| N_c | N_seg | RM/sample | M RM/s (M = 1) | K_d ∞ / WP | σ_e/K_d "
          "∞ | σ_e/K_d WP | capture half-width: P/N_seg, measured | blocks |",
          "|---|---|---|---|---|---|---|---|---|"]
    for nc, ns in pairs():
        m0, m4 = meas[nc, ns, float("inf")], meas[nc, ns, 4.0]
        L.append(f"| {nc} | {ns} | {rm_update(nc, ns) / nc:.3f} | "
                 f"{rm_update(nc, ns) / nc * FS / 1e6:.1f} | {m0['K']:.1f} / "
                 f"{m4['K']:.1f} | {m0['sek']:.2e} | {m4['sek']:.2e} | "
                 f"{P / ns:.2e}, −{-pull[ns][0]:.2e}/+{pull[ns][1]:.2e} | "
                 f"{m0['n']} |")
    L += ["", "σ_e/K_d × √N_c (per-sample efficiency; constant if the "
          "detector noise simply averages):", "",
          "| N_seg | " + " | ".join(f"N_c = {nc}" for nc in NCS) + " |",
          "|---|" + "---|" * len(NCS)]
    for ns in NSEGS:
        cells = []
        for nc in NCS:
            if (nc, ns, 4.0) in meas:
                cells.append(f"{meas[nc, ns, 4.0]['sek'] * math.sqrt(nc):.2e}")
            else:
                cells.append("–")
        L.append(f"| {ns} | " + " | ".join(cells) + " |")
    L += ["", f"## (b) Predicted RMSE at a fixed physical noise bandwidth "
          f"B = {B:.4f} Hz (= g/(2−g)/T_blk of the WP design g = 0.01, "
          "N_c = 24000)", "",
          "RMSE = (σ_e/K_d)·√(B·M·N_c/f_s) (noise only; bias not included).",
          "", "| N_c | N_seg | M | M RM/s | g | RMSE ∞ | RMSE WP |",
          "|---|---|---|---|---|---|---|"]
    for nc, ns in pairs():
        for M in (1, 2, 4):
            r0 = [x for x in pred if (x["N_c"], x["N_seg"], x["M"],
                                      x["snr_db"]) == (nc, ns, M,
                                                       float("inf"))][0]
            r4 = [x for x in pred if (x["N_c"], x["N_seg"], x["M"],
                                      x["snr_db"]) == (nc, ns, M, 4.0)][0]
            L.append(f"| {nc} | {ns} | {M} | {r4['RM_per_s'] / 1e6:.1f} | "
                     f"{r4['g']:.4f} | {r0['rmse_pred']:.2e} | "
                     f"{r4['rmse_pred']:.2e} |")
    L += ["", "## (c) Closed-loop checks", "",
          "R realisations × L updates after 3τ; std of â − a vs prediction "
          "(± its expected relative std err 1/√(2 N B_L)).", "",
          "| N_c | N_seg | M | SNR | g | R × L | std measured | RMSE pred "
          "| ratio (± se) | bias | RMSE measured |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in checks:
        L.append(f"| {c['N_c']} | {c['N_seg']} | {c['M']} | "
                 f"{snr_s(c['snr_db'])} | {c['g']:.4f} | {c['R']} × "
                 f"{c['L']} | {c['std']:.2e} | {c['rmse_pred']:.2e} | "
                 f"{c['std'] / c['rmse_pred']:.2f} (± {c['rel_se']:.2f}) | "
                 f"{c['bias']:+.2e} | {c['rmse']:.2e} |")
    L += ["", f"Runtime {time.time() - t0:.0f} s.", ""]
    (out / "cost_accuracy.md").write_text("\n".join(L), encoding="utf-8")
    with open(out / "cost_accuracy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pred[0]))
        w.writeheader()
        w.writerows(pred)
    print("\n".join(L))
    plot(pred, checks, figd)


def plot(pred, checks, figd):
    cols = {512: "#1b9e77", 1024: "#d95f02", 2048: "#7570b3", 4096: "#e7298a"}
    mk = {1: "o", 2: "s", 4: "^"}
    with plt.rc_context({"font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, snr, ttl in ((axes[0], 4.0, "working point (4 dB)"),
                             (axes[1], float("inf"), "noiseless")):
            for x in pred:
                if x["snr_db"] != snr:
                    continue
                ax.loglog(x["RM_per_s"] / 1e6, x["rmse_pred"], mk[x["M"]],
                          color=cols[x["N_seg"]], ms=7, mfc="none"
                          if x["M"] > 1 else cols[x["N_seg"]])
                ax.annotate(f"{x['N_c'] // 1000}k", (x["RM_per_s"] / 1e6,
                                                     x["rmse_pred"]),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=6.5, color="0.35")
            for c in checks:
                if c["snr_db"] != snr:
                    continue
                ax.errorbar(c["RM_per_s"] / 1e6, c["std"],
                            yerr=c["std"] * c["rel_se"], fmt="*", color="k",
                            ms=11, zorder=5)
            ax.set_xlabel("cost [M real multiplications per second]")
            ax.set_ylabel("steady-state RMSE of â (noise only)")
            ax.set_title(f"{ttl}: fixed noise bandwidth B", fontsize=10.5)
            ax.grid(True, which="both", color="0.9", lw=0.5)
        hs = [plt.Line2D([], [], color=cols[n], marker="o", ls="",
                         label=f"N_seg = {n}") for n in NSEGS]
        hs += [plt.Line2D([], [], color="0.3", marker=mk[M], ls="",
                          mfc="none" if M > 1 else "0.3",
                          label=f"duty cycle M = {M}") for M in (1, 2, 4)]
        hs += [plt.Line2D([], [], color="k", marker="*", ls="", ms=10,
                          label="closed-loop check (± se)")]
        axes[0].legend(handles=hs, fontsize=8, loc="upper right")
        fig.suptitle("Accuracy vs cost (labels: N_c); predicted from the "
                     "open-loop σ_e/K_d", fontsize=11)
        fig.tight_layout()
        fig.savefig(figd / "fig_cost_accuracy.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
