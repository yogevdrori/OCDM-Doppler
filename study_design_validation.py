"""
Validation of the proposed low-cost design (Yogev, 23.11):

    NEW  N_c = 48000, N_seg = 4096, duty cycle M = 4 (one block in four)
    CUR  N_c = 24000, N_seg = 2048, M = 1 (the current WP design)

both with the DC-removed detector (cbsfs_dc), OCDM single path, first-order
loop at the FIXED physical noise bandwidth of study_cost_accuracy.py
(B = 0.8375 Hz: g = 2B'/(1 + B'), B' = B M N_c / f_s; NEW g = 0.0773,
CUR g = 0.01), mu = g / K_d (noiseless K_d from results/cost_raw).

--part 1  accuracy at constant a = 5.1542e-4, noiseless and WP: NEW, R = 6
          seeds x 650 updates after 40, std err from batch means of
          >= 10 loop time constants; CUR from study_cost_accuracy's saved
          check runs (R = 3 x 3000).
--part 2  frozen phase, noiseless: a on a grid with a M N_c in [95, 97]
          (fractional parts 0, .02, .05, .1, .2, .35, .5, .65, .8, .9, .95,
          .98 of each step, plus 97), both designs on the same grid; bias
          vs frac(a M N_c), fraction of a with |bias| > 1e-6.
--part 3  tracking: Step 5 ramp r = 1 (beta = 3.25e-8 per 6 ms block,
          5.42e-6 /s), up and down over the same a interval, same physical
          durations (5.4 s settling + 18 s), R = 4, noiseless and WP;
          bias-free lag, bias, RMSE vs prediction.

Output: results/design_validation.md, raw results/design_raw/,
figures/fig_design_frozen_phase.png.
Usage: .venv\\Scripts\\python.exe study_design_validation.py [--quick]
       [--part 1|2|3|all] [--resume] [--procs 3] [--budget 540]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import audit_bias_se as ab
from ocdm_doppler import preset_noisy_22jan
from ocdm_doppler import loop_filters as lf
from ocdm_doppler.cbsfs_dc import cost_function_dc
from ocdm_doppler.channel import single_path
from ocdm_doppler.doppler_profile import ConstantDoppler, RampDoppler
from ocdm_doppler.loop_wf2 import run_loop_det
from ocdm_doppler.waveforms import OCDMWaveform
from study_otfs import run_tasks

ROOT = Path(__file__).parent
FS = 4e6
B = (0.01 / 1.99) / (24000 / FS)                 # Hz, as study_cost_accuracy
A0 = 5.1542e-4
SNRS = (float("inf"), 4.0)
DESIGNS = {"NEW": dict(N_c=48000, N_seg=4096, M=4),
           "CUR": dict(N_c=24000, N_seg=2048, M=1)}
FRACS = (0, .02, .05, .1, .2, .35, .5, .65, .8, .9, .95, .98)
BETA_BLK = 3.25e-8                               # Step 5 r = 1, per 6 ms
RATE = BETA_BLK / (24000 / FS)                   # per second


def gain(d):
    Bb = B * d["M"] * d["N_c"] / FS
    return 2 * Bb / (1 + Bb), Bb


def kd_noiseless(d):
    """Noiseless K_d of the DC-removed detector from study_cost_accuracy."""
    raw = ROOT / "results" / "cost_raw"
    fs = sorted(raw.glob(f"ol_{d['N_c']}_{d['N_seg']}_*.npz"))
    E = np.concatenate([np.load(f)["e"] for f in fs], axis=1)[0]
    dof = 0.02 * 20 / d["N_seg"]
    return float((E[:, 0] - E[:, 2]).mean() / (2 * dof))


def base(a):
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,),
                                                  a_true=a)


def run_design(dname, mu, profile, a_hat0, n, seed, snrs):
    """a_hat[snr, n + 1] and a_bar[n] (per processed block) of one run."""
    d = DESIGNS[dname]
    p = base(float(profile.a(0.0)))
    filt = [lf.make_filter(mu, 0.0, 1.0, a_hat0=a_hat0)]
    kw = dict(snrs=snrs, N_c=d["N_c"], N_seg=d["N_seg"],
              dets=cost_function_dc, seed=seed)
    if d["M"] == 1:
        r = run_loop_det(p, OCDMWaveform(), single_path(p), profile, filt, n,
                         **kw)
        return r.a_hat[0], r.a_bar
    a_hat = np.empty((len(snrs), n + 1))
    a_hat[:, 0] = a_hat0
    a_bar = np.empty(n)
    states = None
    for i in range(n):
        r = run_loop_det(p, OCDMWaveform(), single_path(p), profile, filt, 1,
                         start_block=i * d["M"], states=states, **kw)
        states = r.states
        a_hat[:, i + 1] = r.a_hat[0, :, 1]
        a_bar[i] = r.a_bar[0]
    return a_hat, a_bar


def _task(args):
    f, dname, mu, prof_spec, a_hat0, n, seed, snrs = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    kind = prof_spec[0]
    prof = (ConstantDoppler(prof_spec[1]) if kind == "const"
            else RampDoppler(prof_spec[1], prof_spec[2]))
    a_hat, a_bar = run_design(dname, mu, prof, a_hat0, n, seed, snrs)
    np.savez(f, a_hat=a_hat, a_bar=a_bar)
    return f, time.time() - t0


def snr_s(x):
    return "∞" if np.isinf(x) else f"{x:g} (WP)"


# ---------------------------------------------------------------------------
def part1(args, raw, t0, mus):
    q = args.quick
    g, Bb = gain(DESIGNS["NEW"])
    tau = 1 / g
    D, L, R = (10, 60, 2) if q else (40, 650, 6)
    tasks = [(str(raw / f"p1_NEW_r{r}.npz"), "NEW", mus["NEW"],
              ("const", A0), A0, D + L, 1000 + r, SNRS) for r in range(R)]
    if not run_tasks(_task, tasks, args.procs, "part 1", t0, args.budget):
        return None
    A = np.array([np.load(t[0])["a_hat"] for t in tasks])     # [R, s, i]
    Lr = ["## 1. Accuracy at constant a = 5.1542e-4", "",
          f"NEW: g = {g:.4f} (τ = {tau:.0f} updates, update every "
          f"{4 * 48000 / FS * 1e3:.0f} ms), R = {R} seeds × {L} updates after "
          f"{D}; std err from batch means of ≥ 10τ (corrected method of the "
          "audit). CUR: study_cost_accuracy's check runs (R = 3 × 3000 "
          "blocks, g = 0.01), the same detector and B.", "",
          "| design | SNR | RMSE measured | RMSE predicted | ratio | bias ± se "
          "(method) | cost [M RM/s] |", "|---|---|---|---|---|---|---|"]
    pred = {("NEW", float("inf")): None}
    import csv
    pr = list(csv.DictReader(open(ROOT / "results" / "cost_accuracy.csv",
                                  encoding="utf-8")))

    def predicted(d, snr):
        for x in pr:
            if (int(x["N_c"]), int(x["N_seg"]), int(x["M"]),
                    float(x["snr_db"])) == (d["N_c"], d["N_seg"], d["M"],
                                            snr):
                return float(x["rmse_pred"]), float(x["RM_per_s"])
    rows = []
    for dname in ("NEW", "CUR"):
        d = DESIGNS[dname]
        gd, _ = gain(d)
        if dname == "NEW":
            arr = A[:, :, D + 1:]
        else:
            cr = ROOT / "results" / "cost_raw"
            arr = np.array([np.load(cr / f"loop_24000_2048_M1_r{r}.npz")
                            ["a_hat"] for r in range(3)])[:, :, 301:]
        for s, snr in enumerate(SNRS):
            x = arr[:, s] - A0
            au = ab.audit(x, old_method="real", tau_design=1 / gd)
            rp, cost = predicted(d, snr)
            rmse = float(np.sqrt(np.mean(x ** 2)))
            Lr.append(f"| {dname} | {snr_s(snr)} | {rmse:.2e} | {rp:.2e} | "
                      f"{rmse / rp:.2f} | {au['bias']:+.2e} ± "
                      f"{au['se_new']:.1e} ({au['new_from']}) | "
                      f"{cost / 1e6:.1f} |")
            rows.append((dname, snr, rmse, rp))
    Lr.append("")
    return Lr


def grid():
    k = np.array([95 + f for f in FRACS] + [96 + f for f in FRACS] + [97.0])
    return k / (4 * 48000)                       # a with a M N_c = k


def part2(args, raw, t0, mus):
    q = args.quick
    a_grid = grid()
    if q:
        a_grid = a_grid[[0, 5, 12, 18, 24]]
    lens = {"NEW": (10, 40) if q else (40, 400),
            "CUR": (30, 100) if q else (300, 3000)}
    tasks = []
    for dname in ("NEW", "CUR"):
        D, L = lens[dname]
        for i, a in enumerate(a_grid):
            tasks.append((str(raw / f"p2_{dname}_{i}.npz"), dname,
                          mus[dname], ("const", float(a)), float(a), D + L,
                          1000, (float("inf"),)))
    if not run_tasks(_task, tasks, args.procs, "part 2", t0, args.budget):
        return None
    res = {}
    for dname in ("NEW", "CUR"):
        D, L = lens[dname]
        d = DESIGNS[dname]
        gd, _ = gain(d)
        out = []
        for i, a in enumerate(a_grid):
            x = np.load(raw / f"p2_{dname}_{i}.npz")["a_hat"][0, D + 1:] - a
            au = ab.audit(x[None, :], old_method="real", tau_design=1 / gd)
            shift = a * d["M"] * d["N_c"]
            out.append((a, shift, shift - math.floor(shift + 1e-9),
                        au["bias"], au["se_new"]))
        res[dname] = out
    Lr = ["## 2. Frozen phase (noiseless closed loop)", "",
          "Grid: a M N_c (NEW) ∈ [95, 97] in steps of the listed fractions; "
          "frozen values (a M N_c integer) are 1/(M N_c) = 5.21e-6 apart in a "
          "for NEW and 1/N_c = 4.17e-5 for CUR (so CUR has only one frozen "
          "point, a = 5e-4, on this grid). NEW: 400 updates after 40, CUR: "
          "3000 blocks after 300 (std err ≈ 3e-7 each, model / τ_int of the "
          "audit).", "",
          "| a | NEW: a·M·N_c | NEW bias ± se | CUR: a·N_c | CUR bias ± se |",
          "|---|---|---|---|---|"]
    for (a, s1, f1, b1, e1), (_, s2, f2, b2, e2) in zip(res["NEW"],
                                                        res["CUR"]):
        Lr.append(f"| {a:.5e} | {s1:.2f} | {b1:+.2e} ± {e1:.1e} | {s2:.3f} | "
                  f"{b2:+.2e} ± {e2:.1e} |")
    Lr += ["", "| design | points | |bias| > 1e-6 | |bias| > 1e-6 + 2 se | "
           "max |bias| | max at frac |", "|---|---|---|---|---|---|"]
    for dname in ("NEW", "CUR"):
        o = res[dname]
        nb = sum(abs(x[3]) > 1e-6 for x in o)
        ns = sum(abs(x[3]) > 1e-6 + 2 * x[4] for x in o)
        mx = max(o, key=lambda x: abs(x[3]))
        Lr.append(f"| {dname} | {len(o)} | {nb}/{len(o)} = {nb / len(o):.0%} "
                  f"| {ns}/{len(o)} = {ns / len(o):.0%} | {mx[3]:+.2e} | "
                  f"{mx[2]:.2f} |")
    Lr.append("")
    plot2(res, args)
    return Lr


def plot2(res, args):
    figd = ROOT / "results" / "quick" if args.quick else ROOT / "figures"
    with plt.rc_context({"font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for dname, c in (("NEW", "#d62728"), ("CUR", "#1f77b4")):
            o = res[dname]
            fr = np.array([x[2] for x in o])
            fr = np.where(fr > 0.5, fr - 1, fr)       # centre on integers
            b = np.array([x[3] for x in o])
            e = np.array([x[4] for x in o])
            lab = ("NEW (48000/4096, M = 4): frac(a·M·N_c)" if dname == "NEW"
                   else "CUR (24000/2048, M = 1): frac(a·N_c)")
            axes[0].errorbar(fr, b, yerr=e, fmt="o", color=c, ms=4,
                             capsize=2, label=lab)
            axes[1].errorbar([x[0] for x in o], b, yerr=e, fmt="o-", color=c,
                             ms=4, capsize=2, label=dname)
        for ax in axes:
            for v in (1e-6, -1e-6):
                ax.axhline(v, color="0.5", ls=":", lw=0.9)
            ax.axhline(0, color="0.3", lw=0.6)
            ax.grid(True, color="0.9", lw=0.5)
            ax.legend(fontsize=8)
            ax.set_ylabel("noiseless closed-loop bias")
        axes[0].set_xlabel("fractional block shift (centred on the frozen "
                           "points)")
        axes[1].set_xlabel("a")
        axes[0].set_title("bias vs sampling-phase drift per update",
                          fontsize=10.5)
        axes[1].set_title("bias over the a grid (dotted: ±1e-6)",
                          fontsize=10.5)
        fig.tight_layout()
        fig.savefig(figd / "fig_design_frozen_phase.png", dpi=150)
        plt.close(fig)


def part3(args, raw, t0, mus):
    q = args.quick
    R = 1 if q else 4
    out = []
    plan = {}
    for dname, d in DESIGNS.items():
        Tu = d["M"] * d["N_c"] / FS
        D = int(round(5.4 / Tu)) if not q else int(round(0.6 / Tu))
        L = int(round(18.0 / Tu)) if not q else int(round(1.2 / Tu))
        half = (D + L / 2) * Tu * RATE
        plan[dname] = (D, L, Tu, half)
        for direction, sgn in (("up", 1), ("down", -1)):
            a_start = 5e-4 - sgn * half
            for r in range(R):
                out.append((str(raw / f"p3_{dname}_{direction}_r{r}.npz"),
                            dname, mus[dname], ("ramp", a_start, sgn * RATE),
                            a_start, D + L, 1000 + r, SNRS))
    if not run_tasks(_task, out, args.procs, "part 3", t0, args.budget):
        return None
    Lr = ["## 3. Tracking: Step 5 ramp r = 1", "",
          f"β = {BETA_BLK:.3g} per 6 ms block = {RATE:.3g} /s, up and down "
          "over the same a interval (centre 5e-4), same physical durations "
          "(5.4 s settling + 18 s measured), R = 4. Predicted lag −β_u/g "
          "(β_u = rate × update interval); RMSE pred. √(lag² + σ_ss²), σ_ss "
          "from the open-loop σ_e/K_d (results/cost_accuracy.csv).", "",
          "| design | SNR | update [ms] | g | lag pred. | lag (bias-free) ± se "
          "| bias ± se | RMSE (per direction) | RMSE pred. |",
          "|---|---|---|---|---|---|---|---|---|"]
    import csv
    pr = list(csv.DictReader(open(ROOT / "results" / "cost_accuracy.csv",
                                  encoding="utf-8")))
    for dname, d in DESIGNS.items():
        D, L, Tu, half = plan[dname]
        g, Bb = gain(d)
        eps = {}
        for direction in ("up", "down"):
            arrs = []
            for r in range(R):
                z = np.load(raw / f"p3_{dname}_{direction}_r{r}.npz")
                arrs.append(z["a_hat"][:, :-1] - z["a_bar"][None, :])
            eps[direction] = np.array(arrs)[:, :, D:D + L]     # [R, s, L]
        for s, snr in enumerate(SNRS):
            up, dn = eps["up"][:, s], eps["down"][:, s]
            lag = ab.audit(0.5 * (up - dn), old_method="real",
                           tau_design=1 / g)
            bias = ab.audit(0.5 * (up + dn), old_method="real",
                            tau_design=1 / g)
            rmse = math.sqrt(0.5 * (np.mean(up ** 2) + np.mean(dn ** 2)))
            sek = [float(x["s_ek"]) for x in pr if
                   (int(x["N_c"]), int(x["N_seg"]), int(x["M"]),
                    float(x["snr_db"])) == (d["N_c"], d["N_seg"], d["M"],
                                            snr)][0]
            lag_p = -RATE * Tu / g
            rmse_p = math.sqrt(lag_p ** 2 + sek ** 2 * Bb)
            Lr.append(f"| {dname} | {snr_s(snr)} | {Tu * 1e3:.0f} | {g:.4f} | "
                      f"{lag_p:+.2e} | {lag['bias']:+.2e} ± {lag['se_new']:.1e}"
                      f" | {bias['bias']:+.2e} ± {bias['se_new']:.1e} | "
                      f"{rmse:.2e} | {rmse_p:.2e} |")
    Lr.append("")
    return Lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("1", "2", "3", "all"), default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--budget", type=float, default=540.0)
    args = ap.parse_args()
    t0 = time.time()
    out = ROOT / "results" / ("quick" if args.quick else "")
    raw = out / "design_raw"
    raw.mkdir(parents=True, exist_ok=True)
    parts = ("1", "2", "3") if args.part == "all" else (args.part,)
    if not args.resume:
        for pp in parts:
            for f in raw.glob(f"p{pp}_*.npz"):
                f.unlink()
    mus = {k: gain(d)[0] / kd_noiseless(d) for k, d in DESIGNS.items()}
    L = ["# Validation of the low-cost design (N_c 48000, N_seg 4096, M 4)",
         "", "Generated by `study_design_validation.py`"
         + (" --quick" if args.quick else "") + ". Fixed physical noise "
         f"bandwidth B = {B:.4f} Hz; NEW g = {gain(DESIGNS['NEW'])[0]:.4f}, "
         f"CUR g = {gain(DESIGNS['CUR'])[0]:.4f}; K_d (noiseless, DC "
         f"removed): NEW {kd_noiseless(DESIGNS['NEW']):.1f}, CUR "
         f"{kd_noiseless(DESIGNS['CUR']):.1f}.", ""]
    for pp, fn in (("1", part1), ("2", part2), ("3", part3)):
        if pp in parts or args.resume:
            try:
                r = fn(args, raw, t0, mus)
            except FileNotFoundError as exc:
                r = [f"## part {pp}: not complete ({exc.filename})", ""]
            if r is None:
                print("stopped at the budget; rerun with --resume")
                return
            L += r
    L.append(f"Runtime {time.time() - t0:.0f} s.")
    (out / "design_validation.md").write_text("\n".join(L) + "\n",
                                              encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
