"""
Step 7 follow-ups (Yogev, 22.11):

--part 1  Harmonic strength vs sigma_e/K_d (OCDM, every Part D channel,
          noiseless, after the power normalisation G): the k = 1 Fourier
          coefficient of the exact mean received power (analytic, incl. the
          cross terms) and the cost C at the true cycle frequency
          (simulated, 20 blocks, DC-removed detector), both relative to the
          single path, next to the sigma_e/K_d ratios of Part D.
--part 2  Multiscale consistency: noiseless open-loop cost C(a_hat) and
          e(a_hat) over a_hat in [a1 - 5e-4, a2 + 5e-4] at blocks 10 and 490
          (2:1 and equal powers), and the per-block lock point (zero of the
          noiseless e) over all 500 blocks, against the capture-effect
          prediction (unresolved lines with amplitude ratio r = p2/p1:
          per-block lock point between -r/(1-r) and r/(1+r) of a2 - a1,
          mean 0 for r < 1) -> figures/fig_multiscale_cost.png.
--part 3  OCDM fixed gains, D_max = 2 and 4: 4 more noise / data seeds
          (16 more batches of 150 blocks) to confirm or reject the +3e-6
          WP bias of Part D.

Output: results/step7_followups.md (sections appended per part), raw
results/followup_raw/.  Run with .venv\\Scripts\\python.exe.
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

import study_otfs as so
from ocdm_doppler import discriminator as disc
from ocdm_doppler.channel import rx_block, single_path
from ocdm_doppler.doppler_profile import ConstantDoppler
from ocdm_doppler.wf_params import detector_params

ROOT = Path(__file__).parent
N_C = 24000


def ocdm_cases(quick):
    d = so.design_d(quick)
    return [c for c in so.d_cases(d) if c[0] == "OCDM" and c[4] < (
        d["R_fix"] if c[3] == "fixed" else d["R_ray"])]


def pattern(ws, ch, p, n_sub=16, skip=10):
    """Exact mean received power m(t) over one period (incl. cross terms)."""
    P = ws.P_samp
    t = (skip * P + (np.arange(P * n_sub) + 0.5) / n_sub) * p.T_samp_syn
    hs = [p.A_p0 * complex(q.gain) * np.exp(-2j * np.pi * p.f_Tx * q.delay
                                            * p.T_samp_syn) / 2.0
          for q in ch.paths]
    taus = [q.delay * p.T_samp_syn for q in ch.paths]
    m = np.zeros(t.size)
    for hp, tp in zip(hs, taus):
        for hq, tq in zip(hs, taus):
            m += np.real(hp * np.conj(hq) * ws.autocorr(t - tp, t - tq))
    return t, m, P * p.T_samp_syn


# ---------------------------------------------------------------------------
def _p1_task(args):
    f, w, si, D_max, g, r, seed, n = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    c, ch, pn, G = so._d_setup(w, si, D_max, g, r, seed)
    ws = so.make_wf(w).signal(pn, seed)
    t, m, Tp = pattern(ws, ch, pn)
    c1 = complex(np.mean(m * np.exp(-2j * np.pi * t / Tp)))
    dp = detector_params(pn, ws.P_samp)
    al = np.array([disc.predicted_alpha(so.A_D, dp, 1)])
    C = [so.DETS["dc"](rx_block(ws, ch, pn, N_C, ConstantDoppler(so.A_D),
                                start_sample=b * N_C), al, dp, N_seg=2048)[0]
         for b in range(n)]
    np.savez(f, c1=abs(c1), m0=float(np.mean(m)), C=np.array(C), G=G)
    return f, time.time() - t0


def part1(args, raw, t0):
    q = args.quick
    n = 3 if q else 20
    cases = ocdm_cases(q)
    tasks = [(str(raw / f"p1_{so._d_name(*c[:1], c[1], c[3], c[4])}.npz"),
              *c, n) for c in cases]
    so.run_tasks(_p1_task, tasks, args.procs, "part 1", t0, args.budget)
    import csv
    rowsD = list(csv.DictReader(open(ROOT / "results" / "otfs_D.csv",
                                     encoding="utf-8")))

    def sek_rel(D_max, g, snr):
        for r in rowsD:
            if (r["waveform"] == "OCDM" and r["detector"] == "dc"
                    and float(r["D_max"]) == D_max and r["gains"] == g
                    and float(r["snr_db"]) == snr):
                return float(r["s_ek_rel"])
        return float("nan")

    ref = np.load(tasks[0][0])
    L = ["## 1. Harmonic strength vs σ_e/K_d (OCDM, DC-removed detector)", "",
         "Per channel realisation, after the power normalisation G (equal "
         "mean received power). |c1|: k = 1 Fourier coefficient of the exact "
         "mean received power (analytic, incl. cross terms); √(C/C_ref): "
         f"square root of the noiseless cost at the true cycle frequency "
         f"(mean of {n} blocks), relative to the single path. If σ_e/K_d "
         "tracked 1/(harmonic strength), the last columns would equal "
         "1/strength.", "",
         "| D_max | gains | r | G | \\|c1\\| / ref | √(C/C_ref) | 1/strength "
         "| σ_e/K_d rel., WP (Part D) | σ_e/K_d rel., noiseless |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (f, w, si, D_max, g, r, seed, _) in tasks:
        z = np.load(f)
        s_an = float(z["c1"] / ref["c1"])
        s_sim = float(np.sqrt(z["C"].mean() / ref["C"].mean()))
        L.append(f"| {D_max:g} | {g} | {r} | {float(z['G']):.3f} | "
                 f"{s_an:.3f} | {s_sim:.3f} | {1 / s_an:.2f} | "
                 f"{sek_rel(D_max, g, so.WP_DB):.2f} | "
                 f"{sek_rel(D_max, g, float('inf')):.2f} |")
    L.append("")
    L.append("(Part D's σ_e/K_d is per channel, pooled over its "
             "realisations; the strength is per realisation.)")
    L.append("")
    return L


# ---------------------------------------------------------------------------
def _p2_task(args):
    f, p2, n, snaps = args
    if Path(f).exists():
        return f, 0.0
    t0 = time.time()
    p = so.base().with_(a_true=so.MS_A1)
    ch = so.ms_channel(p2)
    ws = so.make_wf("OCDM").signal(p, 1000)
    dp = detector_params(p, ws.P_samp)
    delta = 0.25 / 2048
    fine = np.linspace(so.MS_A1 - 5e-4, so.MS_A1 + so.MS_DA + 5e-4, 221)
    coarse = np.linspace(so.MS_A1 - 3 * so.MS_DA, so.MS_A1 + 3 * so.MS_DA,
                         61)
    zeros, Cs, Es = [], [], []
    fn = so.DETS["dc"]
    for b in range(n):
        y = rx_block(ws, ch, p, N_C, ConstantDoppler(so.MS_A1),
                     start_sample=b * N_C)
        ap = disc.predicted_alpha(coarse, dp, 1)
        c = fn(y, np.concatenate([ap + delta, ap - delta]), dp, N_seg=2048)
        e = disc.error_from_costs(c[:coarse.size], c[coarse.size:])
        k = np.flatnonzero((e[:-1] > 0) & (e[1:] <= 0))
        zeros.append(np.nan if k.size == 0 else
                     coarse[k[0]] + (coarse[k[0] + 1] - coarse[k[0]])
                     * e[k[0]] / (e[k[0]] - e[k[0] + 1]))
        if b in snaps:
            apf = disc.predicted_alpha(fine, dp, 1)
            Cs.append(fn(y, apf, dp, N_seg=2048))
            cf = fn(y, np.concatenate([apf + delta, apf - delta]), dp,
                    N_seg=2048)
            Es.append(disc.error_from_costs(cf[:fine.size], cf[fine.size:]))
    np.savez(f, zeros=np.array(zeros), C=np.array(Cs), E=np.array(Es),
             fine=fine, snaps=np.array(snaps))
    return f, time.time() - t0


def capture_pred(r):
    """Per-block lock point range and mean (units of a2 - a1) for two
    unresolved lines with amplitude ratio r (instantaneous frequency of
    1 + r e^{j psi})."""
    psi = np.linspace(0, 2 * np.pi, 20001)[:-1]
    f = r * (r + np.cos(psi)) / (1 + r * r + 2 * r * np.cos(psi))
    return f.min(), f.max(), f.mean()


def part2(args, raw, t0, figd):
    q = args.quick
    n = 30 if q else 500
    snaps = (10, 25) if q else (10, 490)
    tasks = [(str(raw / f"p2_p{p2:g}.npz"), p2, n, snaps)
             for p2 in (1 / 3, 0.5)]
    so.run_tasks(_p2_task, tasks, args.procs, "part 2", t0, args.budget)
    z = {p2: np.load(f) for (f, p2, _, _) in tasks}
    ms = {p2: np.load(ROOT / "results" / ("quick" if q else "") / "otfs_raw"
                      / f"D_ms_p{p2:g}.npz") for p2 in (1 / 3, 0.5)} \
        if not q else {}
    L = ["## 2. Multiscale: resolution, cost and the capture effect", "",
         "**Units.** The detector probes the cycle frequency α_p = k(1 + â)/"
         "P_samp at α_p ± δ, δ = 0.25/N_seg (`discriminator.predicted_alpha`"
         ", `loop_wf.py`): δ is in α (cycles/sample), not in a. In units of "
         "a the probes sit at ±δ P_samp/k = ±2.44e-3 and the main lobe of "
         "C(â) has nulls at about ±P_samp/(k N_seg) = ±9.8e-3 (measured "
         "pull-in ±1.05e-2, Part B). The two paths differ by Δa = 1e-4, "
         "1 % of the lobe half-width — the same ratio as Δα = Δa/20 = 5e-6 vs "
         "1/N_seg = 4.9e-4. Within one block (N_c = 24000) the two lines are "
         "Δα·N_c = 0.12 cycles apart: unresolved by the block as well.", "",
         "**Capture effect.** Unresolved, the two harmonics act as one line "
         "1 + r e^{jψ(t)} (r = amplitude ratio of the mean-power harmonics "
         "= p2/p1; ψ turns once per 1/Δα = 2e5 samples ≈ 8.3 blocks). Its "
         "instantaneous frequency, in units of Δa above a1, is "
         "r(r + cos ψ)/(1 + r² + 2r cos ψ): per block the lock point swings "
         "between −r/(1−r) and r/(1+r); averaged over ψ it is exactly 0 for "
         "r < 1 (the stronger path, path 1) and 1 for r > 1 — not the "
         "power-weighted p2. For r = 1 the sum passes through zero and the "
         "average is undefined (the loop wanders). The loop (time constant "
         "100 blocks) averages over ~12 turns of ψ.", "",
         "| p2 | r | predicted per-block range | predicted mean | measured "
         "per-block lock point: 5–95 % | measured mean (all blocks) | loop, "
         "last 100 blocks (Part D) |", "|---|---|---|---|---|---|---|"]
    for p2 in (1 / 3, 0.5):
        r = p2 / (1 - p2)
        lo, hi, mean = capture_pred(r) if r < 1 else (np.nan, np.nan, np.nan)
        zz = (z[p2]["zeros"] - so.MS_A1) / so.MS_DA
        zz = zz[np.isfinite(zz)]
        loop = (f"{((ms[p2]['a_hat'][0, 1] - so.MS_A1) / so.MS_DA)[-100:].mean():+.3f}"
                if ms else "–")
        L.append(f"| {p2:.3g} | {r:.2f} | "
                 + (f"{lo:+.2f} … {hi:+.2f} | {mean:+.3f}" if r < 1 else
                    "undefined (r = 1) | undefined")
                 + f" | {np.percentile(zz, 5):+.2f} … "
                   f"{np.percentile(zz, 95):+.2f} | {zz.mean():+.3f} | "
                   f"{loop} |")
    L.append("")
    L.append("Figure: `figures/fig_multiscale_cost.png` (C(â) and e(â) at the "
             "two blocks; per-block lock point over the run).")
    L.append("")
    plot2(z, figd)
    return L


def plot2(z, figd):
    with plt.rc_context({"font.size": 10.5}):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for row, p2 in enumerate((1 / 3, 0.5)):
            zz = z[p2]
            x = (zz["fine"] - so.MS_A1) / so.MS_DA
            for j, b in enumerate(zz["snaps"]):
                axes[row, 0].plot(x, zz["C"][j] / zz["C"][j].max(),
                                  label=f"block {int(b)}")
                axes[row, 1].plot(x, zz["E"][j], label=f"block {int(b)}")
            r = p2 / (1 - p2)
            zr = (zz["zeros"] - so.MS_A1) / so.MS_DA
            axes[row, 2].plot(zr, ".", ms=2.5, label="per-block lock point "
                              "(zero of noiseless e)")
            run = np.convolve(np.nan_to_num(zr), np.ones(100) / 100,
                              mode="valid")
            axes[row, 2].plot(np.arange(99, 99 + run.size), run, lw=1.5,
                              label="100-block running mean")
            if r < 1:
                lo, hi, mean = capture_pred(r)
                for v in (lo, hi):
                    axes[row, 2].axhline(v, color="0.4", ls=":", lw=0.9)
                axes[row, 2].axhline(mean, color="k", lw=0.9,
                                     label="capture-effect mean (0)")
            axes[row, 2].axhline(p2, color="tab:red", ls="--", lw=0.9,
                                 label=f"power-weighted ({p2:.2f})")
            for ax in axes[row, :2]:
                for v in (0, 1):
                    ax.axvline(v, color="k", ls=":", lw=0.8)
                ax.axvline(p2, color="tab:red", ls="--", lw=0.8)
                ax.set_xlabel("(â − a1)/(a2 − a1)")
                ax.grid(True, color="0.9", lw=0.5)
                ax.legend(fontsize=8)
            axes[row, 0].set_ylabel(f"p2 = {p2:.2g}:  C(â) / max")
            axes[row, 1].set_ylabel("e(â) (noiseless)")
            axes[row, 1].axhline(0, color="0.5", lw=0.6)
            axes[row, 2].set_xlabel("block")
            axes[row, 2].set_ylabel("(lock point − a1)/(a2 − a1)")
            axes[row, 2].grid(True, color="0.9", lw=0.5)
            axes[row, 2].legend(fontsize=7)
        axes[0, 0].set_title("cost at the predicted harmonic (DC removed)")
        axes[0, 1].set_title("early-late output e(â)")
        axes[0, 2].set_title("lock point per block vs capture effect")
        fig.tight_layout()
        fig.savefig(figd / "fig_multiscale_cost.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
def part3(args, raw, t0):
    q = args.quick
    d = so.design_d(q)
    n = d["D"] + d["L"]
    seeds = (1001, 1002) if q else (1001, 1002, 1003)   # 4 + 12 batches
    base_raw = ROOT / "results" / ("quick" if q else "") / "otfs_raw"
    Kd0 = {}
    E = np.load(base_raw / f"D_ol_{so._d_name('OCDM', 0, 'fixed', 0)}.npz")["e"]
    for di, x in enumerate(so.WAVE_D["OCDM"]["dets"]):
        Kd0[x] = float((E[di, 0, :, 0] - E[di, 0, :, 2]).mean()
                       / (2 * so.WAVE_D["OCDM"]["d_off"]))
    mus = [(1 - so.POLE) / Kd0[x] for x in so.WAVE_D["OCDM"]["dets"]]
    chans = [(0, 0), (1, 2), (2, 4)]              # (spread index, D_max)
    tasks = [(str(raw / f"p3_s{si}_seed{s}.npz"), "OCDM", si, D_max,
              "fixed", 0, s, n, mus) for si, D_max in chans for s in seeds]
    so.run_tasks(so._d_loop_task, tasks, args.procs, "part 3", t0,
                 args.budget)
    bsz = min(150, d["L"] // 4)
    L = ["## 3. OCDM fixed gains: is the +3e-6 WP bias real?", "",
         f"Closed loop as Part D ({d['D']} + {d['L']} blocks, μ from the "
         "single-path K_d), seeds " + ", ".join(map(str, seeds)) + " in "
         "addition to Part D's seed 1000 (the channel is fixed; the seed "
         "sets data and noise). Single path (D_max = 0) included as the "
         f"reference. Std err from batch means of {bsz} blocks.", "",
         "| D_max | detector | SNR | Part D (seed 1000) | new seeds | all "
         "| batches |", "|---|---|---|---|---|---|---|"]
    for si, D_max in chans:
        old = np.load(base_raw / f"D_loop_{so._d_name('OCDM', si, 'fixed', 0)}"
                      ".npz")["a_hat"]
        new = [np.load(raw / f"p3_s{si}_seed{s}.npz")["a_hat"] for s in seeds]
        for di, x in enumerate(so.WAVE_D["OCDM"]["dets"]):
            for s_i, snr in enumerate(so.SNRS_D):
                def stat(arrs):
                    eps = np.array([a[di, s_i, d["D"]:d["D"] + d["L"]]
                                    for a in arrs]) - so.A_D
                    nb = eps.shape[1] // bsz
                    bm = eps[:, :nb * bsz].reshape(len(arrs), nb,
                                                   bsz).mean(-1).ravel()
                    return bm.mean(), bm.std(ddof=1) / math.sqrt(bm.size), \
                        bm.size
                o, n_, a_ = stat([old]), stat(new), stat([old] + new)
                L.append(f"| {D_max} | {so.DET_NAME[x]} | {so.snr_s(snr)} | "
                         f"{o[0]:+.2e} ± {o[1]:.1e} | {n_[0]:+.2e} ± "
                         f"{n_[1]:.1e} | {a_[0]:+.2e} ± {a_[1]:.1e} | "
                         f"{a_[2]} |")
    L.append("")
    return L


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
    raw = out / "followup_raw"
    figd = out if args.quick else ROOT / "figures"
    for d_ in (out, raw, figd):
        d_.mkdir(parents=True, exist_ok=True)
    parts = ("1", "2", "3") if args.part == "all" else (args.part,)
    if not args.resume:
        for pp in parts:
            for f in raw.glob(f"p{pp}_*.npz"):
                f.unlink()
    L = ["# Step 7 follow-ups", "",
         "Generated by `study_step7_followups.py`"
         + (" --quick" if args.quick else "") + ".", ""]
    for pp in ("1", "2", "3"):
        if pp not in parts and not args.resume:
            continue
        try:
            L += {"1": lambda: part1(args, raw, t0),
                  "2": lambda: part2(args, raw, t0, figd),
                  "3": lambda: part3(args, raw, t0)}[pp]()
        except FileNotFoundError as exc:
            L += [f"## part {pp}: not complete ({exc.filename})", ""]
    L.append(f"Runtime {time.time() - t0:.0f} s.")
    (out / "step7_followups.md").write_text("\n".join(L) + "\n",
                                            encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
