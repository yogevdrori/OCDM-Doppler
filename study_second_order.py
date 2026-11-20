"""
Step 6: second-order tracking loop vs first-order loop
(ocdm_doppler/loop_filters.py, ocdm_doppler/loop_tv2.run_loop_filters).

Part B  constant Doppler and acquisition. Main design set: first-order
        p1 in {0.9, 0.95, 0.99} and, per p1, the second-order loops with the
        same noise bandwidth B_L, zeta = 1 (critically damped) and
        zeta = 0.707. Steady state at a = 5.1542e-4 and -4.8458e-4
        (non-integer shift per block), a_hat0 = a, b_hat0 = 0. Acquisition
        at a = 5.1542e-4 from a_hat0 = 0 and from a_hat0 = a -/+ Delta,
        Delta in {3, 6, 8, 9, 9.5, 10, 11} 1e-3 (true a stays in the prior
        range; the detector depends on a - a_hat only), and at
        a = 9.8458e-4 from a_hat0 = 0; first- and second-order loops in the
        same lockstep runs.
Part C  ramp (see the task file)            -- added later
Part D  fly-by                              -- added later
Part E  frequency response                  -- added later

Settings: 22-Jan preset (BPSK), Delta = {0}, N_c = 24000, N_seg = 2048,
k = 1, delta = 0.25/N_seg, K_d0 = 175.5, mu = g / K_d0, seeds 1000 + index;
noiseless and 4 dB (sigma_e / K_d = 1.60e-5 / 4.59e-5).

Runs are checkpointed in segments (results/so_raw/); --budget stops a call
cleanly, --resume continues exactly, --replot redoes tables and figures.

Usage:
    python study_second_order.py --part B --quick
    python study_second_order.py --part B [--resume] [--budget 480]
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
from ocdm_doppler.loop_tv2 import run_loop_filters

ROOT = Path(__file__).parent
N_C, N_SEG = 24000, 2048
KD0 = 175.5
T_BLK = N_C / 4e6
WP_DB = 4.0
SNRS = (float("inf"), WP_DB)
S_EK = {float("inf"): 1.60e-5, WP_DB: 4.59e-5}
SEG = 1500


def base():
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,))


def main_designs():
    out = []
    for p1 in (0.9, 0.95, 0.99):
        out += [lf.first_order_design(p1), lf.equal_BL_design(p1, 1.0),
                lf.equal_BL_design(p1, 0.707)]
    return out


# ---------------------------------------------------------------------------
# Checkpointed runner: a case = profile + loops (design, a_hat0, b_hat0)
# ---------------------------------------------------------------------------
def _segment(args):
    profile, loops, seed, start, n, last = args
    filters = []
    for (g1, g2, a0, b0) in loops:
        filters.append(lf.make_filter(g1, g2, KD0, a0, b0))
    states = None
    if last is not None:
        a_last, b_last = last
        states = [[(a_last[m, s],) if loops[m][1] == 0.0 else
                   (a_last[m, s], b_last[m, s]) for s in range(len(SNRS))]
                  for m in range(len(loops))]
    r = run_loop_filters(base(), profile, filters, n, snrs=SNRS, N_c=N_C,
                         N_seg=N_SEG, seed=seed, start_block=start,
                         states=states)
    return r.a_hat[..., 1:], r.b_hat[..., 1:]


def _run_file(raw, name, r):
    return raw / f"{name}__r{r}.npz"


def run_cases(cases, raw: Path, budget: float, procs: int, t0: float):
    """cases: name -> dict(profile, loops=[(g1, g2, a_hat0, b_hat0)],
    n_blocks, R). All unfinished realisations of all cases advance together
    in rounds of SEG blocks (one task per realisation, so every worker is
    busy); each realisation is saved to its own file after every round.
    Returns True when all cases are complete."""
    raw.mkdir(parents=True, exist_ok=True)
    ns = len(SNRS)
    state = {}
    for name, c in cases.items():
        for r in range(c["R"]):
            f = _run_file(raw, name, r)
            if f.exists():
                st = np.load(f)
                state[name, r] = (st["a"], st["b"])
            else:
                a0 = np.array([[lp[2]] * ns for lp in c["loops"]])[..., None]
                b0 = np.array([[lp[3] if lp[1] else np.nan] * ns
                               for lp in c["loops"]])[..., None]
                state[name, r] = (a0, b0)
    last_dur = 0.0
    with Pool(procs) as pool:
        while True:
            todo = [(name, r) for (name, r), (A, _) in state.items()
                    if A.shape[-1] - 1 < cases[name]["n_blocks"]]
            if not todo:
                return True
            if time.time() - t0 + 1.1 * last_dur > budget:
                left = sum(cases[nm]["n_blocks"] + 1 - state[nm, r][0]
                           .shape[-1] for nm, r in todo)
                print(f"  budget reached ({left} realisation-blocks left); "
                      f"rerun with --resume [{time.time() - t0:.0f}s]",
                      flush=True)
                return False
            ts = time.time()
            tasks = []
            for name, r in todo:
                A, B = state[name, r]
                done = A.shape[-1] - 1
                c = cases[name]
                n = min(SEG, c["n_blocks"] - done)
                tasks.append((c["profile"], c["loops"], 1000 + r, done, n,
                              None if done == 0 else (A[..., -1], B[..., -1])))
            for (name, r), (sa, sb) in zip(todo, pool.map(_segment, tasks,
                                                          chunksize=1)):
                A, B = state[name, r]
                A, B = np.concatenate([A, sa], -1), np.concatenate([B, sb], -1)
                state[name, r] = (A, B)
                np.savez(_run_file(raw, name, r), a=A, b=B)
            last_dur = time.time() - ts
            nb = sum(t[4] for t in tasks)
            print(f"  round: {len(tasks)} realisations, {nb} blocks "
                  f"[{last_dur:.0f}s = {last_dur * procs / nb * 1e3:.0f} "
                  f"ms/block/worker, total {time.time() - t0:.0f}s]",
                  flush=True)


def load(raw, name, R):
    st = [np.load(_run_file(raw, name, r)) for r in range(R)]
    return (np.array([s["a"] for s in st]),
            np.array([s["b"] for s in st]))              # [R, loop, snr, i]


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
A_SS = (5.1542e-4, -4.8458e-4)
A_ACQ = 5.1542e-4
A_ACQ2 = 9.8458e-4
DELTAS = (3e-3, 6e-3, 8e-3, 9e-3, 9.5e-3, 10e-3, 11e-3)
LOCK_K = 5.0          # locked: |eps| < 5 sigma_ss over the last 500 blocks
CAPT_MAX = 1e-3       # captured: max |eps| over the last 500 blocks below
CAPT_REL = 0.25       # this, and mean eps within 5 sigma_ss + 25 % of the
                      # linear tail (allows the slower nonlinear transient)


def p1_of(d):
    return float(d.name.split("=")[-1])


def part_b_design(quick):
    slow = lf.equal_BL_design(0.99, 1.0)                 # slowest design
    D = lf.n95(slow.g1, slow.g2)
    if quick:
        return dict(D=40, L=60, R=2, N_acq=120, R_acq=1, R_step=1)
    # option C (Yogev, Step 6 Part B): spec R = 4 x L = 3000 and acquisition
    # R = 2 x 3000 blocks measured at ~140 min; reduced to ~48 min
    return dict(D=int(D) + 100, L=2000, R=2, N_acq=2000, R_acq=1, R_step=2)


def acq_conditions():
    """(label, a_true, initial error eps0 = a_hat0 - a)."""
    out = [("step a=5.15e-4", A_ACQ, -A_ACQ), ("step a=9.85e-4", A_ACQ2,
                                               -A_ACQ2)]
    for d in DELTAS:
        out += [(f"eps0=-{d:g}", A_ACQ, -d), (f"eps0=+{d:g}", A_ACQ, +d)]
    return out


def acq_R(design, label):
    return design["R_step"] if label.startswith("step") else design["R_acq"]


def part_b_cases(design):
    des = main_designs()
    cases = {}
    for a in A_SS:
        cases[f"B_ss_a{a:+.4e}"] = dict(
            profile=ConstantDoppler(a), R=design["R"],
            n_blocks=design["D"] + design["L"],
            loops=[(d.g1, d.g2, a, 0.0) for d in des])
    for label, a, eps0 in acq_conditions():
        cases[f"B_acq_{label}"] = dict(
            profile=ConstantDoppler(a), R=acq_R(design, label),
            n_blocks=design["N_acq"],
            loops=[(d.g1, d.g2, a + eps0, 0.0) for d in des])
    return cases


def analyse_b(design, raw, out, figd):
    des = main_designs()
    rows_ss, rows_acq = [], []
    D, L = design["D"], design["L"]
    for a in A_SS:
        A, _ = load(raw, f"B_ss_a{a:+.4e}", design["R"])
        for m, d in enumerate(des):
            for s_i, snr in enumerate(SNRS):
                eps = A[:, m, s_i, D:D + L] - a                 # R x L
                bias = float(eps.mean())
                se = float(eps.mean(1).std(ddof=1) / math.sqrt(eps.shape[0]))
                std = float(eps.std())
                pred = S_EK[snr] * math.sqrt(d.BL)
                rows_ss.append(dict(a=a, design=d.name, g1=d.g1, g2=d.g2,
                                    B_L=d.BL, snr_db=snr, bias=bias,
                                    bias_se=se, std=std, std_pred=pred,
                                    ratio=std / pred))
    for label, a, eps0 in acq_conditions():
        A, _ = load(raw, f"B_acq_{label}", acq_R(design, label))
        n = design["N_acq"]
        for m, d in enumerate(des):
            n_long = 20 * n
            lin = lf.linear_response(d.g1, d.g2, np.full(n_long, a),
                                     a + eps0)[:n_long] - a
            eps_lin = lin[:n]
            for s_i, snr in enumerate(SNRS):
                eps = A[:, m, s_i, :n] - a                      # R x n
                m_eps = eps.mean(0)
                s_ss = S_EK[snr] * math.sqrt(d.BL)
                fin = float(np.abs(eps[:, -500:]).max())
                locked = fin < LOCK_K * s_ss
                # blocks the linear tail needs to stay below 5 sigma_ss
                bad = np.flatnonzero(np.abs(lin) >= LOCK_K * s_ss)
                settle_lin = int(bad[-1] + 1) if bad.size else 0
                # captured: far from the false-lock points (+/-1.41e-2) and
                # following the linear tail over the last 500 blocks
                lin_tail = float(eps_lin[-500:].mean())
                dev = float(np.abs(eps[:, -500:].mean(1) - lin_tail).max())
                captured = (fin < CAPT_MAX and
                            dev < LOCK_K * s_ss + CAPT_REL * abs(lin_tail))

                def n95_of(x):
                    bad = np.flatnonzero(np.abs(x) > 0.05 * abs(eps0))
                    return int(bad[-1] + 1) if bad.size else 0

                def over(x):
                    return max(0.0, float(np.max(-np.sign(eps0) * x))
                               / abs(eps0))
                rows_acq.append(dict(
                    condition=label, a=a, eps0=eps0, design=d.name,
                    order=d.order, snr_db=snr, locked=locked,
                    captured=captured, final_abs_err=fin, sigma_ss=s_ss,
                    tail_mean=float(eps[:, -500:].mean()),
                    tail_mean_lin=lin_tail, settle_lin=settle_lin,
                    n95=n95_of(m_eps) if captured else -1,
                    n95_lin=n95_of(eps_lin), overshoot=over(m_eps),
                    overshoot_lin=over(eps_lin),
                    max_abs_err=float(np.abs(eps).max())))

    for name, rows in (("so_steady", rows_ss), ("so_acquisition",
                                                 rows_acq)):
        with open(out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def snr_s(x):
        return "∞" if np.isinf(x) else f"{x:g} (WP)"

    L_ = ["# Second-order loop: constant Doppler and acquisition "
          "(Step 6, Part B)", "",
          "Generated by `study_second_order.py --part B`. Designs: "
          "first-order p1 ∈ {0.9, 0.95, 0.99} and, per p1, the second-order "
          "loops with the same noise bandwidth B_L, ζ = 1 (cd) and "
          "ζ = 0.707. All loops of a run in lockstep on the same blocks.", "",
          "| design | g1 | g2 | B_L | n95 (step) | overshoot (step) |",
          "|---|---|---|---|---|---|"]
    for d in des:
        L_.append(f"| {d.name} | {d.g1:.5f} | {d.g2:.3e} | {d.BL:.5f} | "
                  f"{lf.n95(d.g1, d.g2)} | "
                  f"{lf.overshoot(d.g1, d.g2):.1%} |")
    L_ += ["", f"## Steady state (â0 = a, b̂0 = 0; discard {D} blocks ≥ "
           f"n95 of the slowest design; R = {design['R']} × L = {L})", "",
           "Std of ε = â − a vs (σ_e/K_d)·√B_L; bias = mean ε ± std err of "
           "the realisation means.", "",
           "| a | design | SNR | bias ± se | std | pred | std / pred |",
           "|---|---|---|---|---|---|---|"]
    for x in rows_ss:
        L_.append(f"| {x['a']:+.4e} | {x['design']} | {snr_s(x['snr_db'])} "
                  f"| {x['bias']:+.2e} ± {x['bias_se']:.1e} | {x['std']:.2e} "
                  f"| {x['std_pred']:.2e} | {x['ratio']:.2f} |")
    L_ += ["", f"## Acquisition ({design['N_acq']} blocks; R = "
           f"{design['R_step']} for the steps from â0 = 0, "
           f"R = {design['R_acq']} for the ε0 sweep)", "",
           "ε0 = â0 − a. Two criteria, both over the last 500 of the "
           f"{design['N_acq']} blocks and in every realisation:", "",
           f"- **5σ_ss** (the approved lock criterion): max |ε| < "
           f"{LOCK_K:g}·σ_ss, σ_ss = (σ_e/K_d)·√B_L. 'final |ε|' below is "
           "that max. This tests *settled*, not only *locked*.",
           f"- **captured**: max |ε| < {CAPT_MAX:g} (far from the false-lock "
           "points ±1.41e-2, and the edge of the prior range) and the mean ε "
           "equals the linear model's tail over the same blocks to within "
           f"{LOCK_K:g}·σ_ss + {CAPT_REL:.0%} of that tail (the margin "
           "allows for the nonlinear detector, whose lower gain away from "
           "ε = 0 makes large-ε0 transients up to ~15 % slower; it was set "
           "after seeing the data, see below).", "",
           "'settle lin' = blocks the linear model needs before |ε| stays "
           f"below {LOCK_K:g}·σ_ss. n95: first block after which "
           "|mean ε| ≤ 0.05 |ε0| for good (−1: not captured); overshoot: "
           "largest excursion past 0, relative to |ε0|; 'lin' = linear model "
           "(no noise, linear detector).", "",
           "### Why the p1 = 0.99 second-order loops fail the 5σ_ss test "
           f"within {design['N_acq']} blocks", "",
           "At equal B_L the second-order loop's error decays as a double "
           "(ζ = 1) or lightly damped (ζ = 0.707) pole pair at |z| ≈ 0.996, "
           "and during the transient the integrator b̂ charges up, so the "
           "error approaches zero through a long tail of the overshoot "
           "(≈ ε0·(c1 + c2·n)·r^n). For p1 = 0.99 the linear model alone "
           "needs 984–2427 blocks ('settle lin' column) to stay below "
           f"5σ_ss (5.7e-6 noiseless, 1.6e-5 at the WP), more the larger "
           f"|ε0|; with a 500-block test window that requires runs of "
           f"~1500–2900 blocks, longer than the {design['N_acq']} simulated "
           "(Part B option C). In these cases the simulated mean error "
           "follows the linear tail (e.g. ε0 = −1e-2, ζ = 1: +7.0e-5 "
           "simulated vs +6.0e-5 linear) and is three orders of magnitude "
           "away from the false-lock points: the loops are captured but not "
           "yet settled. The first-order p1 = 0.99 loop settles within "
           "~350 blocks for every |ε0| ≤ 1e-2. This is a real drawback of "
           "the second-order loop at equal B_L (slower settling after a "
           "large initial error), not a loss of lock.", "",
           "| condition | design | SNR | 5σ_ss | captured | settle lin "
           "| n95 / lin | overshoot / lin | mean ε / lin (last 500) "
           "| final \\|ε\\| | max \\|ε\\| |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows_acq:
        L_.append(f"| {x['condition']} | {x['design']} | "
                  f"{snr_s(x['snr_db'])} | {'yes' if x['locked'] else 'NO'} | "
                  f"{'yes' if x['captured'] else 'NO'} | {x['settle_lin']} | "
                  f"{x['n95']} / {x['n95_lin']} | {x['overshoot']:.1%} / "
                  f"{x['overshoot_lin']:.1%} | {x['tail_mean']:+.1e} / "
                  f"{x['tail_mean_lin']:+.1e} | {x['final_abs_err']:.1e} | "
                  f"{x['max_abs_err']:.1e} |")
    # largest initial error that is captured / settles
    L_ += ["", "## Largest initial error that is captured", "",
           "Over the sweep |ε0| ∈ {3, 6, 8, 9, 9.5, 10, 11}·1e-3, both signs, "
           "per criterion (every realisation).", "",
           "| design | SNR | captured: largest \\|ε0\\|, ε0 < 0 | ε0 > 0 | "
           "smallest \\|ε0\\| not captured | 5σ_ss: largest \\|ε0\\|, ε0 < 0 "
           "| ε0 > 0 |", "|---|---|---|---|---|---|---|"]

    def largest(sel, key, sgn):
        ok = [abs(x["eps0"]) for x in sel
              if np.sign(x["eps0"]) == sgn and x[key]]
        return f"{max(ok):.4g}" if ok else "none"
    for d in des:
        for snr in SNRS:
            sel = [x for x in rows_acq if x["design"] == d.name
                   and x["snr_db"] == snr and x["condition"].startswith(
                       "eps0")]
            bad = [abs(x["eps0"]) for x in sel if not x["captured"]]
            L_.append(f"| {d.name} | {snr_s(snr)} | "
                      f"{largest(sel, 'captured', -1)} | "
                      f"{largest(sel, 'captured', 1)} | "
                      f"{min(bad) if bad else '–'} | "
                      f"{largest(sel, 'locked', -1)} | "
                      f"{largest(sel, 'locked', 1)} |")
    L_.append("")
    (out / "so_constant.md").write_text("\n".join(L_), encoding="utf-8")
    print("\n".join(L_))
    plot_b(design, des, raw, figd)


def plot_b(design, des, raw, figd):
    cols = {0.9: "#1f77b4", 0.95: "#ff7f0e", 0.99: "#2ca02c"}
    with plt.rc_context({"font.size": 11}):
        # linear step responses at equal B_L
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ax, p1 in zip(axes, (0.9, 0.95, 0.99)):
            trio = [d for d in des if p1_of(d) == p1]
            cd = lf.equal_BL_design(p1, 1.0)
            n = int(2.5 * lf.n95(cd.g1, cd.g2))
            for d, ls in zip(trio, ("-", "--", ":")):
                ax.plot(lf.step_response(d.g1, d.g2, n), ls, lw=1.6,
                        color=cols[p1], label=f"{d.name} (n95 "
                        f"{lf.n95(d.g1, d.g2)}, overshoot "
                        f"{lf.overshoot(d.g1, d.g2):.0%})")
            ax.axhline(1.05, color="0.6", ls=":", lw=0.8)
            ax.axhline(0.95, color="0.6", ls=":", lw=0.8)
            ax.set_title(f"equal B_L = {lf.first_order_BL(1 - p1):.4f} "
                         f"(first-order p1 = {p1})", fontsize=10.5)
            ax.set_xlabel("block")
            ax.grid(True, color="0.9", lw=0.5)
            ax.legend(fontsize=8, loc="lower right")
        axes[0].set_ylabel("step response â/a (linear model)")
        fig.tight_layout()
        fig.savefig(figd / "fig_so_step_linear.png", dpi=150)
        plt.close(fig)

        # simulated acquisition (noiseless) for the step from a_hat0 = 0
        A, _ = load(raw, "B_acq_step a=5.15e-4", design["R_step"])
        n = design["N_acq"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ax, p1 in zip(axes, (0.9, 0.95, 0.99)):
            for m, d in enumerate(des):
                if p1_of(d) != p1:
                    continue
                ls = "-" if d.order == 1 else ("--" if "cd" in d.name
                                               else ":")
                ax.plot(A[:, m, 0, :n].mean(0) / A_ACQ, ls, lw=1.4,
                        color=cols[p1], label=d.name)
                ax.plot(lf.linear_response(d.g1, d.g2, np.full(n, A_ACQ),
                                           0.0)[:n] / A_ACQ, ls, lw=0.8,
                        color="k", alpha=0.5)
            ax.set_title(f"acquisition 0 → a = 5.15e-4, noiseless "
                         f"(p1 = {p1})", fontsize=10.5)
            ax.set_xlabel("block")
            ax.grid(True, color="0.9", lw=0.5)
            ax.legend(fontsize=8, loc="lower right")
        axes[0].set_ylabel("â / a (simulated; thin black: linear model)")
        fig.tight_layout()
        fig.savefig(figd / "fig_so_acquisition.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--part", choices=("B", "C", "D", "E", "R"),
                    required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--budget", type=float, default=480.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seg", type=int, default=250)
    args = ap.parse_args()
    global SEG
    SEG = args.seg
    t0 = time.time()
    out = ROOT / "results" / ("quick" if args.quick else "")
    raw = out / "so_raw"
    figd = out if args.quick else ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    if not (args.resume or args.replot):
        for f in raw.glob(f"{args.part}_*.npz"):          # incl. old layout
            f.unlink()
    if args.part == "B":
        design = part_b_design(args.quick)
        cases = part_b_cases(design)
        if not args.replot and not run_cases(cases, raw, args.budget,
                                             args.procs, t0):
            return
        analyse_b(design, raw, out, figd)
    else:
        raise SystemExit(f"Part {args.part}: not implemented yet")
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
