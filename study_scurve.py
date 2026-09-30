"""
Step 3: early-late discriminator S-curve in the cycle-frequency domain,
contrasted with the tau-domain (AF-magnitude curve-fitting) cost.

  (a) S-curve e(a_hat - a) for a = 5e-4 (a_max = 1e-3) and a = 5e-3
      (a_max = 1e-2), noiseless / 20 dB / 10 dB, delta in {0.25, 0.5,
      0.75} / N_seg, N_mc = 64 per condition
  (b) per condition: zero-crossing bias, slope K_d, pull-in range, sigma_e at
      zero offset, implied single-block error sigma_e / K_d, and the CB-SFS
      RMSE on the same realisations
  (c) tau domain, same realisations (a_max = 1e-3, 20 dB): curve-fitting cost
      J(a) (N_a = 2^12) and e_tau = -dJ/da
  (d) figures: fig_scurve_vs_curvefit.png (main), fig_scurve_delta.png,
      fig_scurve_snr.png, fig_scurve_tau.png

Settings as everywhere else: 22-Jan preset (BPSK), N_c = 24000 samples at the
nominal rate, N_seg = 2048, harmonic k = 1, deltas = params.deltas, seeds
1000 + trial index.

VECTORISATION. All cycle frequencies needed for one realisation lie on one
uniform grid alpha_1 (1 + a) + m h, h = 1 / (64 N_seg), centred on the true
harmonic: an offset a_hat - a = j h N_sym moves the predicted harmonic to
grid point j, and delta = 16 h, 32 h, 48 h. The cost is evaluated once per
realisation on the union of the needed grid points (one cost_function call);
every e(offset, delta) is then a lookup, and equals
discriminator.early_late(...) at that a_hat (checked in --quick).

Raw costs are saved per condition (results/scurve_raw/*.npz), so an
interrupted run resumes (--resume) without recomputing finished conditions.

Usage (from this directory):
    python study_scurve.py --quick     # smoke test, 3 trials, -> results/quick/
    python study_scurve.py             # full run (~20-25 min, 4 processes;
                                       # ~10-15 s per realisation-process)
    python study_scurve.py --resume --max-new 2   # in chunks of 2 conditions
    python study_scurve.py --resume    # continue / only redo the analysis
    python study_scurve.py --replot    # tables + figures from the raw data only
"""

from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocdm_doppler import cbsfs, preset_noisy_22jan
from ocdm_doppler import af_estimation as afe
from ocdm_doppler import curve_fit as cf
from ocdm_doppler import discriminator as disc

ROOT = Path(__file__).parent

N_C, N_SEG, K = 24000, 2048, 1
H = 1.0 / (64 * N_SEG)                      # alpha grid step
DELTA_M = {0.25: 16, 0.5: 32, 0.75: 48}     # delta * N_seg -> grid points
DENSE_J = np.arange(-13, 14)                # offsets j h N_sym: +/- 1.98e-3
WIDE_J = np.arange(-200, 201, 5)            # +/- 3.05e-2, step 7.6e-4
OFF_J = np.union1d(DENSE_J, WIDE_J)
M_ALL = np.unique(np.concatenate(
    [OFF_J] + [OFF_J + s * m for m in DELTA_M.values() for s in (1, -1)]))
SNRS = (float("inf"), 20.0, 10.0)
A_MAXS = (1e-3, 1e-2)
N_A_TAU = 2 ** 12
CBSFS_REF_RMSE = {10.0: 4.5e-5, 20.0: 2.7e-5}   # results/mc_curvefit.md, S1


def params_for(a_max, snr):
    return preset_noisy_22jan(snr).with_(a_max=a_max)


def offset_of(j, p):
    """a_hat - a for grid offset index j (k = 1)."""
    return np.asarray(j) * H * p.N_sym / K


def _signal(p, seed):
    M = N_C // p.N_sym
    sig, t0 = afe.make_signal(p, n_extra_symbols=M + 2, seed=seed)
    return sig, t0, sig.rx_nominal(M * p.N_sym)


# ---------------------------------------------------------------------------
# Per-realisation work (run in worker processes)
# ---------------------------------------------------------------------------
def trial_alpha(args):
    """CB-SFS cost on the union grid for one realisation."""
    a_max, snr, seed = args
    p = params_for(a_max, snr)
    _, _, y = _signal(p, seed)
    alphas = disc.predicted_alpha(p.a_true_eff, p, K) + M_ALL * H
    return cbsfs.cost_function(y, alphas, p, N_seg=N_SEG)


def trial_tau(seed):
    """Step-2 curve-fitting cost J(a) for the same realisation (20 dB)."""
    p = params_for(1e-3, 20.0)
    sig, t0, y = _signal(p, seed)
    ak, _, _ = cbsfs.locate_harmonic(y, p, K, n_alpha=128,
                                     half_width_factor=0.05, N_seg=N_SEG)
    a_sr = ak / K / p.alpha_1_syn - 1.0
    w = afe.estimate_windowed_af(
        sig, t0, a_sr, t_candidates=(0.0, p.T_sym / (3.0 * (1.0 + p.a_max))))
    fit = cf.fit_doppler(w, p, N_a=N_A_TAU)
    return a_sr, fit.cost


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def e_matrix(C, m, normalized=True):
    """e[trial, offset j in OFF_J] for delta = m grid points."""
    idx = {v: i for i, v in enumerate(M_ALL)}
    ip = np.array([idx[j + m] for j in OFF_J])
    im = np.array([idx[j - m] for j in OFF_J])
    return disc.error_from_costs(C[:, ip], C[:, im], normalized)


def zero_crossing(off, e_mean):
    """Crossing of the mean S-curve nearest to 0 (linear interpolation)."""
    s = np.sign(e_mean)
    cand = []
    for i in range(off.size - 1):
        if s[i] > 0 and s[i + 1] <= 0:
            x = off[i] + (off[i + 1] - off[i]) * e_mean[i] / (e_mean[i]
                                                          - e_mean[i + 1])
            cand.append(x)
    return min(cand, key=abs) if cand else float("nan")


def stable_crossings(off, e_mean):
    """All + -> - zero crossings of the mean S-curve (as offset increases):
    the points where a loop a_hat += mu e would settle. Linear interp."""
    out = []
    for i in range(off.size - 1):
        if e_mean[i] > 0 and e_mean[i + 1] <= 0:
            out.append(off[i] + (off[i + 1] - off[i]) * e_mean[i]
                       / (e_mean[i] - e_mean[i + 1]))
    return out


def pull_in(off, correct, x0, skip=0.0):
    """Largest interval around x0 on which `correct` holds at every grid
    offset; returned as the outermost correct offsets on each side.

    Offsets within `skip` of x0 count as correct: there the sign is set by
    noise (|x - x0| of order sigma_e / K_d), not by the pull-in range.
    """
    if not np.isfinite(x0):
        return float("nan"), float("nan")
    correct = correct | (np.abs(off - x0) <= skip)
    lo_i = np.searchsorted(off, x0) - 1
    hi_i = lo_i + 1
    lo, hi = float("nan"), float("nan")
    i = hi_i
    while i < off.size and correct[i]:
        hi = off[i]
        i += 1
    i = lo_i
    while i >= 0 and correct[i]:
        lo = off[i]
        i -= 1
    return lo, hi


def cbsfs_estimates(C, p):
    """CB-SFS Doppler estimates from the same costs: argmax of C on the
    contiguous part of the grid, parabolic refinement."""
    cont = np.flatnonzero(np.abs(M_ALL) <= 61)
    est = []
    for c in C:
        cc = c[cont]
        i = int(np.argmax(cc))
        frac = 0.0
        if 0 < i < cc.size - 1:
            den = cc[i - 1] - 2 * cc[i] + cc[i + 1]
            if den != 0:
                frac = float(np.clip(0.5 * (cc[i - 1] - cc[i + 1]) / den,
                                     -1, 1))
        m = M_ALL[cont][i] + frac
        alpha = disc.predicted_alpha(p.a_true_eff, p, K) + m * H
        est.append(alpha * p.N_sym / K - 1.0)
    return np.array(est)


def analyse_condition(C, p, delta_key):
    off = offset_of(OFF_J, p)
    E = e_matrix(C, DELTA_M[delta_key])
    e_mean, e_std = E.mean(0), E.std(0)
    x0 = zero_crossing(off, e_mean)

    dense = np.abs(off) <= 2.0e-3 + 1e-12
    c1, c0 = np.polyfit(off[dense], e_mean[dense], 1)
    fit = c0 + c1 * off[dense]
    r2 = 1 - np.sum((e_mean[dense] - fit) ** 2) / np.sum(
        (e_mean[dense] - e_mean[dense].mean()) ** 2)
    K_d = -c1

    want = -np.sign(off - x0)
    j0 = int(np.flatnonzero(OFF_J == 0)[0])
    sig_e = float(E[:, j0].std())

    lo, hi = pull_in(off, np.sign(e_mean) == want, x0)
    frac_ok = np.mean(np.sign(E) == want[None, :], axis=0)
    # 90 % criterion outside the noise zone |x - x0| <= 2 sigma_e / K_d
    lo90, hi90 = pull_in(off, frac_ok >= 0.9, x0, skip=2.0 * sig_e / K_d)
    a_cb = cbsfs_estimates(C, p)
    err_cb = a_cb - p.a_true_eff
    return dict(off=off, E=E, e_mean=e_mean, e_std=e_std, x0=x0, K_d=K_d,
                r2=r2, lo=lo, hi=hi, lo90=lo90, hi90=hi90, sig_e=sig_e,
                sig_over_K=sig_e / K_d,
                cb_rmse=float(np.sqrt(np.mean(err_cb ** 2))),
                cb_bias=float(err_cb.mean()))


def tau_analysis(a_srs, J, p):
    """Classify the curve-fitting cost J(a) of each realisation."""
    lo, hi = p.a_min_eff, p.a_max
    a_grid = lo + (hi - lo) * np.arange(1, N_A_TAU + 1) / (N_A_TAU + 1)
    step = a_grid[1] - a_grid[0]
    a = p.a_true_eff
    rows = []
    E_tau = []
    for a_sr, j in zip(a_srs, J):
        near_sr = np.abs(a_grid - a_sr) <= 3 * step
        i_min = int(np.argmin(j))
        if i_min == 0:
            where = "a_min"
        elif i_min == j.size - 1:
            where = "a_max"
        elif near_sr[i_min]:
            where = "a_sr"
        else:
            where = "interior"
        et = -np.gradient(j, a_grid)
        # local minima of J (e_tau + -> -), excluding ends and the a_sr jump
        s = np.sign(et)
        lm = [i for i in range(1, et.size - 1)
              if s[i - 1] > 0 and s[i] <= 0 and not near_sr[i]
              and not near_sr[i - 1]]
        d_near = min((abs(a_grid[i] - a) for i in lm), default=np.nan)
        scale = np.max(np.abs(et[~near_sr])) or 1.0
        etn = et / scale
        etn[near_sr] = np.nan
        E_tau.append(etn)
        rows.append(dict(where=where, a_hat=a_grid[i_min], n_local_min=len(lm),
                         dist_nearest_min=d_near, a_sr=a_sr))
    return a_grid, np.array(E_tau), rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="3 trials per condition, outputs to results/quick/")
    ap.add_argument("--resume", action="store_true",
                    help="reuse finished conditions in the raw directory")
    ap.add_argument("--n-mc", type=int, default=64,
                    help="trials per condition at a = 5e-4 (main results)")
    ap.add_argument("--n-mc-check", type=int, default=16,
                    help="trials per condition at a = 5e-3 (only checks "
                         "that the S-curve depends on the offset alone)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--replot", action="store_true",
                    help="only redo the analysis, tables and figures from "
                         "results/scurve_raw/; never computes (exits if a "
                         "condition is missing)")
    ap.add_argument("--max-new", type=int, default=None,
                    help="compute at most this many new conditions, then "
                         "exit (use with --resume to split the run into "
                         "shorter foreground calls)")
    args = ap.parse_args()
    if args.replot:
        args.resume, args.max_new = True, 0

    n_mc = 3 if args.quick else args.n_mc
    n_mc_for = {1e-3: n_mc, 1e-2: 3 if args.quick else args.n_mc_check}
    out = ROOT / "results" / ("quick" if args.quick else "")
    fig_dir = out if args.quick else ROOT / "figures"
    raw = out / "scurve_raw"
    raw.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "scurve_log.txt"
    if not args.resume:
        for f in raw.glob("*.npz"):
            f.unlink()
        log_path.unlink(missing_ok=True)

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t_start = time.time()
    seeds = [1000 + s for s in range(n_mc)]
    log(f"study_scurve.py  n_mc={n_mc}  N_c={N_C}  N_seg={N_SEG}  k={K}  "
        f"grid h=1/(64 N_seg), {M_ALL.size} alphas per realisation, "
        f"{OFF_J.size} offsets")

    # ---------------- (a) raw costs, per condition -------------------------
    costs = {}
    n_new = 0

    def budget_left():
        return args.max_new is None or n_new < args.max_new

    with Pool(args.procs) as pool:
        for a_max in A_MAXS:
            for snr in SNRS:
                f = raw / f"alpha_amax{a_max:g}_snr{snr:g}.npz"
                nm = n_mc_for[a_max]
                seeds_c = [1000 + s for s in range(nm)]
                if f.exists():
                    d = np.load(f)
                    if d["C"].shape[0] == nm:
                        costs[(a_max, snr)] = d["C"]
                        log(f"  reuse {f.name}")
                        continue
                if not budget_left():
                    log("  --max-new reached; rerun with --resume")
                    return
                n_new += 1
                ts = time.time()
                C = np.array(pool.map(trial_alpha,
                                      [(a_max, snr, s) for s in seeds_c]))
                np.savez(f, C=C, M_ALL=M_ALL, seeds=seeds_c)
                costs[(a_max, snr)] = C
                log(f"  a_max={a_max:g} SNR={snr:g}: {nm} realisations "
                    f"[{time.time() - ts:.0f}s]")

        f = raw / "tau_amax0.001_snr20.npz"
        if f.exists() and np.load(f)["J"].shape[0] == n_mc:
            d = np.load(f)
            a_srs, J = d["a_sr"], d["J"]
            log(f"  reuse {f.name}")
        else:
            if not budget_left():
                log("  --max-new reached; rerun with --resume")
                return
            ts = time.time()
            res = pool.map(trial_tau, seeds)
            a_srs = np.array([r[0] for r in res])
            J = np.array([r[1] for r in res])
            np.savez(f, a_sr=a_srs, J=J, seeds=seeds)
            log(f"  tau domain (curve fit, N_a=2^12), 20 dB: "
                f"[{time.time() - ts:.0f}s]")

    # ---------------- consistency check of the grid lookup ------------------
    if not args.replot:
        p0 = params_for(1e-3, 20.0)
        _, _, y0 = _signal(p0, seeds[0])
        jj = np.array([-13, -2, 0, 3, 13, -200, 200])
        direct = disc.s_curve(y0, p0.a_true_eff, offset_of(jj, p0), p0,
                              delta=0.5 / N_SEG, N_seg=N_SEG)
        E_look = e_matrix(costs[(1e-3, 20.0)][:1], DELTA_M[0.5])[0]
        look = E_look[[int(np.flatnonzero(OFF_J == j)[0]) for j in jj]]
        log(f"  grid lookup vs discriminator.early_late: max |diff| = "
            f"{np.max(np.abs(direct - look)):.2e}")

    # ---------------- (b) metrics ------------------------------------------
    results = {}
    rows = []
    for a_max in A_MAXS:
        p = params_for(a_max, 20.0)
        for snr in SNRS:
            for dk in DELTA_M:
                r = analyse_condition(costs[(a_max, snr)], p, dk)
                results[(a_max, snr, dk)] = r
                ref = CBSFS_REF_RMSE.get(snr, np.nan)
                rows.append(dict(
                    a=p.a_true_eff, snr_db=snr, delta_Nseg=dk,
                    n_mc=costs[(a_max, snr)].shape[0],
                    bias=r["x0"], bias_rel=r["x0"] / p.a_true_eff,
                    K_d=r["K_d"], lin_R2=r["r2"],
                    pullin_lo=r["lo"], pullin_hi=r["hi"],
                    pullin90_lo=r["lo90"], pullin90_hi=r["hi90"],
                    sigma_e=r["sig_e"], sigma_e_over_K_d=r["sig_over_K"],
                    cbsfs_rmse_same=r["cb_rmse"], cbsfs_bias_same=r["cb_bias"],
                    ratio=r["sig_over_K"] / r["cb_rmse"],
                    cbsfs_rmse_ref=ref))

    # delta for the main figure: rule fixed in advance -- smallest
    # sigma_e / K_d at a = 5e-4, 20 dB, among deltas whose 90 % pull-in range
    # covers [a_min - a, a_max - a]
    p1 = params_for(1e-3, 20.0)
    need_lo, need_hi = p1.a_min_eff - p1.a_true_eff, p1.a_max - p1.a_true_eff
    ok = [dk for dk in DELTA_M
          if results[(1e-3, 20.0, dk)]["lo90"] <= need_lo
          and results[(1e-3, 20.0, dk)]["hi90"] >= need_hi]
    d_main = min(ok or list(DELTA_M),
                 key=lambda dk: results[(1e-3, 20.0, dk)]["sig_over_K"])
    log(f"  main delta = {d_main}/N_seg (smallest sigma_e/K_d at 20 dB among "
        f"{ok} whose 90% pull-in covers [{need_lo:.1e}, {need_hi:.1e}])")

    with open(out / "scurve.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # ---------------- (c) tau domain ---------------------------------------
    a_grid, E_tau, tau_rows = tau_analysis(a_srs, J, p1)
    where = [r["where"] for r in tau_rows]
    counts = {w: where.count(w) for w in ("a_min", "a_max", "a_sr",
                                           "interior")}
    n_lm = np.array([r["n_local_min"] for r in tau_rows])
    d_near = np.array([r["dist_nearest_min"] for r in tau_rows])
    near_thr = 1e-4
    n_near = int(np.sum(d_near < near_thr))
    a_hat_tau = np.array([r["a_hat"] for r in tau_rows])
    tau_rmse = float(np.sqrt(np.mean((a_hat_tau - p1.a_true_eff) ** 2)))
    sr_rmse = float(np.sqrt(np.mean((a_srs - p1.a_true_eff) ** 2)))

    # ---------------- tables / log -----------------------------------------
    def fe(x, f=".2e"):
        return "–" if not np.isfinite(x) else format(x, f)

    L = ["# Step 3: early-late discriminator S-curve",
         "",
         f"Generated by `study_scurve.py`. N_mc per row: {n_mc_for[1e-3]} "
         f"for a = 5e-4 (the main results), {n_mc_for[1e-2]} for a = 5e-3, "
         "which only checks that the S-curve depends on the offset alone "
         "(seeds 1000 + trial index). 22-Jan preset (BPSK), N_c = 24000, "
         "N_seg = 2048, harmonic k = 1. Offsets are a_hat - a. "
         "e > 0 means a_hat too small; K_d = -de/d(a_hat) at the crossing "
         "(linear fit over |a_hat - a| <= 2e-3). Pull-in: outermost offsets "
         "up to which the sign is correct at every offset — for the mean "
         "S-curve (mean), and in >= 90 % of trials (90 %; offsets within "
         "2 σ_e/K_d of the crossing are exempt, since there the sign is set "
         "by noise). Grid step 1.5e-4 within ±2e-3, 7.6e-4 beyond.",
         "",
         f"Main-figure delta: **{d_main} / N_seg** — smallest sigma_e / K_d "
         "at a = 5e-4, 20 dB, among the deltas whose 90 % pull-in range "
         "covers [a_min - a, a_max - a] (rule fixed before the run).",
         "",
         "## (b) Discriminator characteristics", "",
         "| a | SNR (dB) | δ·N_seg | N_mc | bias (zero crossing) | K_d | "
         "lin. R² | pull-in (mean) | pull-in (90 %) | σ_e at 0 | "
         "σ_e / K_d | CB-SFS RMSE, same data | ratio |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            f"| {r['a']:.0e} | {r['snr_db']:g} | {r['delta_Nseg']} | "
            f"{r['n_mc']} | "
            f"{fe(r['bias'])} | {r['K_d']:.1f} | {r['lin_R2']:.4f} | "
            f"[{fe(r['pullin_lo'])}, {fe(r['pullin_hi'])}] | "
            f"[{fe(r['pullin90_lo'])}, {fe(r['pullin90_hi'])}] | "
            f"{r['sigma_e']:.2e} | **{r['sigma_e_over_K_d']:.2e}** | "
            f"{r['cbsfs_rmse_same']:.2e} | {r['ratio']:.2f} |")
    L += ["",
          "CB-SFS RMSE reference from the Step-2 Monte-Carlo "
          "(`results/mc_curvefit.md`, S1, 128 trials): 4.5e-5 at 10 dB, "
          "2.7e-5 at 20 dB.",
          "",
          f"### Stable zero crossings of the mean S-curve (δ = {d_main}/"
          "N_seg, a = 5e-4)", "",
          "Points where e goes from + to − as â increases, i.e. where a loop "
          "â ← â + μe would settle. The one at ≈ 0 is the true Doppler; any "
          "other is a possible false lock for a loop started outside the "
          "pull-in range (Dirichlet sidelobes of the cost function). "
          "Searched over |â − a| ≤ 3.05e-2.", "",
          "| SNR (dB) | stable crossings (â − a) |", "|---|---|"]
    for snr in SNRS:
        r = results[(1e-3, snr, d_main)]
        xs = stable_crossings(r["off"], r["e_mean"])
        L.append(f"| {snr:g} | " + ", ".join(f"{x:+.2e}" for x in xs) + " |")
    L += ["",
          "## (c) Tau domain: curve-fitting cost J(a), same realisations "
          "(a = 5e-4, 20 dB)", "",
          f"N_a = 2^12 candidates in (a_min, a_max); a_sr from CB-SFS as in "
          f"Step 2.",
          "",
          "| global minimum of J at | a_min | a_max | a_sr (window-edge "
          "jump) | interior |",
          "|---|---|---|---|---|",
          f"| realisations (of {n_mc}) | {counts['a_min']} | "
          f"{counts['a_max']} | {counts['a_sr']} | {counts['interior']} |",
          "",
          f"- local minima of J (zero crossings + → − of e_tau = −dJ/da) away "
          f"from the ends and the a_sr jump: {int(np.sum(n_lm > 0))} of "
          f"{n_mc} realisations have any (total {int(n_lm.sum())}); within "
          f"{near_thr:.0e} of the true a: {n_near}",
          f"- RMSE of argmin J: {tau_rmse:.2e} (a_sr from CB-SFS on the same "
          f"data: {sr_rmse:.2e})",
          ""]
    (out / "scurve.md").write_text("\n".join(L), encoding="utf-8")
    log("")
    for line in L[L.index("## (b) Discriminator characteristics"):]:
        log(line)

    # ---------------- (d) figures ------------------------------------------
    plt.rcParams.update({"font.size": 10})
    sc = 1e3
    COL_SNR = {float("inf"): "#1f77b4", 20.0: "#ff7f0e", 10.0: "#2ca02c"}
    LBL_SNR = {float("inf"): "noiseless", 20.0: "20 dB", 10.0: "10 dB"}

    def band(ax, r, color, label, sel=None, ls="-"):
        x = r["off"] * sc
        sel = slice(None) if sel is None else sel
        ax.plot(x[sel], r["e_mean"][sel], color=color, lw=1.6, ls=ls,
                label=label)
        ax.fill_between(x[sel], (r["e_mean"] - r["e_std"])[sel],
                        (r["e_mean"] + r["e_std"])[sel], color=color,
                        alpha=0.18, lw=0)

    def deco(ax, xlabel=True):
        ax.axhline(0, color="0.3", lw=0.7)
        ax.axvline(0, color="0.3", lw=0.7, ls=":")
        ax.grid(True, color="0.9", lw=0.5)
        if xlabel:
            ax.set_xlabel(r"$\hat a - a$  ($\times 10^{-3}$)")

    r_main = results[(1e-3, 20.0, d_main)]
    dense = np.abs(r_main["off"]) <= 2.0e-3 + 1e-12

    # main figure (for the email: must stay readable scaled to ~1000 px wide,
    # hence the larger fonts and the uncluttered left panel)
    off_tau = (a_grid - p1.a_true_eff) * sc
    jn = np.array([(j - j.min()) / (np.ptp(j) or 1) for j in J])
    # one representative realisation per class of the global minimum of J
    reps = []
    for cls in ("a_max", "a_sr", "a_min", "interior"):
        idx = [i for i, r in enumerate(tau_rows) if r["where"] == cls]
        if idx:
            reps.append(idx[0])

    with plt.rc_context({"font.size": 12}):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9),
                                 gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
        ax = axes[0]
        for n, i in enumerate(reps):
            ax.plot(off_tau, jn[i], color="0.7", lw=1.0,
                    label="single realisations" if n == 0 else None)
        ax.plot(off_tau, jn.mean(0), color="black", lw=2.2,
                label=f"mean of {n_mc}")
        ax.axvline(0, color="0.3", lw=0.8, ls=":")
        ax.set_xlim(off_tau[0], off_tau[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(r"$a - a_{true}$  ($\times 10^{-3}$)")
        ax.set_ylabel("normalised cost J")
        ax.set_title("τ domain: curve-fitting cost J(a)", fontsize=12.5)
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(loc="upper right", fontsize=10)
        ax.text(0.03, 0.04,
                f"global min at $a_{{max}}$: {counts['a_max']}/{n_mc}\n"
                f"at $\\hat a_{{sr}}$ jump: {counts['a_sr']}/{n_mc}\n"
                f"local min near true $a$: {n_near}/{n_mc}",
                transform=ax.transAxes, fontsize=10.5, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))

        ax = axes[1]
        band(ax, r_main, COL_SNR[20.0], "mean ± 1 std", sel=dense)
        deco(ax)
        ax.set_xlim(off_tau[0], off_tau[-1])
        ax.set_ylabel("discriminator output e")
        ax.text(0.03, 0.04,
                f"slope $K_d$ = {r_main['K_d']:.0f}\n"
                f"$\\sigma_e$ = {r_main['sig_e']:.1e} (band too thin to see)\n"
                f"$\\sigma_e/K_d$ = {r_main['sig_over_K']:.1e}\n"
                f"(CB-SFS RMSE {r_main['cb_rmse']:.1e})",
                transform=ax.transAxes, fontsize=10.5, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_title(f"α domain: S-curve (δ = {d_main}/N_seg)",
                     fontsize=12.5)
        ax.legend(loc="upper right", fontsize=10)

        ax = axes[2]
        for snr in SNRS:
            band(ax, results[(1e-3, snr, d_main)], COL_SNR[snr],
                 LBL_SNR[snr])
        ax.axvspan(r_main["lo90"] * sc, r_main["hi90"] * sc, color="0.85",
                   zorder=0, label="pull-in (20 dB)")
        ax.axvspan(need_lo * sc, need_hi * sc, color="#c7e9c0", zorder=0,
                   label=r"$[a_{min}-a,\ a_{max}-a]$")
        xs = [x for x in stable_crossings(r_main["off"], r_main["e_mean"])
              if abs(x) > 1e-3]
        ax.plot(np.array(xs) * sc, np.zeros(len(xs)), "v", color="#d62728",
                ms=8, label="false-lock points")
        deco(ax)
        ax.set_title("α domain: pull-in range", fontsize=12.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3,
                  fontsize=9.5, frameon=False, columnspacing=1.0,
                  handletextpad=0.5)
        fig.suptitle("Doppler discriminator: α domain (CB-SFS cost) vs "
                     "τ domain (AF curve fit), a = 5e-4, 20 dB",
                     fontsize=13.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_scurve_vs_curvefit.png", dpi=150)
        plt.close(fig)

    # delta comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cols = {0.25: "#1f77b4", 0.5: "#ff7f0e", 0.75: "#2ca02c"}
    for dk in DELTA_M:
        r = results[(1e-3, 20.0, dk)]
        band(axes[0], r, cols[dk], f"δ = {dk}/N_seg")
        band(axes[1], r, cols[dk],
             f"δ = {dk}/N_seg: K_d = {r['K_d']:.0f}, "
             f"σ_e/K_d = {r['sig_over_K']:.1e}", sel=dense)
    for ax in axes:
        deco(ax)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("discriminator output e")
    axes[0].set_title("S-curve vs δ (a = 5e-4, 20 dB, mean ± 1 std)",
                      fontsize=10)
    axes[1].set_title("zoom: linear region", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_scurve_delta.png", dpi=150)
    plt.close(fig)

    # SNR comparison and independence of a
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for snr in SNRS:
        band(axes[0], results[(1e-3, snr, d_main)], COL_SNR[snr],
             LBL_SNR[snr], sel=dense)
    for a_max, ls, c in ((1e-3, "-", "#ff7f0e"), (1e-2, "--", "#9467bd")):
        r = results[(a_max, 20.0, d_main)]
        axes[1].plot(r["off"] * sc, r["e_mean"], color=c, ls=ls, lw=1.6,
                     label=f"a = {a_max / 2:.0e}")
    for ax in axes:
        deco(ax)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("discriminator output e")
    axes[0].set_title(f"S-curve vs SNR (a = 5e-4, δ = {d_main}/N_seg, "
                      "mean ± 1 std)\nthe mean curves coincide; only σ_e "
                      "grows with noise", fontsize=10)
    axes[1].set_title("mean S-curve, a = 5e-4 vs 5e-3 (20 dB):\n"
                      "depends only on the offset", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_scurve_snr.png", dpi=150)
    plt.close(fig)

    # tau-domain discriminator next to the alpha-domain S-curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    m_tau = np.nanmean(E_tau, 0)
    s_tau = np.nanstd(E_tau, 0)
    axes[0].plot(off_tau, m_tau, color="black", lw=1.4, label="mean")
    axes[0].fill_between(off_tau, m_tau - s_tau, m_tau + s_tau, color="0.5",
                         alpha=0.25, lw=0, label="± 1 std")
    axes[0].set_title(r"τ domain: $e_\tau = -dJ/da$ (normalised; "
                      r"$\pm 3$ grid steps around $\hat a_{sr}$ removed)",
                      fontsize=9.5)
    band(axes[1], r_main, COL_SNR[20.0], "mean ± 1 std", sel=dense)
    axes[1].set_title(f"α domain: early-late e (δ = {d_main}/N_seg)",
                      fontsize=9.5)
    for ax in axes:
        deco(ax)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_xlim(off_tau[0], off_tau[-1])
    axes[0].set_ylabel("discriminator output")
    fig.suptitle(f"Same {n_mc} realisations, a = 5e-4, 20 dB", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_scurve_tau.png", dpi=150)
    plt.close(fig)

    log(f"\ntotal runtime {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
