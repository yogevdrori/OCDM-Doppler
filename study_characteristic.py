"""
Step 4, Part B: open-loop detector characteristic, Delta = {0} vs {0..15}.

Fix a_hat = 0 and sweep the true Doppler a. For every (a, block) the
early-late output e (discriminator.error_from_costs, delta = 0.25/N_seg,
k = 1) is computed for both lag sets and all SNRs. x = a (= a - a_hat),
y = mean(e) / K_d with K_d the noiseless gain of the lag set (the loop's
design constant), so the ideal characteristic is y = x.

Settings: 22-Jan preset (BPSK), N_c = 24000 samples at the nominal rate,
N_seg = 2048, N_mc = 64 blocks per (a, SNR), seeds 1000 + index.

The noisy block at SNR s is the noiseless block plus sqrt(P_n(s)) times one
unit-power realisation of the *same* noise stream OCDMSignal uses
(_BlockNoise with seed ^ 0x5EED, indexed on the T_s grid) -- identical to
OCDMSignal(...).rx_nominal at that SNR (asserted in --quick), but generated
once per (a, block) instead of once per SNR. The same data and noise are
used for every a and SNR of a given block index (common random numbers):
each point is still an average over 64 independent blocks, and the curves
are smoother across a and SNR than independent draws would be.

Raw e values are saved per a (results/char_raw/), so --resume continues an
interrupted run and --replot redoes the tables and figures only.

Usage:
    python study_characteristic.py --quick       # 3 values of a, 3 blocks
    python study_characteristic.py               # full run
    python study_characteristic.py --resume | --replot
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

from ocdm_doppler import OCDMSignal, cbsfs, preset_noisy_22jan
from ocdm_doppler import discriminator as disc
from ocdm_doppler.signal_model import _BlockNoise

ROOT = Path(__file__).parent
N_C, N_SEG, K = 24000, 2048, 1
DELTA = 0.25 / N_SEG
A_HAT = 0.0
WP_DB = 4.0                              # Part A: BPSK SER = 10 % working point
SNRS = (float("inf"), 20.0, 10.0, 7.5, 5.0, WP_DB, 2.5, 0.0, -5.0)
LAGSETS = {"0": (0,), "0..15": None}     # None = params.deltas = 0..N_is-1
A_DENSE = np.round(np.linspace(-1e-3, 1e-3, 41), 12)
A_WIDE = np.round(np.arange(-2e-2, 2e-2 + 1e-12, 1e-3), 12)
A_GRID = np.unique(np.concatenate([A_DENSE, A_WIDE]))


def base_params(a=0.0, snr=float("inf")):
    return preset_noisy_22jan(snr).with_(a_true=float(a))


def unit_noise(sig: OCDMSignal, n: int, seed: int) -> np.ndarray:
    """The noise OCDMSignal(seed=seed).rx_nominal adds, at unit power."""
    t = np.arange(n, dtype=np.int64) * sig.p.T_samp_ns
    return _BlockNoise(1.0, seed=seed ^ 0x5EED).at(sig.grid_index(t))


def block(a, seed):
    p = base_params(a)
    M = N_C // p.N_sym
    sig = OCDMSignal(p, n_symbols=M + 2, seed=seed)
    y0 = sig.rx_nominal(M * p.N_sym, noiseless=True)
    return p, sig, y0, unit_noise(sig, y0.size, seed)


def trial(args):
    """e[snr, lagset] for one (a, block)."""
    a, seed = args
    p, _, y0, w = block(a, seed)
    ap_ = disc.predicted_alpha(A_HAT, p, K)
    alphas = np.array([ap_ + DELTA, ap_ - DELTA])
    out = np.empty((len(SNRS), len(LAGSETS)))
    for i, snr in enumerate(SNRS):
        y = y0 if np.isinf(snr) else y0 + np.sqrt(
            p.sigma_D2 / 10 ** (snr / 10)) * w
        for j, lags in enumerate(LAGSETS.values()):
            pl = p.with_(cbsfs_deltas=lags)
            c = cbsfs.cost_function(y, alphas, pl, N_seg=N_SEG)
            out[i, j] = disc.error_from_costs(c[0], c[1])
    return out


# ---------------------------------------------------------------------------
def crossing_up(x, y):
    """Zero crossing (- to +) of y nearest to 0, linear interpolation."""
    c = [x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
         for i in range(len(x) - 1) if y[i] < 0 <= y[i + 1]]
    return min(c, key=abs) if c else float("nan")


def pull_in(x, ok, x0, skip=0.0):
    """Outermost x on each side of x0 up to which `ok` holds everywhere
    (points within `skip` of x0 count as ok: there the sign is noise)."""
    if not np.isfinite(x0):
        return float("nan"), float("nan")
    ok = ok | (np.abs(x - x0) <= skip)
    i0 = np.searchsorted(x, x0)
    lo = hi = float("nan")
    i = i0
    while i < x.size and ok[i]:
        hi, i = x[i], i + 1
    i = i0 - 1
    while i >= 0 and ok[i]:
        lo, i = x[i], i - 1
    return lo, hi


def complexity(n_c=N_C, n_seg=N_SEG, n_lags_nonzero=15):
    """Real multiplications per received sample, early-late (2 alphas).

    Conventions (as in the Step-3 count): twiddles e^{-j2 pi alpha k} by
    recursive rotation (4 RM per alpha per sample, shared by all lags);
    segmented CAF by a running sum (additions only); autocorrelated CAF
    R[n] R*[n - N_seg] (4 RM per alpha) for n = N_seg .. N_c - N_seg, i.e.
    a fraction (N_c - 2 N_seg + 1) / N_c of the samples. Lag 0: |Y|^2 is
    real (2 RM) and real x complex modulation costs 2 RM; lag != 0:
    Y Y* (4 RM) and complex x complex modulation (4 RM).
    """
    frac = (n_c - 2 * n_seg + 1) / n_c
    tw = 2 * 4
    lag0 = 2 + 2 * 2 + 2 * 4 * frac
    lagk = 4 + 2 * 4 + 2 * 4 * frac
    lag0_complex = 4 + 2 * 4 + 2 * 4 * frac      # counting |Y|^2 as complex
    return dict(frac=frac, set0=tw + lag0, set0_upper=tw + 2 + 4 + 8,
                set0_complex=tw + lag0_complex,
                set_all=tw + lag0 + n_lags_nonzero * lagk,
                set_all_complex=tw + lag0_complex + n_lags_nonzero * lagk)


def analyse(E):
    """E[a, block, snr, lagset] -> per (snr, lagset) metrics.

    The point a = 0 is excluded from K_d, the zero crossing and max |bias|
    and reported separately: at a = 0 exactly the receiver samples
    synchronously (N_sym = 20 samples per symbol) and the detector has an
    offset of ~ +1.1e-5 that is absent at every a != 0 (README, Step 4).
    """
    a = A_GRID
    dense = np.isin(a, A_DENSE)
    i0 = int(np.flatnonzero(a == 0.0)[0])
    fitpts = dense & (a != 0.0)
    res = {}
    Kd0 = {}
    for j, ls in enumerate(LAGSETS):
        m = E[:, :, 0, j].mean(1)
        Kd0[ls] = np.polyfit(a[fitpts], m[fitpts], 1)[0]
    for i, snr in enumerate(SNRS):
        for j, ls in enumerate(LAGSETS):
            e = E[:, :, i, j]
            m, s = e.mean(1), e.std(1)
            c1, c0 = np.polyfit(a[fitpts], m[fitpts], 1)
            fit = c0 + c1 * a[fitpts]
            r2 = 1 - np.sum((m[fitpts] - fit) ** 2) / np.sum(
                (m[fitpts] - m[fitpts].mean()) ** 2)
            sig_e = float(s[i0])
            x0 = -c0 / c1            # zero of the linear fit (a != 0 points)
            want = np.sign(a - x0)
            se_a = float(np.median(s[fitpts]) / c1 / np.sqrt(e.shape[1]))
            # within 3 standard errors of x0 the sign of the mean is noise
            lo, hi = pull_in(a, np.sign(m) == want, x0, skip=3 * se_a)
            frac = np.mean(np.sign(e) == want[:, None], axis=1)
            skip90 = 2 * sig_e / max(c1, 1e-12)
            if skip90 >= 0.5 * np.max(np.abs(a)):     # noise zone too wide
                lo90, hi90 = float("nan"), float("nan")
            else:
                lo90, hi90 = pull_in(a, frac >= 0.9, x0, skip=skip90)
            bias = m / Kd0[ls] - a
            res[(snr, ls)] = dict(
                mean=m, std=s, Kd=c1, Kd_rel=c1 / Kd0[ls], r2=r2,
                sig_e=sig_e, sig_over_Kd=sig_e / c1,
                sig_over_Kd0=sig_e / Kd0[ls], x0=x0, lo=lo, hi=hi,
                lo90=lo90, hi90=hi90, bias=bias,
                maxbias=float(np.max(np.abs(bias[fitpts]))),
                bias0=float(bias[i0]),
                se=float(np.median(s[fitpts]) / Kd0[ls] / np.sqrt(e.shape[1])))
    return res, Kd0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--n-mc", type=int, default=64)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=None,
                    help="compute at most this many new values of a, then "
                         "exit (split the run into foreground chunks)")
    ap.add_argument("--near-zero", action="store_true",
                    help="print the noiseless bias mean(e)/K_d0 - a at the "
                         "values of a around 0 from the saved data (shows "
                         "that only a = 0 is offset), then exit")
    args = ap.parse_args()
    if args.near_zero:
        raw = ROOT / "results" / "char_raw"
        E = np.array([np.load(raw / f"a_{a:+.6f}.npz")["e"] for a in A_GRID])
        for j, ls in enumerate(LAGSETS):
            m = E[:, :, 0, j]
            kd0 = np.polyfit(A_GRID[np.isin(A_GRID, A_DENSE) & (A_GRID != 0)],
                             m.mean(1)[np.isin(A_GRID, A_DENSE)
                                       & (A_GRID != 0)], 1)[0]
            print(f"lag set {{{ls}}}, noiseless, K_d0 = {kd0:.1f}: "
                  "mean(e)/K_d0 - a  [std err]")
            for a in (-2e-4, -1e-4, -5e-5, 0.0, 5e-5, 1e-4, 2e-4, 5e-4,
                      1e-3):
                i = int(np.flatnonzero(np.isclose(A_GRID, a))[0])
                print(f"   a = {a:+.1e}: {m[i].mean() / kd0 - a:+.2e}  "
                      f"[{m[i].std() / kd0 / np.sqrt(m.shape[1]):.1e}]")
        return
    if args.replot:
        args.resume = True

    out = ROOT / "results" / ("quick" if args.quick else "")
    figd = out if args.quick else ROOT / "figures"
    raw = out / "char_raw"
    raw.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for f in raw.glob("*.npz"):
            f.unlink()
    n_mc = 3 if args.quick else args.n_mc
    a_list = (np.array([-1e-3, 0.0, 1e-3]) if args.quick else A_GRID)
    seeds = [1000 + s for s in range(n_mc)]
    t0 = time.time()

    if args.quick:            # the noise shortcut equals OCDMSignal exactly
        p, sig, y0, w = block(5e-4, 1000)
        p10 = base_params(5e-4, 10.0)
        ref = OCDMSignal(p10, n_symbols=N_C // p.N_sym + 2,
                         seed=1000).rx_nominal(N_C // p.N_sym * p.N_sym)
        mine = y0 + np.sqrt(p10.noise_power) * w
        d = np.max(np.abs(ref - mine))
        print(f"noise shortcut vs OCDMSignal.rx_nominal: max |diff| = {d:.1e}")
        assert d < 1e-12

    E = {}
    n_new = 0
    with Pool(args.procs) as pool:
        for a in a_list:
            f = raw / f"a_{a:+.6f}.npz"
            if f.exists() and np.load(f)["e"].shape[0] == n_mc:
                E[a] = np.load(f)["e"]
                continue
            if args.replot:
                raise SystemExit(f"--replot: missing {f.name}")
            if args.max_new is not None and n_new >= args.max_new:
                print(f"--max-new reached ({len(E)}/{len(a_list)} done); "
                      "rerun with --resume")
                return
            n_new += 1
            ts = time.time()
            e = np.array(pool.map(trial, [(a, s) for s in seeds]))
            np.savez(f, e=e, a=a, seeds=seeds, snrs=SNRS,
                     lagsets=list(LAGSETS))
            E[a] = e
            print(f"  a = {a:+.5f}: {n_mc} blocks [{time.time() - ts:.1f}s, "
                  f"total {time.time() - t0:.0f}s]", flush=True)
    if args.quick:
        print(f"quick done in {time.time() - t0:.0f}s "
              f"({len(a_list) * n_mc} (a, block) pairs)")
        return

    Earr = np.array([E[a] for a in A_GRID])        # [a, block, snr, lag]
    res, Kd0 = analyse(Earr)
    cx = complexity()

    # ---------------- tables ----------------
    rows = []
    for ls in LAGSETS:
        for snr in SNRS:
            r = res[(snr, ls)]
            rows.append(dict(lagset=ls, snr_db=snr, n_mc=n_mc,
                             bias_std_err=r["se"], Kd=r["Kd"],
                             Kd_rel=r["Kd_rel"], lin_R2=r["r2"],
                             sigma_e=r["sig_e"], sigma_e_over_Kd=r["sig_over_Kd"],
                             sigma_e_over_Kd0=r["sig_over_Kd0"],
                             zero_crossing=r["x0"],
                             pullin_lo=r["lo"], pullin_hi=r["hi"],
                             pullin90_lo=r["lo90"], pullin90_hi=r["hi90"],
                             max_abs_bias_prior=r["maxbias"],
                             bias_at_a0=r["bias0"]))
    res_dir = ROOT / "results"
    with open(res_dir / "characteristic.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def f(x, fmt=".2e"):
        return "–" if not np.isfinite(x) else format(x, fmt)

    def snr_s(s):
        return "∞" if np.isinf(s) else (f"{s:g} (WP)" if s == WP_DB
                                        else f"{s:g}")

    L = ["# Detector characteristic (open loop), Δ = {0} vs {0..15}", "",
         "Generated by `study_characteristic.py`. â = 0, true a swept "
         f"(41 points over |a| ≤ 1e-3, plus 1e-3 steps to ±2e-2); "
         f"N_mc = {n_mc} blocks per (a, SNR); δ = 0.25/N_seg, N_c = 24000, "
         "N_seg = 2048, 22-Jan preset (BPSK). SNR = σ_D²/P_n; WP = 4.0 dB, "
         "the BPSK 10 % SER working point (Part A).", "",
         "K_d: slope of mean e vs a, linear fit over |a| ≤ 1e-3 (the prior "
         "range). K_d/K_d0: relative to the noiseless value (the loop's "
         "design constant). σ_e: std of e at a = â. Bias: mean(e)/K_d0 − a "
         "(the error a loop calibrated with K_d0 would see); 'std err' is "
         "its standard error over the blocks. Pull-in: outermost a with the "
         "correct sign of e (mean / in ≥ 90 % of blocks, |a − x0| ≤ "
         "2σ_e/K_d exempt); grid 1e-3 beyond ±1e-3.", "",
         "**a = 0 is reported separately.** At a = 0 exactly the receiver "
         "samples synchronously (N_sym = 20 samples per symbol) and the "
         "detector shows an offset of ≈ +1.1e-5 (11 σ in a separate check "
         "with 256 fresh blocks), absent at every a ≠ 0 (a = 2.5e-4, 5e-4, "
         "1e-3: ≤ 1.5 σ). K_d, the zero crossing (zero of the linear fit) "
         "and max |bias| therefore use the 40 points a ≠ 0 of the prior "
         "range; the a = 0 bias has its own column.", ""]
    for ls in LAGSETS:
        L += [f"## Lag set Δ = {{{ls}}}  (noiseless K_d0 = {Kd0[ls]:.1f})", "",
              "| SNR (dB) | K_d | K_d/K_d0 | lin. R² | σ_e | σ_e/K_d | "
              "zero crossing | max \\|bias\\|, a ≠ 0 | std err | "
              "bias at a = 0 | pull-in (mean) | pull-in (90 %) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for snr in SNRS:
            r = res[(snr, ls)]
            L.append(f"| {snr_s(snr)} | {r['Kd']:.1f} | {r['Kd_rel']:.3f} | "
                     f"{r['r2']:.4f} | {r['sig_e']:.2e} | "
                     f"**{r['sig_over_Kd']:.2e}** | {f(r['x0'])} | "
                     f"{r['maxbias']:.2e} | {r['se']:.1e} | "
                     f"{r['bias0']:+.2e} | [{f(r['lo'])}, {f(r['hi'])}] | "
                     f"[{f(r['lo90'])}, {f(r['hi90'])}] |")
        L.append("")
    L += ["## Complexity per received sample (early-late, 2 cycle "
          "frequencies)", "",
          "Real multiplications (RM) per sample. Conventions as in the "
          "Step-3 count: recursive twiddles (4 RM per α, shared by all "
          "lags), segmented CAF by running sums (no RM), autocorrelated "
          "CAF R[n]R*[n−N_seg] (4 RM per α) on a fraction "
          f"(N_c − 2N_seg + 1)/N_c = {cx['frac']:.3f} of the samples.", "",
          "| lag set | RM / sample | note |", "|---|---|---|",
          f"| {{0}}, exact (|Y|² real) | **{cx['set0']:.1f}** | 2 (|Y|²) + "
          f"8 (twiddles) + 4 (real·complex modulation) + "
          f"{2 * 4 * cx['frac']:.1f} (autocorrelation) |",
          f"| {{0}}, upper bound | {cx['set0_upper']:.0f} | autocorrelation "
          "counted on every sample |",
          f"| {{0}}, counting |Y|² as complex | {cx['set0_complex']:.1f} | "
          "the Step-3 convention (\"at most 28\" without the fraction) |",
          f"| {{0..15}}, exact (lag 0 real) | **{cx['set_all']:.1f}** | |",
          f"| {{0..15}}, lag 0 counted complex | {cx['set_all_complex']:.1f} "
          "| reproduces the Step-3 figure of ~306 |", ""]
    (res_dir / "characteristic.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    # ---------------- figures ----------------
    sc = 1e3
    a = A_GRID
    dense = np.isin(a, A_DENSE)
    show = (float("inf"), 20.0, 10.0, WP_DB, 0.0, -5.0)
    cmap = plt.get_cmap("viridis")
    col = {s: cmap(k / (len(show) - 1)) for k, s in enumerate(show)}
    lbl = {s: ("noiseless" if np.isinf(s) else
               f"{s:g} dB (WP)" if s == WP_DB else f"{s:g} dB") for s in show}

    with plt.rc_context({"font.size": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        for j, ls in enumerate(LAGSETS):
            ax = axes[0, j]
            for s in show:
                r = res[(s, ls)]
                y = r["mean"][dense] / Kd0[ls] * sc
                ax.plot(a[dense] * sc, y, color=col[s], lw=1.6, label=lbl[s])
                if s in (20.0, WP_DB):
                    sd = r["std"][dense] / Kd0[ls] * sc
                    ax.fill_between(a[dense] * sc, y - sd, y + sd,
                                    color=col[s], alpha=0.15, lw=0)
            ax.plot([-1, 1], [-1, 1], color="0.2", ls="--", lw=1,
                    label="ideal y = x")
            ax.set_title(f"Δ = {{{ls}}}: mean(e)/K_d0 vs true a "
                         f"(â = 0; ±1 std at 20 dB, WP)", fontsize=11)
            ax.set_xlabel(r"true $a$ ($\times 10^{-3}$)")
            ax.set_ylabel(r"detector output / $K_{d0}$ ($\times 10^{-3}$)")
            ax.grid(True, color="0.9", lw=0.5)
            ax.legend(fontsize=8.5, loc="upper left")
            ax = axes[1, j]
            for s in show:
                r = res[(s, ls)]
                ax.plot(a * sc, r["mean"] / Kd0[ls] * sc, color=col[s],
                        lw=1.4, label=lbl[s])
            r20 = res[(20.0, ls)]
            ax.axvspan(r20["lo90"] * sc, r20["hi90"] * sc, color="0.9",
                       zorder=0, label="pull-in (90 %, 20 dB)")
            ax.axhline(0, color="0.3", lw=0.7)
            ax.set_title(f"Δ = {{{ls}}}: wide range", fontsize=11)
            ax.set_xlabel(r"true $a$ ($\times 10^{-3}$)")
            ax.set_ylabel(r"detector output / $K_{d0}$ ($\times 10^{-3}$)")
            ax.grid(True, color="0.9", lw=0.5)
            ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(figd / "fig_char_detector.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
        for j, ls in enumerate(LAGSETS):
            ax = axes[j]
            for s in show:
                ax.plot(a[dense] * sc, res[(s, ls)]["bias"][dense] * sc,
                        color=col[s], lw=1.5, label=lbl[s])
            ax.axhline(0, color="0.3", lw=0.7)
            ax.set_title(f"Δ = {{{ls}}}: bias mean(e)/K_d0 − (a − â)",
                         fontsize=11)
            ax.set_xlabel(r"true $a$ ($\times 10^{-3}$)")
            ax.grid(True, color="0.9", lw=0.5)
        axes[0].set_ylabel(r"bias ($\times 10^{-3}$)")
        axes[1].legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(figd / "fig_char_bias.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        xs = [s for s in SNRS if np.isfinite(s)]
        for ls, mk in zip(LAGSETS, ("o-", "s--")):
            axes[0].plot(xs, [res[(s, ls)]["Kd_rel"] for s in xs], mk,
                         label=f"Δ = {{{ls}}}")
            axes[1].semilogy(xs, [res[(s, ls)]["sig_over_Kd"] for s in xs],
                             mk, label=f"Δ = {{{ls}}}")
        for ax in axes:
            ax.axvline(WP_DB, color="0.5", ls=":", lw=1)
            ax.set_xlabel(r"$\sigma_D^2/P_n$ (dB)")
            ax.grid(True, which="both", color="0.9", lw=0.5)
            ax.legend()
        axes[0].set_ylabel(r"$K_d / K_{d0}$")
        axes[0].set_title("detector gain vs SNR (dotted: working point)",
                          fontsize=11)
        axes[1].set_ylabel(r"$\sigma_e / K_d$  (single-block error)")
        axes[1].set_title("implied single-block Doppler error", fontsize=11)
        fig.tight_layout()
        fig.savefig(figd / "fig_char_kd_vs_snr.png", dpi=150)
        plt.close(fig)
    print(f"total runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
