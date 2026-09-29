"""
Reproduce the figures of the previous results documents.

  fig_31jan_noiseless.png     31-Jan doc, Exp. 1: windowed AF, noiseless,
                              a_sr = a, lags inside the main lobe, with the
                              curve-fit ('chosen Doppler') curve
  fig_22jan_setting{1..5}.png 22-Jan doc, Settings 1-5: windowed AF at
                              sigma_D^2/P_n = 10 and 20 dB
  fig_22jan_dense_window.png  22-Jan doc, p. 3: N_lag = 2^8 lags placed
                              only inside the known window, 20 dB
  fig_curvefit_cost.png       curve-fitting MSE vs candidate Doppler
                              (31-Jan noiseless, 22-Jan S1 at 20 / 10 dB)
  fig_cbsfs_cost.png          [CBSFS] Fig. 5 analogue: cost function (29)
                              for a = 0 vs the true Doppler, plus a zoom on
                              the harmonic used for estimation

Each AF figure is one Monte-Carlo realisation, as in the documents. The
per-figure numbers (a_sr, t_sel, window, curve-fit estimates) are written to
figures/figures_log.txt; Monte-Carlo statistics are in mc_curvefit.py.

Usage (from this directory):
    python make_figures.py                 # all figures, a_sr from CB-SFS
    python make_figures.py --a-sr genie    # a_sr = true a (faster)
    python make_figures.py --only 31jan cost
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocdm_doppler import (SystemParams, cbsfs, preset_noiseless_31jan,
                          preset_noisy_22jan)
from ocdm_doppler import af_estimation as afe
from ocdm_doppler import curve_fit as cf

OUT = Path(__file__).parent / "figures"

# Line styles of the MATLAB figures in the results documents.
STYLE_EMP = dict(color="red", ls="-", marker="o", ms=4, lw=1.4,
                 label="Empirical noisy AF magnitude")
STYLE_TRUE = dict(color="blue", ls="--", marker=".", ms=6, lw=1.0,
                  label="Analytical noiseless AF magnitude with true Doppler")
STYLE_CHOSEN = dict(color="green", ls="-.", marker="+", ms=5, lw=1.0,
                    label="Analytical noiseless AF magnitude with chosen Doppler")


def _style_axes(ax, title):
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel("Lag (s)")
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)


def plot_windowed_af(ax, w: afe.WindowedAF, params: SystemParams, title,
                     fit: cf.CurveFitResult | None = None):
    """Empirical AF, analytical AF with the true Doppler and, given a
    curve-fit result, the analytical AF with the chosen Doppler (drawn with
    the |A0|^2 and tau_p0 that the fit used)."""
    x = w.lags_win
    ax.plot(x, w.af_win, **STYLE_EMP)
    ax.plot(x, w.analytical(params.a_true_eff, params), **STYLE_TRUE)
    if fit is not None:
        ax.plot(x, w.analytical(fit.a_hat, params, fit.A0_sq, fit.tau_p0),
                **STYLE_CHOSEN)
    ax.set_xlim(x[0], x[-1])
    _style_axes(ax, title)


# ---------------------------------------------------------------------------
# Initial (symbol-rate) Doppler estimate
# ---------------------------------------------------------------------------
N_C_CBSFS = 24000      # samples observed by CB-SFS (not stated in the docs)
N_SEG_CBSFS = 2048     # see README: N_seg dominates CB-SFS accuracy


def initial_estimate(sig, params: SystemParams, mode: str) -> float:
    """CB-SFS a_sr with the settings of mc_curvefit.py / mc_cbsfs_22jan.py:
    harmonic 1, +/-0.05 alpha_1 window, 128 points, parabolic refinement."""
    if mode == "genie":
        return params.a_true_eff
    M = N_C_CBSFS // params.N_sym
    y = sig.rx_nominal(M * params.N_sym)
    ak, _, _ = cbsfs.locate_harmonic(y, params, 1, n_alpha=128,
                                     half_width_factor=0.05,
                                     N_seg=N_SEG_CBSFS)
    return ak / params.alpha_1_syn - 1.0


LOG: list[str] = []


def log(msg: str):
    """Print and keep for figures/figures_log.txt."""
    print(msg, flush=True)
    LOG.append(msg)


def _cbsfs_symbols(params):
    return N_C_CBSFS // params.N_sym + 2


# ---------------------------------------------------------------------------
# 31-Jan document
# ---------------------------------------------------------------------------
def windowed_af_31jan(p: SystemParams, seed: int) -> afe.WindowedAF:
    """31-Jan Exp. 1: a_sr = a (noiseless), t'_2 = T_sym / (2 (1 + a)),
    lags inside (-T_is / (N_is (1 + a)), T_is / (N_is (1 + a)))."""
    a = p.a_true_eff
    sig, t0 = afe.make_signal(p, seed=seed)
    return afe.estimate_windowed_af(
        sig, t0, a_sr=a,
        t_candidates=(0.0, p.T_sym / (2.0 * (1.0 + a))),
        lag_half_width=p.T_is / (p.N_is * (1.0 + a)))


def fit_31jan(w: afe.WindowedAF, p: SystemParams) -> cf.CurveFitResult:
    # The window is the whole lag set here, so tau_p0 cannot be read off its
    # edge; Gamma = 1 on all these lags, so the true tau_p0 is used.
    return cf.fit_doppler(w, p, tau_p0=p.tau_p0)


def fig_31jan(seed: int = 0):
    p = preset_noiseless_31jan()
    a = p.a_true_eff
    w = windowed_af_31jan(p, seed)
    fit = fit_31jan(w, p)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plot_windowed_af(ax, w, p,
                     "Windowed autocorrelation function for SNR = Inf (dB)",
                     fit)
    fig.tight_layout()
    fig.savefig(OUT / "fig_31jan_noiseless.png", dpi=150)
    plt.close(fig)
    log(f"  31-Jan: t_sel={w.t_sel:.4g}s  "
        f"max|emp-ana|/peak="
        f"{np.max(np.abs(w.af_win - w.analytical(a, p))) / w.af_win.max():.3%}"
        f"  a_hat_mag={fit.a_hat:+.4e} (true {a:+.1e})  "
        f"|A0|^2_hat={fit.A0_sq:.4f} (true {abs(p.A_p0) ** 2:.4f})")


def _log_fit(label: str, w: afe.WindowedAF, p: SystemParams):
    """Log the curve-fit result for this realisation (not plotted: the
    22-Jan figures show only the empirical and true-Doppler curves)."""
    fit = cf.fit_doppler(w, p)
    log(f"  {label}: a_sr={w.a_sr:+.4e} (true {p.a_true_eff:+.1e})  "
        f"t_sel={w.t_sel:.4g}s  "
        f"window=[{w.lags_win[0]:.3e}, {w.lags_win[-1]:.3e}]s  "
        f"a_hat_mag={fit.a_hat:+.4e}  |A0|^2_hat={fit.A0_sq:.4f} "
        f"(true {abs(p.A_p0) ** 2:.4f})  tau_p0_hat={fit.tau_p0:.4e} "
        f"(true {p.tau_p0:.1e})")


# ---------------------------------------------------------------------------
# 22-Jan document
# ---------------------------------------------------------------------------
SETTINGS_22JAN = {
    1: dict(rho=2 ** 5),
    2: dict(rho=2 ** 6),
    3: dict(rho=2 ** 5, a_max=1e-2),
    4: dict(rho=2 ** 6, a_max=1e-2),
    5: dict(rho=2 ** 6, N_is=64),
}


def _run_22jan(p: SystemParams, a_sr_mode: str, seed: int,
               lags=None) -> afe.WindowedAF:
    sig, t0 = afe.make_signal(p, n_extra_symbols=_cbsfs_symbols(p), seed=seed)
    a_sr = initial_estimate(sig, p, a_sr_mode)
    return afe.estimate_windowed_af(
        sig, t0, a_sr,
        t_candidates=(0.0, p.T_sym / (3.0 * (1.0 + p.a_max))),
        lags=lags)


def fig_22jan_setting(k: int, a_sr_mode: str, seed: int = 0):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4))
    for ax, snr in zip(axes, (10.0, 20.0)):
        p = preset_noisy_22jan(snr).with_(**SETTINGS_22JAN[k])
        w = _run_22jan(p, a_sr_mode, seed)
        plot_windowed_af(ax, w, p,
                         f"Windowed autocorrelation function for "
                         f"$\\sigma_D^2/P_n$ = {snr:g} (dB)")
        _log_fit(f"22-Jan S{k} {snr:g} dB", w, p)
    fig.suptitle(f"22-Jan setting {k}", fontsize=9, color="0.4")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_22jan_setting{k}.png", dpi=150)
    plt.close(fig)


def fig_22jan_dense(a_sr_mode: str, seed: int = 0, snr: float = 20.0):
    """N_lag = 2^8 lags uniformly inside the known (true) window."""
    p = preset_noisy_22jan(snr).with_(N_lag=2 ** 8)
    t_sel = p.T_sym / (3.0 * (1.0 + p.a_max))
    sig, t0 = afe.make_signal(p, n_extra_symbols=_cbsfs_symbols(p), seed=seed)
    lo, hi = afe.true_window(t0 + t_sel, p)
    lags = np.linspace(lo, hi, p.N_lag + 1)[1:]          # (lo, hi]
    a_sr = initial_estimate(sig, p, a_sr_mode)
    w = afe.estimate_windowed_af(sig, t0, a_sr, t_candidates=(0.0, t_sel),
                                 lags=lags)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plot_windowed_af(ax, w, p,
                     f"Windowed autocorrelation function for "
                     f"$\\sigma_D^2/P_n$ = {snr:g} (dB)")
    fig.suptitle("22-Jan: $N_{lag}=2^8$ inside the known window",
                 fontsize=9, color="0.4")
    fig.tight_layout()
    fig.savefig(OUT / "fig_22jan_dense_window.png", dpi=150)
    plt.close(fig)
    _log_fit("22-Jan dense window 20 dB", w, p)


# ---------------------------------------------------------------------------
# Curve-fitting cost surface
# ---------------------------------------------------------------------------
def fig_curvefit_cost(a_sr_mode: str, seed: int = 0):
    """MSE cost vs candidate Doppler: 31-Jan noiseless and 22-Jan S1."""
    cases = []
    p = preset_noiseless_31jan()
    w = windowed_af_31jan(p, seed)
    cases.append(("31-Jan, noiseless ($\\hat a_{sr}=a$)", p, w, fit_31jan(w, p)))
    for snr in (20.0, 10.0):
        p = preset_noisy_22jan(snr)
        w = _run_22jan(p, a_sr_mode, seed)
        cases.append((f"22-Jan setting 1, $\\sigma_D^2/P_n$ = {snr:g} dB",
                      p, w, cf.fit_doppler(w, p)))

    fig, axes = plt.subplots(len(cases), 1, figsize=(7.5, 7.8), sharex=True)
    for ax, (title, p, w, fit) in zip(axes, cases):
        sc = 1e3
        ax.plot(fit.a_grid[::16] * sc, fit.cost[::16], color="black", lw=1.0)
        ax.axvline(p.a_true_eff * sc, color="blue", ls="--", lw=1.0,
                   label=f"true $a$ = {p.a_true_eff:.1e}")
        ax.axvline(fit.a_hat * sc, color="green", ls="-.", lw=1.0,
                   label=f"$\\hat a_{{mag}}$ = {fit.a_hat:.2e}")
        c_true = cf.cost_curve(w, p, np.array([p.a_true_eff]),
                               fit.A0_sq, fit.tau_p0)[0]
        ax.set_title(f"{title}   (cost range {fit.cost.min():.2e} – "
                     f"{fit.cost.max():.2e}; at true $a$: {c_true:.2e})",
                     fontsize=9)
        ax.set_ylabel("MSE")
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(loc="upper right", fontsize=8)
        log(f"  cost [{title}]: a_sr={w.a_sr:+.4e}  a_hat={fit.a_hat:+.4e}  "
            f"|A0|^2={fit.A0_sq:.4f}  tau_p0={fit.tau_p0:.4e}  "
            f"cost min/true/max={fit.cost.min():.3e}/{c_true:.3e}/"
            f"{fit.cost.max():.3e}")
    axes[-1].set_xlabel(r"Candidate Doppler $a$ ($\times 10^{-3}$)")
    fig.suptitle("Curve-fitting cost: MSE(empirical AF, analytical AF) "
                 "vs candidate $a$", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_curvefit_cost.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# [CBSFS] Fig. 5 analogue
# ---------------------------------------------------------------------------
def fig_cbsfs_cost(seed: int = 0, snr: float = 10.0, N_seg: int = 1024,
                   k_zoom: int = 1):
    p = preset_noisy_22jan(snr).with_(N_seg=N_seg)
    a = p.a_true_eff
    M = N_C_CBSFS // p.N_sym
    from ocdm_doppler import OCDMSignal
    sig = OCDMSignal(p, n_symbols=M + 4, seed=seed)
    y_sync = sig.rx_nominal(M * p.N_sym, a=0.0)
    y_dopp = sig.rx_nominal(M * p.N_sym, a=a)

    alphas = np.linspace(0.02 * p.alpha_1_syn, p.alpha_NT * 1.05, 1500)
    c_sync = cbsfs.cost_function(y_sync, alphas, p)
    c_dopp = cbsfs.cost_function(y_dopp, alphas, p)

    ac = k_zoom * p.alpha_1_syn
    zoom = np.linspace(ac - 2.0 / N_seg, ac + 2.0 / N_seg, 401)
    z_sync = cbsfs.cost_function(y_sync, zoom, p)
    z_dopp = cbsfs.cost_function(y_dopp, zoom, p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax1.semilogy(alphas, c_sync, color="black", lw=1.4,
                 label="Synchronized ($a=0$)")
    ax1.semilogy(alphas, c_dopp, color="0.55", lw=0.9,
                 label=f"Received, $a={a:g}$")
    ax1.set_xlabel(r"Cycle frequency ($\alpha$)")
    ax1.set_ylabel("Cost function")
    ax1.set_title(f"CB-SFS cost function (29), $\\sigma_D^2/P_n$ = {snr:g} dB",
                  fontsize=10)
    ax1.grid(True, which="both", color="0.9", lw=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    sc = 1e3
    ax2.plot((zoom - ac) * sc, z_sync, color="black", lw=1.4,
             label="Synchronized ($a=0$)")
    ax2.plot((zoom - ac) * sc, z_dopp, color="0.55", lw=0.9,
             label=f"Received, $a={a:g}$")
    ax2.axvline(0.0, color="black", ls=":", lw=0.8)
    ax2.axvline(k_zoom * a * p.alpha_1_syn * sc, color="0.55", ls=":", lw=0.8)
    ax2.set_xlabel(rf"$\alpha - {k_zoom}/N_{{sym}}$  ($\times 10^{{-3}}$)")
    ax2.set_title(f"Harmonic {k_zoom}: shift $k a \\alpha_1$ = "
                  f"{k_zoom * a * p.alpha_1_syn:.1e},  peak width "
                  f"1/$N_{{seg}}$ = {1 / N_seg:.1e}", fontsize=9)
    ax2.grid(True, color="0.9", lw=0.5)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cbsfs_cost.png", dpi=150)
    plt.close(fig)
    log(f"  CB-SFS cost: M={M}  N_seg={N_seg}  SNR={snr:g} dB")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a-sr", choices=("cbsfs", "genie"), default="cbsfs",
                    help="initial Doppler estimate for the 22-Jan figures")
    ap.add_argument("--only", nargs="*",
                    choices=("31jan", "22jan", "dense", "cost", "cbsfs"),
                    default=("31jan", "22jan", "dense", "cost", "cbsfs"))
    ap.add_argument("--settings", nargs="*", type=int,
                    default=sorted(SETTINGS_22JAN))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    t = time.time()
    log(f"make_figures.py  seed={args.seed}  a_sr={args.a_sr}  "
        f"figures={' '.join(args.only)}  "
        f"(one realisation per figure; CB-SFS N_c={N_C_CBSFS}, "
        f"N_seg={N_SEG_CBSFS}, harmonic 1)")
    if "31jan" in args.only:
        fig_31jan(args.seed)
    if "22jan" in args.only:
        for k in args.settings:
            fig_22jan_setting(k, args.a_sr, args.seed)
    if "dense" in args.only:
        fig_22jan_dense(args.a_sr, args.seed)
    if "cost" in args.only:
        fig_curvefit_cost(args.a_sr, args.seed)
    if "cbsfs" in args.only:
        fig_cbsfs_cost(args.seed)
    print(f"done in {time.time() - t:.1f}s -> {OUT}")
    (OUT / "figures_log.txt").write_text("\n".join(LOG) + "\n",
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
