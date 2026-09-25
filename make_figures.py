"""
Reproduce the figures of the previous results documents.

  fig_31jan_noiseless.png     31-Jan doc, Exp. 1: windowed AF, noiseless,
                              a_sr = a, lags inside the main lobe
  fig_22jan_setting{1..5}.png 22-Jan doc, Settings 1-5: windowed AF at
                              sigma_D^2/P_n = 10 and 20 dB
  fig_22jan_dense_window.png  22-Jan doc, p. 3: N_lag = 2^8 lags placed
                              only inside the known window, 20 dB
  fig_cbsfs_cost.png          [CBSFS] Fig. 5 analogue: cost function (29)
                              for a = 0 vs the true Doppler, plus a zoom on
                              the harmonic used for estimation

Each AF figure is one Monte-Carlo realisation, as in the documents.

Usage (from this directory):
    python make_figures.py                 # all figures, a_sr from CB-SFS
    python make_figures.py --a-sr genie    # a_sr = true a (faster)
    python make_figures.py --only 31jan cbsfs
    python make_figures.py --a-chosen 1.2e-3   # adds the 'chosen Doppler'
                                               # curve to the 31-Jan figure
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
                     a_chosen: float | None = None):
    x = w.lags_win
    ax.plot(x, w.af_win, **STYLE_EMP)
    ax.plot(x, w.analytical(params.a_true_eff, params), **STYLE_TRUE)
    if a_chosen is not None:
        ax.plot(x, w.analytical(a_chosen, params), **STYLE_CHOSEN)
    ax.set_xlim(x[0], x[-1])
    _style_axes(ax, title)


# ---------------------------------------------------------------------------
# Initial (symbol-rate) Doppler estimate
# ---------------------------------------------------------------------------
N_C_CBSFS = 24000      # samples observed by CB-SFS (not stated in the docs)
N_SEG_CBSFS = 2048     # see README: N_seg dominates CB-SFS accuracy


def initial_estimate(sig, params: SystemParams, mode: str) -> float:
    if mode == "genie":
        return params.a_true_eff
    M = N_C_CBSFS // params.N_sym
    y = sig.rx_nominal(M * params.N_sym)
    res = cbsfs.estimate_doppler_harmonic(y, params, N_last=1,
                                          N_seg=N_SEG_CBSFS)
    return res.a_hat


def _cbsfs_symbols(params):
    return N_C_CBSFS // params.N_sym + 2


# ---------------------------------------------------------------------------
# 31-Jan document
# ---------------------------------------------------------------------------
def fig_31jan(a_chosen: float | None, seed: int = 0):
    p = preset_noiseless_31jan()
    a = p.a_true_eff
    sig, t0 = afe.make_signal(p, seed=seed)
    # a_sr = a (noiseless experiment); t'_2 = T_sym / (2 (1 + a));
    # lags inside (-T_is / (N_is (1 + a)), T_is / (N_is (1 + a))).
    w = afe.estimate_windowed_af(
        sig, t0, a_sr=a,
        t_candidates=(0.0, p.T_sym / (2.0 * (1.0 + a))),
        lag_half_width=p.T_is / (p.N_is * (1.0 + a)))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plot_windowed_af(ax, w, p,
                     "Windowed autocorrelation function for SNR = Inf (dB)",
                     a_chosen)
    fig.tight_layout()
    fig.savefig(OUT / "fig_31jan_noiseless.png", dpi=150)
    plt.close(fig)
    print(f"  31-Jan: t_sel={w.t_sel:.4g}s  sigma2={w.sigma2}  "
          f"max|emp-ana|/peak="
          f"{np.max(np.abs(w.af_win - w.analytical(a, p))) / w.af_win.max():.3%}")


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
        print(f"  22-Jan S{k} {snr:g} dB: a_sr={w.a_sr:+.4e} "
              f"(true {p.a_true_eff:+.1e})  t_sel={w.t_sel:.4g}s  "
              f"window=[{w.lags_win[0]:.3e}, {w.lags_win[-1]:.3e}]s")
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
    print(f"  22-Jan dense: a_sr={w.a_sr:+.4e}  t_sel={w.t_sel:.4g}s")


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
    print(f"  CB-SFS cost: M={M}  N_seg={N_seg}  SNR={snr:g} dB")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a-sr", choices=("cbsfs", "genie"), default="cbsfs",
                    help="initial Doppler estimate for the 22-Jan figures")
    ap.add_argument("--a-chosen", type=float, default=None,
                    help="curve-fit Doppler to draw on the 31-Jan figure")
    ap.add_argument("--only", nargs="*",
                    choices=("31jan", "22jan", "dense", "cbsfs"),
                    default=("31jan", "22jan", "dense", "cbsfs"))
    ap.add_argument("--settings", nargs="*", type=int,
                    default=sorted(SETTINGS_22JAN))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    t = time.time()
    if "31jan" in args.only:
        fig_31jan(args.a_chosen, args.seed)
    if "22jan" in args.only:
        for k in args.settings:
            fig_22jan_setting(k, args.a_sr, args.seed)
    if "dense" in args.only:
        fig_22jan_dense(args.a_sr, args.seed)
    if "cbsfs" in args.only:
        fig_cbsfs_cost(args.seed)
    print(f"done in {time.time() - t:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
