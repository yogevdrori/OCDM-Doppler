"""
Step 4, Part A: SNR definitions and the 10 % symbol-error working point.

The SNR used so far is sigma_D^2 / P_n (results documents). From
signal_model.rx and _BlockNoise:

    Y(t) = h e^{j 2 pi Delta_f t} X(t) + n(t),   h = A_p0 e^{-j 2 pi f_Tx tau_p0} / 2
    X = Phi D on the active part (Phi unitary)  =>  E|X|^2 = sigma_D^2
    n CSCG, E|n|^2 = P_n per sample

so, with |h|^2 = |A0|^2 / 4 = 0.324 (-4.89 dB):

    per received sample, active part      |A0|^2/4 * sigma_D^2/P_n
    per received sample, symbol + GI avg  |A0|^2/4 * N_is/N_sym * sigma_D^2/P_n
    per data symbol after DFnT (Es/N0)    |A0|^2/4 * sigma_D^2/P_n
      (Phi^H is unitary: the noise stays white with variance P_n)
    Eb/N0 = Es/N0 (BPSK), Es/N0 / 2 (QPSK)

SER: BPSK Q(sqrt(2 Es/N0)); QPSK 1 - (1 - Q(sqrt(Es/N0)))^2.

Simulation: a = 0, perfect timing (sample symbol m at tau_p0 + m T_sym +
n T_samp), CFO removed with its known value, DFnT (Phi^H), divide by h, hard
decisions; SER vs sigma_D^2/P_n for BPSK (22-Jan preset) and QPSK.

Usage:  python study_snr_definitions.py [--quick]
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

from ocdm_doppler import OCDMSignal, chirp_matrix, preset_noisy_22jan

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"


def qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def ser_theory(mod, esn0):
    if mod == "bpsk":
        return qfunc(math.sqrt(2.0 * esn0))
    q = qfunc(math.sqrt(esn0))
    return 1.0 - (1.0 - q) ** 2


def esn0_for_ser(mod, target=0.1):
    """Es/N0 (linear) with SER = target, by bisection in dB."""
    lo, hi = -20.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ser_theory(mod, 10 ** (mid / 10)) > target:
            lo = mid
        else:
            hi = mid
    return 10 ** (0.5 * (lo + hi) / 10)


def simulate_ser(mod, snr_db, n_sym, seed):
    """Measured SER, plus the measured signal and noise powers."""
    p = preset_noisy_22jan(snr_db).with_(modulation=mod, a_true=0.0)
    sig = OCDMSignal(p, n_symbols=n_sym, seed=seed)
    m = np.arange(n_sym)[:, None]
    n = np.arange(p.N_is)[None, :]
    # a = 0. The tiny offset keeps n = 0 off the symbol boundary, where
    # floating-point rounding in floor() could assign the sample to the
    # previous symbol's guard interval (phase error ~ 6e-6 rad).
    t = (p.tau_p0 + m * p.T_sym + (n + 1e-6) * p.T_samp_syn).ravel()
    derot = np.exp(-2j * np.pi * p.Delta_f_of_a(0.0) * t)
    y = (sig.rx(t, a=0.0) * derot).reshape(n_sym, p.N_is)
    y0 = (sig.rx(t, a=0.0, noiseless=True) * derot).reshape(n_sym, p.N_is)
    h = p.A_p0 * np.exp(-2j * np.pi * p.f_Tx * p.tau_p0) / 2.0
    Phi = chirp_matrix(p.N_is)
    d_hat = (y @ Phi.conj()) / h            # row m: Phi^H y_m (Phi symmetric)
    d0 = (y0 @ Phi.conj()) / h
    D = sig.D
    if mod == "bpsk":
        dec = np.where(d_hat.real >= 0, 1.0, -1.0)
        err = dec != D.real
    else:
        dec = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.floor(
            np.mod(np.angle(d_hat) - 0.0, 2 * np.pi) / (np.pi / 2))))
        err = np.abs(dec - D) > 1e-6
    return dict(ser=float(err.mean()),
                noiseless_err=float(np.max(np.abs(d0 - D))),
                p_sig=float(np.mean(np.abs(y0) ** 2)),
                p_noise=float(np.mean(np.abs(y - y0) ** 2)))


def crossing(x, y, target):
    """x where y (decreasing) crosses target, linear in log10(y)."""
    for i in range(len(x) - 1):
        if y[i] >= target > y[i + 1] and y[i + 1] > 0:
            ly0, ly1 = math.log10(y[i]), math.log10(y[i + 1])
            return x[i] + (x[i + 1] - x[i]) * (ly0 - math.log10(target)) / (
                ly0 - ly1)
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    n_sym = 200 if args.quick else 4000           # x 16 data symbols
    grid = np.arange(-2.0, 14.01, 0.5)

    p = preset_noisy_22jan(10.0)
    g = abs(p.A_p0) ** 2 / 4.0
    off_sym = 10 * math.log10(g)
    off_avg = 10 * math.log10(g * p.N_is / p.N_sym)

    rows, meas = [], {}
    for mod in ("bpsk", "qpsk"):
        ser = []
        for k, s in enumerate(grid):
            r = simulate_ser(mod, float(s), n_sym, seed=1000 + k)
            esn0 = g * 10 ** (s / 10)
            rows.append(dict(modulation=mod, snr_sigmaD2_Pn_db=s,
                             EsN0_db=10 * math.log10(esn0),
                             ser_sim=r["ser"],
                             ser_theory=ser_theory(mod, esn0),
                             n_data_symbols=n_sym * p.N_is,
                             noiseless_max_err=r["noiseless_err"],
                             p_sig_active=r["p_sig"],
                             p_noise=r["p_noise"]))
            ser.append(r["ser"])
        meas[mod] = crossing(grid, ser, 0.1)

    th = {mod: esn0_for_ser(mod) for mod in ("bpsk", "qpsk")}
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    with open(RES / "snr_definitions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    sigD2 = p.sigma_D2
    ps = np.array([r["p_sig_active"] for r in rows])
    pn = np.array([r["p_noise"] / (sigD2 / 10 ** (r["snr_sigmaD2_Pn_db"] / 10))
                   for r in rows])
    max_noiseless = max(r["noiseless_max_err"] for r in rows)

    def wp(mod):
        es_db = 10 * math.log10(th[mod])
        return es_db, es_db - off_sym

    L = ["# SNR definitions and the 10 % symbol-error working point", "",
         "Generated by `study_snr_definitions.py`. All in terms of "
         "σ_D²/P_n (the SNR used so far), single path, 22-Jan channel "
         f"(|A0|² = {abs(p.A_p0) ** 2:.4f}).", "",
         "## Definitions", "",
         "| SNR definition | in terms of σ_D²/P_n | offset (dB) |",
         "|---|---|---|",
         f"| per received sample, active part | |A0|²/4 · σ_D²/P_n | "
         f"{off_sym:+.2f} |",
         f"| per received sample, average over symbol + GI | "
         f"|A0|²/4 · N_is/N_sym · σ_D²/P_n | {off_avg:+.2f} |",
         f"| per data symbol after DFnT, Es/N0 | |A0|²/4 · σ_D²/P_n | "
         f"{off_sym:+.2f} |",
         f"| Eb/N0, BPSK | = Es/N0 | {off_sym:+.2f} |",
         f"| Eb/N0, QPSK | Es/N0 / 2 | {off_sym - 3.01:+.2f} |", "",
         "Derivation: `signal_model.rx` gives Y = h e^{j2πΔf t} X + n with "
         "h = A_p0 e^{−j2πf_Tx τ_p0}/2 (the e^{jπ/4} of Δφ is the one in X); "
         "X = ΦD with Φ unitary, so E|X|² = σ_D² on the active part and the "
         "DFnT (Φ^H) leaves the noise white with variance P_n "
         "(`_BlockNoise`: E|n|² = P_n).", "",
         "Checks from the simulation: noiseless demodulation reproduces D "
         f"(max |D̂ − D| = {max_noiseless:.1e}); measured active-part signal "
         f"power {ps.mean():.4f} σ_D² (expected {g:.4f}); measured noise "
         f"power / P_n = {pn.mean():.4f} (expected 1).", "",
         "## 10 % symbol-error working point", "",
         "| modulation | Es/N0 (dB), theory | σ_D²/P_n (dB), theory | "
         "σ_D²/P_n (dB), simulated |",
         "|---|---|---|---|"]
    for mod in ("bpsk", "qpsk"):
        es_db, s_db = wp(mod)
        L.append(f"| {mod.upper()} | {es_db:+.2f} | {s_db:.2f} | "
                 f"{meas[mod]:.2f} |")
    L += ["",
          f"Simulation: a = 0, perfect timing, CFO removed, "
          f"{n_sym * p.N_is} data symbols per SNR point, SNR step 0.5 dB, "
          "10 % point by interpolation in log(SER). "
          "**Working point used in Part B–D (22-Jan preset is BPSK): "
          f"σ_D²/P_n = {wp('bpsk')[1]:.1f} dB.**", ""]
    (RES / "snr_definitions.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for mod, c in (("bpsk", "#1f77b4"), ("qpsk", "#ff7f0e")):
        sel = [r for r in rows if r["modulation"] == mod]
        x = [r["snr_sigmaD2_Pn_db"] for r in sel]
        ax.semilogy(x, [max(r["ser_sim"], 1e-6) for r in sel], "o", ms=4,
                    color=c, label=f"{mod.upper()} simulated")
        ax.semilogy(x, [r["ser_theory"] for r in sel], "-", color=c, lw=1.2,
                    label=f"{mod.upper()} theory")
        ax.axvline(wp(mod)[1], color=c, ls=":", lw=1)
    ax.axhline(0.1, color="0.4", lw=0.8, ls="--")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(r"$\sigma_D^2/P_n$ (dB)")
    ax.set_ylabel("symbol error rate")
    ax.set_title("SER after DFnT (a = 0, perfect sync); 10 % points dotted",
                 fontsize=10)
    ax.grid(True, which="both", color="0.9", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ser_working_point.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
