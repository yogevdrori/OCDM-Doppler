"""
Step 5: tracking a time-varying Doppler with the first-order loop
(ocdm_doppler/loop_tv.run_loop_profile).

Part B  ramp, beta = r g sigma_ss per block (r = 0.1, 1, 10; g = 0.01,
        sigma_ss = 3.25e-6 at the WP), poles {0.9, 0.95, 0.98, 0.99, 0.995}
        (+ 0.998 for r = 0.1, 1), noiseless and 4 dB, UP and DOWN ramps over
        the same a interval: (eps_up - eps_down)/2 is the lag free of any
        a-dependent detector bias, (eps_up + eps_down)/2 the bias.
Part C  sinusoid (frequency response of the loop)        -- see run_part_c
Part D  fly-by                                            -- see run_part_d

Settings: 22-Jan preset (BPSK), Delta = {0}, N_c = 24000, N_seg = 2048, k = 1,
delta = 0.25/N_seg, K_d0 = 175.5, mu = (1 - p)/K_d0, seeds 1000 + index.
All poles and both SNRs of one case run in lockstep on the same blocks.

Long runs: every realisation is advanced in segments and checkpointed
(results/track_raw/), so a call can stop at any time (--budget seconds) and
--resume continues exactly (verified in verify_time_varying.py, check 3).
--replot redoes tables and figures from the saved trajectories.

Usage:
    python study_tracking.py --part B --quick
    python study_tracking.py --part B [--resume] [--budget 480] [--replot]
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
from ocdm_doppler import loop_tv as lt
from ocdm_doppler.doppler_profile import (RampDoppler, SinusoidalDoppler,
                                          FlyByDoppler)

ROOT = Path(__file__).parent
N_C, N_SEG = 24000, 2048
KD0 = 175.5
T_BLK = N_C / 4e6
WP_DB = 4.0
SNRS = (float("inf"), WP_DB)
# sigma_e / K_d from Step 4 Part B (Delta = {0}): noiseless and WP
S_EK = {float("inf"): 1.60e-5, WP_DB: 4.59e-5}
SIGMA_SS = lt.sigma_ss(S_EK[WP_DB], 0.01)          # 3.25e-6
SEG = 1500                                         # blocks per segment


def base():
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,))


def n95(p):
    return math.log(0.05) / math.log(p)


# ---------------------------------------------------------------------------
# Generic checkpointed runner
# ---------------------------------------------------------------------------
def _segment(args):
    """Advance one realisation of one case by n blocks."""
    profile, poles, seed, start, n, a_last = args
    mus = [(1 - p) / KD0 for p in poles]
    r = lt.run_loop_profile(base(), profile, mu=mus, n_blocks=n, snrs=SNRS,
                            a_hat0=a_last, N_c=N_C, N_seg=N_SEG, seed=seed,
                            start_block=start)
    return r.a_hat[..., 1:]


def run_cases(cases, raw: Path, budget: float, procs: int, t0: float):
    """cases: name -> dict(profile, poles, n_blocks, R, a_hat0).
    Returns True when every case is complete."""
    raw.mkdir(parents=True, exist_ok=True)
    last = 0.0                       # duration of the last segment
    with Pool(procs) as pool:
        for name, c in cases.items():
            f = raw / f"{name}.npz"
            if f.exists():
                st = np.load(f)
                traj = [st[f"r{r}"] for r in range(c["R"])]
            else:
                a0 = np.full((len(c["poles"]), len(SNRS), 1), c["a_hat0"])
                traj = [a0.copy() for _ in range(c["R"])]
            while traj[0].shape[-1] - 1 < c["n_blocks"]:
                # start a segment only if one more (as long as the last)
                # still fits in the budget
                if time.time() - t0 + 1.1 * last > budget:
                    print(f"  budget reached; rerun with --resume "
                          f"[{time.time() - t0:.0f}s]", flush=True)
                    return False
                done = traj[0].shape[-1] - 1
                n = min(SEG, c["n_blocks"] - done)
                ts = time.time()
                segs = pool.map(_segment, [
                    (c["profile"], c["poles"], 1000 + r, done, n,
                     traj[r][..., -1]) for r in range(c["R"])])
                traj = [np.concatenate([traj[r], segs[r]], axis=-1)
                        for r in range(c["R"])]
                np.savez(f, **{f"r{r}": traj[r] for r in range(c["R"])})
                last = time.time() - ts
                print(f"  {name}: {done + n}/{c['n_blocks']} blocks x "
                      f"{c['R']} [{time.time() - ts:.0f}s, total "
                      f"{time.time() - t0:.0f}s]", flush=True)
    return True


def load_case(raw: Path, name: str, R: int) -> np.ndarray:
    st = np.load(raw / f"{name}.npz")
    return np.array([st[f"r{r}"] for r in range(R)])     # [R, pole, snr, i]


# ---------------------------------------------------------------------------
# Part B: ramps
# ---------------------------------------------------------------------------
R_VALUES = (0.1, 1.0, 10.0)
POLES_B = (0.9, 0.95, 0.98, 0.99, 0.995)


def part_b_design(quick: bool):
    """Per r: beta, poles, discard D, measurement length L, interval centre;
    up and down ramps measured over the same a interval."""
    out = {}
    for r in R_VALUES:
        beta = r * 0.01 * SIGMA_SS
        poles = POLES_B + ((0.998,) if r < 5 else ())
        D = int(math.ceil(3 * n95(max(poles))))
        L = 3000 if r < 5 else 2500       # r = 10: keep |a| <= 1e-3 both ways
        a_c = 5e-4 if r < 5 else 0.0
        R = 8
        if quick:
            D, L, R = 40, 60, 2
        half = (D + L / 2) * beta         # a(0) = a_c -/+ half
        assert a_c + half <= 1e-3 + 1e-12 and a_c - half >= -1e-3 - 1e-12, \
            f"r = {r}: |a| would exceed 1e-3"
        out[r] = dict(beta=beta, poles=poles, D=D, L=L, a_c=a_c, R=R,
                      half=half)
    return out


def part_b_cases(design):
    cases = {}
    for r, d in design.items():
        rate = d["beta"] / T_BLK
        for direction, sgn in (("up", 1), ("down", -1)):
            a0 = d["a_c"] - sgn * d["half"]
            cases[f"B_r{r:g}_{direction}"] = dict(
                profile=RampDoppler(a0, sgn * rate), poles=d["poles"],
                n_blocks=d["D"] + d["L"], R=d["R"], a_hat0=a0)
    return cases


def analyse_b(design, raw, out, figd):
    rows = []
    for r, d in design.items():
        beta, D, L = d["beta"], d["D"], d["L"]
        eps = {}
        for direction, sgn in (("up", 1), ("down", -1)):
            tr = load_case(raw, f"B_r{r:g}_{direction}", d["R"])
            a0 = d["a_c"] - sgn * d["half"]
            prof = RampDoppler(a0, sgn * beta / T_BLK)
            tb = np.arange(D + L) * T_BLK
            a_bar = prof.block_mean(tb, tb + T_BLK)
            eps[direction] = (tr[..., :D + L] - a_bar)[..., D:]  # R,P,S,L
        for j, p in enumerate(d["poles"]):
            g = 1 - p
            for s_i, snr in enumerate(SNRS):
                up = eps["up"][:, j, s_i, :]
                dn = eps["down"][:, j, s_i, :]
                m_up, m_dn = up.mean(), dn.mean()
                se_up = up.mean(1).std(ddof=1) / np.sqrt(up.shape[0])
                se_dn = dn.mean(1).std(ddof=1) / np.sqrt(dn.shape[0])
                lag = 0.5 * (m_up - m_dn)
                bias = 0.5 * (m_up + m_dn)
                se_half = 0.5 * math.hypot(se_up, se_dn)
                std = math.sqrt(0.5 * (up.var() + dn.var()))
                rmse_up = math.sqrt(np.mean(up ** 2))
                rmse_dn = math.sqrt(np.mean(dn ** 2))
                rmse_dir = math.sqrt(0.5 * (rmse_up ** 2 + rmse_dn ** 2))
                rmse_nb = math.sqrt(0.5 * (np.mean((up - bias) ** 2)
                                           + np.mean((dn - bias) ** 2)))
                s = S_EK[snr]
                rows.append(dict(
                    r=r, beta=beta, beta_per_s=beta / T_BLK, pole=p, g=g,
                    snr_db=snr, R=up.shape[0], L=L, D=D,
                    lag_up=m_up, lag_down=m_dn, lag_se=se_half,
                    lag_biasfree=lag, lag_pred=lt.ramp_lag(beta, g),
                    bias_est=bias, std=std,
                    std_pred=lt.sigma_ss(s, g), rmse_up=rmse_up,
                    rmse_down=rmse_dn, rmse_dir=rmse_dir,
                    rmse_biasfree=rmse_nb,
                    rmse_pred=float(lt.ramp_rmse(beta, g, s))))
    with open(out / "tracking_ramp.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def snr_s(x):
        return "∞" if np.isinf(x) else f"{x:g} (WP)"

    L_ = ["# Ramp tracking and the choice of pole (Step 5, Part B)", "",
          "Generated by `study_tracking.py --part B`. Ramp β = r·g·σ_ss per "
          f"block (g = 0.01, σ_ss = {SIGMA_SS:.3e}), block = 6 ms. Poles "
          "p, μ = (1 − p)/K_d0, K_d0 = 175.5, Δ = {0}, δ = 0.25/N_seg. Each "
          "(r) is run as an UP and a DOWN ramp measured over the same a "
          "interval, after discarding D = 3·n95 of the slowest pole; "
          "â_0 = a(0). ε = â − a_bar per block (estimate used in the block "
          "minus the block mean). Lag (bias-free) = (ε_up − ε_down)/2, "
          "bias = (ε_up + ε_down)/2. Std: within each direction, pooled. "
          "RMSE: per direction (√ of the mean of the two directions' MSE) "
          "and bias-removed (ε minus the bias estimate). Prediction: lag "
          "−β/g, std (σ_e/K_d)·√(g/(2−g)), RMSE √((β/g)² + (σ_e/K_d)² "
          "g/(2−g)), with σ_e/K_d = 1.60e-5 (noiseless) and 4.59e-5 (WP) "
          "from Step 4. Std errors from the spread of the R realisation "
          "means.", ""]
    for r, d in design.items():
        L_ += [f"## r = {r:g}: β = {d['beta']:.3e} per block "
               f"({d['beta'] / T_BLK:.3e} /s); a measured over "
               f"[{d['a_c'] - d['L'] * d['beta'] / 2:.2e}, "
               f"{d['a_c'] + d['L'] * d['beta'] / 2:.2e}]; R = {d['R']}, "
               f"D = {d['D']}, L = {d['L']}", "",
               "| p | SNR | lag up | lag down | lag (bias-free) ± se | "
               "−β/g | bias | std / pred | RMSE per dir. | RMSE bias-removed "
               "| RMSE pred |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in [x for x in rows if x["r"] == r]:
            L_.append(
                f"| {x['pole']} | {snr_s(x['snr_db'])} | {x['lag_up']:+.2e} | "
                f"{x['lag_down']:+.2e} | **{x['lag_biasfree']:+.2e}** ± "
                f"{x['lag_se']:.1e} | {x['lag_pred']:+.2e} | "
                f"{x['bias_est']:+.2e} | {x['std']:.2e} / "
                f"{x['std_pred']:.2e} | {x['rmse_dir']:.2e} | "
                f"**{x['rmse_biasfree']:.2e}** | {x['rmse_pred']:.2e} |")
        L_.append("")

    # optimal pole
    L_ += ["## Optimal pole", "",
           "Analytical: g minimising √((β/g)² + (σ_e/K_d)² g/(2−g)) "
           "(continuous g). Simulated: the pole of the grid with the "
           "smallest RMSE — **based on the bias-removed RMSE** (the "
           "a-dependent detector bias is a separate, open issue, not a "
           "property of the loop bandwidth); the per-direction choice is "
           "given for comparison.", "",
           "| r | SNR | analytical p_opt | analytical RMSE | best grid p "
           "(analytical) | simulated p_opt (bias-removed) | its RMSE | "
           "simulated p_opt (per direction) | its RMSE |",
           "|---|---|---|---|---|---|---|---|---|"]
    opt = {}
    for r, d in design.items():
        for snr in SNRS:
            g_o, rm_o = lt.optimal_g(d["beta"], S_EK[snr])
            sel = [x for x in rows if x["r"] == r and x["snr_db"] == snr]
            grid = min(sel, key=lambda x: x["rmse_pred"])
            b_nb = min(sel, key=lambda x: x["rmse_biasfree"])
            b_dir = min(sel, key=lambda x: x["rmse_dir"])
            opt[(r, snr)] = (1 - g_o, rm_o)
            L_.append(f"| {r:g} | {snr_s(snr)} | {1 - g_o:.4f} | "
                      f"{rm_o:.2e} | {grid['pole']} | "
                      f"**{b_nb['pole']}** | {b_nb['rmse_biasfree']:.2e} | "
                      f"{b_dir['pole']} | {b_dir['rmse_dir']:.2e} |")
    L_ += ["", "**Open item (not investigated in this step):** the "
           "constant-a detector bias varies with a — +1.3e-6 ± 0.4e-6 at "
           "a = 5e-4 vs −0.56e-6 ± 0.4e-6 averaged over a ∈ [4.35, 5.65]e-4 "
           "(`results/verify_time_varying.md`, check 5). Probably the same "
           "phenomenon as the Step 4 closed-loop bias floor and the a = 0 "
           "sampling-phase offset. The up/down design removes it from the "
           "lag; the 'bias' column shows it.", ""]
    (out / "tracking_ramp.md").write_text("\n".join(L_), encoding="utf-8")
    print("\n".join(L_))
    plot_b(design, rows, opt, figd)


def plot_b(design, rows, opt, figd):
    cols = {0.1: "#1f77b4", 1.0: "#ff7f0e", 10.0: "#2ca02c"}
    with plt.rc_context({"font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for ax, snr in zip(axes, SNRS):
            gg = np.logspace(-3.2, -0.9, 300)
            for r, d in design.items():
                c = cols[r]
                ax.loglog(gg, lt.ramp_rmse(d["beta"], gg, S_EK[snr]), "-",
                          color=c, lw=1.2, label=f"r = {r:g}: prediction")
                sel = sorted([x for x in rows if x["r"] == r
                              and x["snr_db"] == snr], key=lambda x: x["g"])
                ax.loglog([x["g"] for x in sel],
                          [x["rmse_biasfree"] for x in sel], "o", color=c,
                          ms=7, label=f"r = {r:g}: simulated, bias removed")
                ax.loglog([x["g"] for x in sel],
                          [x["rmse_dir"] for x in sel], "x", color=c, ms=7,
                          label=f"r = {r:g}: simulated, per direction")
                po, rm = opt[(r, snr)]
                ax.plot(1 - po, rm, "*", color=c, ms=15, mec="k")
            ax.set_xlabel("loop gain g = 1 − p  (pole p = 1 − g)")
            ax.set_title(("noiseless" if np.isinf(snr) else
                          f"working point, {snr:g} dB")
                         + ": total RMSE vs loop gain (★ analytical optimum)",
                         fontsize=11)
            ax.grid(True, which="both", color="0.9", lw=0.5)
            sec = ax.secondary_xaxis("top", functions=(lambda g: g,
                                                       lambda g: g))
            sec.set_xticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
            sec.set_xticklabels(["0.998", "0.995", "0.99", "0.98", "0.95",
                                 "0.9"], fontsize=8)
            sec.set_xlabel("pole p", fontsize=9)
        axes[0].set_ylabel("RMSE of â − a_bar")
        axes[1].legend(fontsize=7.5, loc="upper right", ncol=1)
        fig.tight_layout()
        fig.savefig(figd / "fig_track_rmse_vs_pole.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, snr in zip(axes, SNRS):
            for r, d in design.items():
                sel = [x for x in rows if x["r"] == r and x["snr_db"] == snr]
                x = np.array([d["beta"] / s["g"] for s in sel])
                ax.errorbar(x, [-s["lag_biasfree"] for s in sel],
                            yerr=[2 * s["lag_se"] for s in sel], fmt="o",
                            color=cols[r], capsize=3,
                            label=f"r = {r:g} (bias-free)")
                ax.plot(x, [-s["lag_up"] for s in sel], "^", color=cols[r],
                        mfc="none", ms=6)
                ax.plot(x, [s["lag_down"] for s in sel], "v", color=cols[r],
                        mfc="none", ms=6)
            lim = [1e-8, 3e-4]
            ax.plot(lim, lim, "k--", lw=1, label="lag = β/g")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-7)
            ax.set_xlabel("predicted lag β/g")
            ax.set_title(("noiseless" if np.isinf(snr) else
                          f"working point, {snr:g} dB")
                         + ": measured lag vs β/g", fontsize=11)
            ax.grid(True, which="both", color="0.9", lw=0.5)
        axes[0].set_ylabel("measured lag −(ε_up − ε_down)/2 "
                           "(△ −ε_up, ▽ ε_down)")
        axes[1].legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(figd / "fig_track_lag.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Part C: sinusoid (frequency response)
# ---------------------------------------------------------------------------
POLES_C = (0.9, 0.99)
RATIOS = (0.1, 0.3, 1.0, 3.0)
AMP_C = 5e-4


def part_c_design(quick: bool, R: int):
    """One profile per (driving pole, ratio): freq = ratio f_3dB(pole).
    Both poles run on every profile (lockstep); the transient is 3 n95 of
    the slower pole, then >= 3 periods."""
    D = int(math.ceil(3 * n95(max(POLES_C))))
    out = {}
    for pd in POLES_C:
        f3 = lt.f_3db(pd, T_BLK)
        for ratio in RATIOS:
            f = ratio * f3
            period = 1.0 / (f * T_BLK)                  # blocks
            M = int(math.ceil(3 * period))
            if quick:
                M = min(M, 120)
            out[(pd, ratio)] = dict(freq=f, f3=f3, D=40 if quick else D,
                                    M=M, R=2 if quick else R,
                                    period=period)
    return out


def part_c_cases(design):
    return {f"C_p{pd:g}_x{ratio:g}": dict(
        profile=SinusoidalDoppler(0.0, AMP_C, d["freq"]), poles=POLES_C,
        n_blocks=d["D"] + d["M"], R=d["R"], a_hat0=0.0)
        for (pd, ratio), d in design.items()}


def _fit(x, w):
    """Complex amplitude of x[i] ~ c + Re(F e^{j w i}) (least squares)."""
    i = np.arange(x.size)
    X = np.column_stack([np.ones_like(i, dtype=float), np.cos(w * i),
                         np.sin(w * i)])
    c, A, B = np.linalg.lstsq(X, x, rcond=None)[0]
    return complex(A, -B)


def analyse_c(design, raw, out, figd):
    rows = []
    for (pd, ratio), d in design.items():
        tr = load_case(raw, f"C_p{pd:g}_x{ratio:g}", d["R"])
        n = d["D"] + d["M"]
        tb = np.arange(n) * T_BLK
        prof = SinusoidalDoppler(0.0, AMP_C, d["freq"])
        a_bar = prof.block_mean(tb, tb + T_BLK)[d["D"]:]
        w = 2 * np.pi * d["freq"] * T_BLK               # rad per block
        # fit on the post-transient blocks, time index from block D
        Fa = _fit(a_bar, w)
        for j, p in enumerate(POLES_C):
            Hp = complex(lt.H(d["freq"], p, T_BLK))
            for s_i, snr in enumerate(SNRS):
                ah = tr[:, j, s_i, d["D"]:n]                # R x M
                ratios = np.array([_fit(x, w) / Fa for x in ah])
                Hm = ratios.mean()
                se = (np.abs(ratios).std(ddof=1) / np.sqrt(len(ratios))
                      if len(ratios) > 1 else float("nan"))
                rmse = float(np.sqrt(np.mean((ah - a_bar) ** 2)))
                g = 1 - p
                pred = math.sqrt(abs(Fa) ** 2 * abs(1 - Hp) ** 2 / 2
                                 + S_EK[snr] ** 2 * g / (2 - g))
                rows.append(dict(
                    profile_pole=pd, ratio_drive=ratio, freq=d["freq"],
                    pole=p, x=d["freq"] / lt.f_3db(p, T_BLK), snr_db=snr,
                    R=d["R"], blocks=d["M"], periods=d["M"] / d["period"],
                    abar_amp=abs(Fa), gain=abs(Hm), gain_se=se,
                    gain_pred=abs(Hp), phase_deg=math.degrees(np.angle(Hm)),
                    phase_pred_deg=math.degrees(np.angle(Hp)),
                    rmse=rmse, rmse_pred=pred))
    rows.sort(key=lambda r: (r["pole"], r["snr_db"], r["x"]))
    with open(out / "tracking_sine.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def snr_s(x):
        return "∞" if np.isinf(x) else f"{x:g} (WP)"

    L_ = ["# Sinusoidal Doppler: frequency response of the loop (Step 5, "
          "Part C)", "",
          "Generated by `study_tracking.py --part C`. a(t) = 5e-4 sin(2π f "
          "t); for each pole p ∈ {0.9, 0.99} the profile frequencies are "
          "f = x·f_3dB(p), x ∈ {0.1, 0.3, 1, 3}; both poles run on every "
          "profile (lockstep), so each pole also gets the other pole's "
          "frequencies. f_3dB exact from |H|² = 1/2 with "
          "H(z) = g z⁻¹/(1 − p z⁻¹): "
          f"{lt.f_3db(0.9, T_BLK):.4f} Hz (p = 0.9), "
          f"{lt.f_3db(0.99, T_BLK):.4f} Hz (p = 0.99). Transient: "
          f"{design[(0.9, 0.1)]['D']} blocks (3·n95 of p = 0.99), then ≥ 3 "
          "periods. Gain and phase: least-squares fit of a sinusoid at the "
          "known frequency to â[i] and to a_bar[i] (same block index), "
          "ratio averaged over the realisations; prediction H at that "
          "frequency. RMSE of â − a_bar vs √(|a_bar amp|² |1 − H|²/2 + "
          "(σ_e/K_d)² g/(2 − g)) (the a_bar amplitude is 5e-4 · sinc of the "
          "block averaging, ≥ 0.996 · 5e-4 here).", ""]
    for p in POLES_C:
        L_ += [f"## Pole p = {p}", "",
               "| f / f_3dB | f (Hz) | SNR | gain ± se | \\|H\\| | phase (°) "
               "| arg H (°) | RMSE | RMSE pred | periods |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for x in [x for x in rows if x["pole"] == p]:
            L_.append(f"| {x['x']:.3g} | {x['freq']:.4g} | "
                      f"{snr_s(x['snr_db'])} | {x['gain']:.4f} ± "
                      f"{x['gain_se']:.4f} | {x['gain_pred']:.4f} | "
                      f"{x['phase_deg']:+.1f} | {x['phase_pred_deg']:+.1f} | "
                      f"{x['rmse']:.2e} | {x['rmse_pred']:.2e} | "
                      f"{x['periods']:.1f} |")
        L_.append("")
    (out / "tracking_sine.md").write_text("\n".join(L_), encoding="utf-8")
    print("\n".join(L_))
    plot_c(design, rows, raw, figd)


def plot_c(design, rows, raw, figd):
    with plt.rc_context({"font.size": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex="col")
        for j, p in enumerate(POLES_C):
            f3 = lt.f_3db(p, T_BLK)
            # up to just below the Nyquist frequency of the block rate
            xx = np.logspace(-2.3, np.log10(0.98 / (2 * T_BLK) / f3), 400)
            Hc = lt.H(xx * f3, p, T_BLK)
            axes[0, j].semilogx(xx, 20 * np.log10(np.abs(Hc)), "k-", lw=1.2,
                                label="|H| (prediction)")
            axes[1, j].semilogx(xx, np.degrees(np.angle(Hc)), "k-", lw=1.2,
                                label="arg H")
            for snr, mk, c in ((float("inf"), "o", "#1f77b4"),
                               (WP_DB, "s", "#d62728")):
                sel = [x for x in rows if x["pole"] == p
                       and x["snr_db"] == snr]
                lbl = "noiseless" if np.isinf(snr) else f"{snr:g} dB (WP)"
                axes[0, j].semilogx([x["x"] for x in sel],
                                    [20 * np.log10(x["gain"]) for x in sel],
                                    mk, color=c, mfc="none", ms=8, label=lbl)
                axes[1, j].semilogx([x["x"] for x in sel],
                                    [x["phase_deg"] for x in sel], mk,
                                    color=c, mfc="none", ms=8, label=lbl)
            axes[0, j].axhline(-3.01, color="0.5", ls=":", lw=1)
            axes[0, j].set_title(f"pole p = {p}  (f_3dB = {f3:.3g} Hz)",
                                 fontsize=11)
            axes[0, j].set_ylabel("gain (dB)")
            axes[1, j].set_ylabel("phase (°)")
            axes[1, j].set_xlabel("f / f_3dB")
            for ax in axes[:, j]:
                ax.grid(True, which="both", color="0.9", lw=0.5)
                ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(figd / "fig_track_sine_response.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(len(POLES_C), 1, figsize=(11, 6.5))
        for ax, p in zip(axes, POLES_C):
            d = design[(p, 1.0)]
            tr = load_case(raw, f"C_p{p:g}_x1", d["R"])
            j = POLES_C.index(p)
            n = d["D"] + d["M"]
            tb = np.arange(n) * T_BLK
            prof = SinusoidalDoppler(0.0, AMP_C, d["freq"])
            ax.plot(tb, prof.block_mean(tb, tb + T_BLK) * 1e3, "k-", lw=1.4,
                    label="a_bar")
            ax.plot(tb, tr[0, j, SNRS.index(WP_DB), :n] * 1e3, "-",
                    color="#d62728", lw=0.9, label="â (4 dB)")
            ax.axvline(d["D"] * T_BLK, color="0.6", ls="--", lw=1)
            ax.set_title(f"p = {p}, f = f_3dB = {d['freq']:.3g} Hz "
                         "(dashed: end of the discarded transient)",
                         fontsize=10.5)
            ax.set_ylabel(r"$a$ ($\times 10^{-3}$)")
            ax.grid(True, color="0.9", lw=0.5)
            ax.legend(fontsize=8.5, loc="upper right")
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(figd / "fig_track_sine_example.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Part D: fly-by
# ---------------------------------------------------------------------------
A_MAX_D = 5e-4
POLES_D = (0.9, 0.95, 0.99)


def part_d_design(quick: bool, R: int):
    """T_c from the peak slope per block a_max T_blk / T_c = beta(r).
    Poles: {0.9, 0.95, 0.99} + Part B's analytical WP optimum for r = 10
    (the simulated one, 0.95, is already in the set) + for r = 100 its own
    analytical WP optimum."""
    s = S_EK[WP_DB]
    p_b10 = round(1 - lt.optimal_g(10 * 0.01 * SIGMA_SS, s)[0], 4)
    out = {}
    for r in (10.0, 100.0):
        beta = r * 0.01 * SIGMA_SS
        T_c = A_MAX_D * T_BLK / beta
        poles = POLES_D + (p_b10,)
        if r == 100.0:
            poles = poles + (round(1 - lt.optimal_g(beta, s)[0], 4),)
        n = int(round(8 * T_c / T_BLK))
        out[r] = dict(beta=beta, T_c=T_c, t0=4 * T_c, n=min(n, 150) if quick
                      else n, poles=poles, R=2 if quick else R)
    return out


def part_d_cases(design):
    cases = {}
    for r, d in design.items():
        prof = FlyByDoppler(A_MAX_D, d["t0"], d["T_c"])
        cases[f"D_r{r:g}"] = dict(profile=prof, poles=d["poles"],
                                  n_blocks=d["n"], R=d["R"],
                                  a_hat0=float(prof.a(0.0)))
    return cases


def analyse_d(design, raw, out, figd):
    rows, series = [], {}
    for r, d in design.items():
        prof = FlyByDoppler(A_MAX_D, d["t0"], d["T_c"])
        tr = load_case(raw, f"D_r{r:g}", d["R"])
        n = d["n"]
        tb = np.arange(n + 1) * T_BLK
        a_bar = prof.block_mean(tb[:-1], tb[1:])
        a_bar_next = prof.block_mean(tb[1:], tb[1:] + T_BLK)
        beta_t = a_bar_next - a_bar                       # slope per block
        for j, p in enumerate(d["poles"]):
            g = 1 - p
            # exact linear-model response (noise-free): a_hat += g(a_bar-a_hat)
            lin = np.empty(n + 1)
            lin[0] = prof.a(0.0)
            for i in range(n):
                lin[i + 1] = lin[i] + g * (a_bar[i] - lin[i])
            eps_lin = lin[:n] - a_bar
            eps_qs = -beta_t / g
            # smoothing window for the deterministic peak: T_c / 20
            w = max(1, int(round(d["T_c"] / T_BLK / 20)))
            ker = np.ones(w) / w
            for s_i, snr in enumerate(SNRS):
                eps = tr[:, j, s_i, :n] - a_bar               # R x n
                m = eps.mean(0)
                m_s = np.convolve(m, ker, mode="same")
                sig = lt.sigma_ss(S_EK[snr], g)
                # std of the smoothed mean from noise alone (AR(1), rho=1-g)
                floor = (sig * math.sqrt(min(1.0, (2 - g) / (g * w)))
                         / math.sqrt(d["R"]))
                lin_s = np.convolve(eps_lin, ker, mode="same")
                core = slice(w, n - w) if n > 3 * w else slice(None)
                rows.append(dict(
                    r=r, T_c=d["T_c"], beta_peak=d["beta"], pole=p, g=g,
                    snr_db=snr, R=d["R"], blocks=n, smooth_w=w,
                    peak_smooth=float(np.max(np.abs(m_s[core]))),
                    peak_noise_floor=floor,
                    peak_lin_smooth=float(np.max(np.abs(lin_s[core]))),
                    peak_mean_err=float(np.max(np.abs(m))),
                    peak_qs=float(np.max(np.abs(eps_qs))),
                    peak_lin=float(np.max(np.abs(eps_lin))),
                    t_peak=float(tb[int(np.argmax(np.abs(m)))]),
                    peak_abs_err=float(np.mean(np.max(np.abs(eps), axis=1))),
                    rmse=float(np.sqrt(np.mean(eps ** 2))),
                    rmse_pred_lin=float(np.sqrt(np.mean(eps_lin ** 2)
                                                + sig ** 2)),
                    rmse_pred_qs=float(np.sqrt(np.mean(eps_qs ** 2)
                                               + sig ** 2)),
                    sigma_ss=sig))
                series[(r, p, snr)] = (tb[:n], a_bar, tr[0, j, s_i, :n], m,
                                       eps_qs, eps_lin)
    with open(out / "tracking_flyby.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def snr_s(x):
        return "∞" if np.isinf(x) else f"{x:g} (WP)"

    L_ = ["# Fly-by Doppler (Step 5, Part D)", "",
          "Generated by `study_tracking.py --part D`. a(t) = −a_max (t − t0)"
          " / √((t − t0)² + T_c²), a_max = 5e-4, run from t0 − 4T_c to "
          "t0 + 4T_c (t = 0 .. 8T_c), â_0 = a(0). T_c from the peak slope "
          "per block a_max T_blk / T_c = β(r), β(r) = r·g·σ_ss "
          "(g = 0.01, σ_ss = 3.25e-6). Poles {0.9, 0.95, 0.99} + Part B's "
          "analytical WP optimum for r = 10 (p = "
          f"{design[10.0]['poles'][3]}; the simulated grid optimum 0.95 is "
          "already in the set); for r = 100 also its own analytical WP "
          f"optimum (p = {design[100.0]['poles'][-1]}). ε = â − a_bar per "
          "block. Predictions: quasi-static ε(t) ≈ −β(t)/g (β(t) = slope "
          "per block); and the exact linear-model response (a_bar through "
          "â ← â + g(a_bar − â), noise-free). **Peak error** (the "
          "deterministic tracking error): max over t of |mean ε| after "
          "smoothing over w ≈ T_c/20 blocks (the profile changes on the T_c "
          "scale, so this removes noise but hardly the lag), with its noise "
          "floor σ_ss·√((2 − g)/(g w))/√R: a peak below ~3× the floor is not "
          "resolved. The linear model is smoothed the same way. 'Raw peak' = "
          "max |mean ε| without smoothing (noise-dominated for fast poles); "
          "'peak |ε|' = mean over realisations of max |ε| (with noise). "
          "RMSE over the whole run; prediction √(mean(ε_lin²) + σ_ss²).", ""]
    for r, d in design.items():
        w_r = [x for x in rows if x["r"] == r][0]["smooth_w"]
        L_ += [f"## r = {r:g}: T_c = {d['T_c']:.3f} s, peak slope β = "
               f"{d['beta']:.3e} per block, {d['n']} blocks "
               f"({d['n'] * T_BLK:.1f} s), R = {d['R']}, smoothing w = {w_r} "
               "blocks", "",
               "| p | SNR | peak ε (smoothed) | noise floor | linear model "
               "(smoothed) | quasi-static β/g | raw peak | peak \\|ε\\| (with "
               "noise) | RMSE | RMSE pred (linear) | RMSE pred (quasi-static) "
               "|",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in [x for x in rows if x["r"] == r]:
            L_.append(f"| {x['pole']} | {snr_s(x['snr_db'])} | "
                      f"**{x['peak_smooth']:.2e}** | "
                      f"{x['peak_noise_floor']:.1e} | "
                      f"{x['peak_lin_smooth']:.2e} | {x['peak_qs']:.2e} | "
                      f"{x['peak_mean_err']:.2e} | {x['peak_abs_err']:.2e} | "
                      f"**{x['rmse']:.2e}** | {x['rmse_pred_lin']:.2e} | "
                      f"{x['rmse_pred_qs']:.2e} |")
        L_.append("")
    (out / "tracking_flyby.md").write_text("\n".join(L_), encoding="utf-8")
    print("\n".join(L_))
    plot_d(design, series, figd)


def plot_d(design, series, figd):
    cmap = plt.get_cmap("viridis")
    with plt.rc_context({"font.size": 11}):
        for r, d in design.items():
            poles = d["poles"]
            col = {p: cmap(k / max(len(poles) - 1, 1))
                   for k, p in enumerate(sorted(poles))}
            fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            t, a_bar = series[(r, poles[0], WP_DB)][:2]
            axes[0].plot(t, a_bar * 1e3, "k-", lw=2, label="a_bar (true)")
            for p in poles:
                _, _, ah, m, eqs, elin = series[(r, p, WP_DB)]
                axes[0].plot(t, ah * 1e3, "-", color=col[p], lw=0.8,
                             label=f"â, p = {p} (4 dB, 1 realisation)")
                axes[1].plot(t, m * 1e6, "-", color=col[p], lw=1.2,
                             label=f"p = {p}: mean ε (4 dB)")
                axes[1].plot(t, eqs * 1e6, ":", color=col[p], lw=1.2)
            axes[1].plot([], [], "k:", lw=1.2, label="quasi-static −β(t)/g")
            axes[0].set_ylabel(r"$a$ ($\times 10^{-3}$)")
            axes[0].set_title(f"Fly-by, r = {r:g} (T_c = {d['T_c']:.2f} s, "
                              f"peak slope {d['beta']:.2e}/block), working "
                              "point 4 dB", fontsize=11)
            axes[1].set_ylabel(r"error ε = â − a_bar ($\times 10^{-6}$)")
            axes[1].set_xlabel("time (s)")
            for ax in axes:
                ax.grid(True, color="0.9", lw=0.5)
                ax.legend(fontsize=8, ncol=2, loc="upper right")
            fig.tight_layout()
            fig.savefig(figd / f"fig_track_flyby_r{r:g}.png", dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--part", choices=("B", "C", "D"), required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--budget", type=float, default=480.0,
                    help="seconds of computing per call before stopping "
                         "cleanly (rerun with --resume)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--R", type=int, default=4,
                    help="Part C/D: realisations per case")
    ap.add_argument("--seg", type=int, default=1500,
                    help="blocks per checkpointed segment (results do not "
                         "depend on it; smaller keeps each call short)")
    args = ap.parse_args()
    global SEG
    SEG = args.seg
    t0 = time.time()
    out = ROOT / "results" / ("quick" if args.quick else "")
    raw = out / "track_raw"
    figd = out if args.quick else ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    if not (args.resume or args.replot):
        for f in raw.glob(f"{args.part}_*.npz"):
            f.unlink()

    if args.part == "B":
        design = part_b_design(args.quick)
        cases = part_b_cases(design)
        if not args.replot:
            if not run_cases(cases, raw, args.budget, args.procs, t0):
                return
        analyse_b(design, raw, out, figd)
    elif args.part == "C":
        design = part_c_design(args.quick, args.R)
        cases = part_c_cases(design)
        if not args.replot:
            if not run_cases(cases, raw, args.budget, args.procs, t0):
                return
        analyse_c(design, raw, out, figd)
    else:
        design = part_d_design(args.quick, args.R)
        cases = part_d_cases(design)
        if not args.replot:
            if not run_cases(cases, raw, args.budget, args.procs, t0):
                return
        analyse_d(design, raw, out, figd)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
