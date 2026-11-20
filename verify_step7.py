"""
Step 7, Part A.4: verification of the waveform, channel and runner layers
(ocdm_doppler/waveforms.py, channel.py, loop_wf.py, wf_params.py).

1. OCDM through the new layers vs the existing code, bit for bit: received
   block (vs signal_tv.rx_nominal_profile on loop.LongSignal), open-loop e
   (vs discriminator.early_late), closed loop (vs loop_tv2.run_loop_filters,
   first- and second-order filters), noiseless and 4 dB, constant and ramp
   Doppler. Also: the general multipath formula with OCDM single path vs the
   old code (float rounding only).
2. OTFS (both guards), a = 0, single path, perfect timing: noiseless
   demodulation recovers the delay-Doppler data; guard samples are zero.
3. Mean power |X|^2 (OCDM, OTFS-symbol, OTFS-frame) vs the on/off pattern
   -> figures/fig_wf_mean_power.png.
4. OTFS working point: BPSK SER after demodulation vs sigma_D^2 / P_n
   (simulated and theory), 10 % point; OCDM on the same grid for reference.
5. Multipath at a = 0: received mean power vs the exact expression
   sum_pq h_p h_q* R_X(t - tau_p, t - tau_q) and vs the sum of shifted
   patterns sum_p |h_p|^2 pattern(t - tau_p) (equal when the cross terms
   vanish: integer-sample delay differences).

Output: results/verify_step7.md.   Usage: python verify_step7.py [--quick]
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

from ocdm_doppler import preset_noisy_22jan, chirp_matrix
from ocdm_doppler import loop_filters as lf
from ocdm_doppler import discriminator as disc
from ocdm_doppler.doppler_profile import ConstantDoppler, RampDoppler
from ocdm_doppler.loop import LongSignal
from ocdm_doppler.signal_tv import rx_nominal_profile
from ocdm_doppler.loop_tv2 import run_loop_filters
from ocdm_doppler.waveforms import OCDMWaveform, OTFSWaveform
from ocdm_doppler.channel import (single_path, exponential_pdp, received,
                                  rx_block, MultipathChannel, Path as CPath)
from ocdm_doppler.loop_wf import run_loop_wf, open_loop_e

ROOT = Path(__file__).parent
N_C, N_SEG, KD0 = 24000, 2048, 175.5
A0 = 5.1542e-4
LINES: list[str] = []
FAILS: list[str] = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def check(name, ok, detail):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def base():
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,))


def amp_of(p, snr):
    return 0.0 if np.isinf(snr) else math.sqrt(p.sigma_D2 / 10 ** (snr / 10) / 2)


# ---------------------------------------------------------------------------
def check1(quick):
    log("## 1. OCDM through the new layers vs the existing code")
    profiles = {"constant a = 5.1542e-4": ConstantDoppler(A0),
                "ramp 5.1542e-4 + 5.4e-5 /s": RampDoppler(A0, 5.4e-5)}
    blocks = (0, 7) if quick else (0, 7, 1000)
    gen_rel = []
    for name, prof in profiles.items():
        p = base().with_(a_true=float(prof.a(0.0)))
        d_max = 0.0
        for seed in (1000, 1001):
            ws = OCDMWaveform().signal(p, seed)
            ref = LongSignal(p, seed)
            for b in blocks:
                y_new = rx_block(ws, single_path(p), p, N_C, prof,
                                 start_sample=b * N_C)
                y_old = rx_nominal_profile(ref, N_C, prof, noiseless=True,
                                           start_sample=b * N_C)
                d_max = max(d_max, float(np.max(np.abs(y_new - y_old))))
                t = (b * N_C + np.arange(N_C)) * p.T_samp_ns
                y_gen = received(ws, single_path(p), p, t, prof)
                gen_rel.append(float(np.max(np.abs(y_gen - y_old))
                                     / np.sqrt(np.mean(np.abs(y_old) ** 2))))
        check(f"received block, {name}", d_max == 0.0,
              f"max |diff| = {d_max:.1e} (blocks {blocks}, 2 seeds)")
    log(f"   general multipath formula with OCDM single path vs the old code "
        f"(not used for this case; float rounding): max |diff| / rms = "
        f"{max(gen_rel):.1e}")

    # open-loop e vs discriminator.early_late
    p = base().with_(a_true=A0)
    prof = ConstantDoppler(A0)
    a_hats = A0 + np.array([-3e-3, -1e-4, 0.0, 1e-4, 3e-3])
    snrs = (float("inf"), 4.0)
    nb = 3
    e_new = open_loop_e(p, OCDMWaveform(), single_path(p), prof, a_hats, nb,
                        snrs=snrs, ks=(1,), seed=1000)
    ref = LongSignal(p, 1000)
    d_max = 0.0
    for i in range(nb):
        y0 = rx_nominal_profile(ref, N_C, prof, noiseless=True,
                                start_sample=i * N_C)
        rng = np.random.default_rng([1000, 0xA015E, i])
        w = rng.standard_normal(N_C) + 1j * rng.standard_normal(N_C)
        for s, snr in enumerate(snrs):
            y = y0 + amp_of(p, snr) * w if not np.isinf(snr) else y0
            e_old = disc.early_late(y, a_hats, p, k=1, N_seg=N_SEG)
            d_max = max(d_max, float(np.max(np.abs(e_new[s, i, 0] - e_old))))
    check("open-loop e vs discriminator.early_late", d_max == 0.0,
          f"max |diff| = {d_max:.1e} (5 estimates x {nb} blocks x "
          f"{{noiseless, 4 dB}})")

    # closed loop vs run_loop_filters
    nb = 30 if quick else 80
    for name, prof in profiles.items():
        p = base().with_(a_true=float(prof.a(0.0)))
        des = [lf.first_order_design(0.99), lf.equal_BL_design(0.95, 1.0)]
        filt = [des[0].filter(KD0, a_hat0=0.0),
                des[1].filter(KD0, a_hat0=A0 - 1e-4, b_hat0=0.0)]
        r_old = run_loop_filters(p, prof, filt, nb, snrs=snrs, seed=1000)
        r_new = run_loop_wf(p, OCDMWaveform(), single_path(p), prof, filt,
                            nb, snrs=snrs, seed=1000)
        d = max(float(np.nanmax(np.abs(r_new.a_hat - r_old.a_hat))),
                float(np.max(np.abs(r_new.e - r_old.e))),
                float(np.nanmax(np.abs(r_new.b_hat - r_old.b_hat))))
        check(f"closed loop vs run_loop_filters, {name}", d == 0.0,
              f"max |diff| (a_hat, b_hat, e) = {d:.1e} ({nb} blocks, 1st + "
              f"2nd order, noiseless + 4 dB)")
    log()


# ---------------------------------------------------------------------------
EPS = 1e-9   # sampling offset (in samples) that keeps floor() off slot edges


def otfs_sample(ws, p, frames, tau_s, prof=ConstantDoppler(0.0),
                chan=None, noise=None):
    """Active samples of the given frames, perfect timing, carrier and gain
    removed: y[f, n, i] (plus optional noise samples)."""
    chan = single_path(p) if chan is None else chan
    off = ws.active_sample_offsets()                       # (N, M)
    L = ws.wf.frame_len
    k = (np.asarray(frames)[:, None, None] * L + off[None]).astype(float)
    t = (tau_s + (k + EPS) * p.T_samp_syn).ravel()
    y = received(ws, chan, p, t, prof)
    if noise is not None:
        y = y + noise(t.size)
    derot = np.exp(-1j * (2 * np.pi * (p.f_Tx - p.f_Rx) * t
                          + p.phi_Tx - p.phi_Rx))
    h = p.A_p0 * np.exp(-2j * np.pi * p.f_Tx * p.tau_p0) / 2.0
    return (y * derot / h).reshape(k.shape)


def check2(quick):
    log("## 2. OTFS: noiseless demodulation at a = 0, single path, perfect "
        "timing")
    p = base().with_(a_true=0.0)
    nf = 20 if quick else 100
    for guard in ("symbol", "frame"):
        ws = OTFSWaveform(guard=guard).signal(p, 1000)
        frames = np.arange(nf) + 3
        y = otfs_sample(ws, p, frames, p.tau_p0)
        err = float(np.max(np.abs(ws.demod(y) - ws.data(frames))))
        # guard samples are zero
        L = ws.wf.frame_len
        allk = np.arange(L * 4) + 3 * L
        act = np.zeros(allk.size, bool)
        act[(ws.active_sample_offsets().ravel()[None, :]
             + (np.arange(4)[:, None] * L)).ravel()] = True
        t = p.tau_p0 + (allk + EPS) * p.T_samp_syn
        yg = received(ws, single_path(p), p, t, ConstantDoppler(0.0))
        g_max = float(np.max(np.abs(yg[~act])))
        n_on = int(act.sum() // 4)
        check(f"OTFS-{guard} demodulation", err < 1e-7,
              f"max |X_dd_hat - X_dd| = {err:.1e} (sampling offset "
              f"{EPS:g} T_samp alone gives ~{np.pi * EPS:.0e}) over {nf} frames "
              f"({nf * ws.M * ws.N} symbols); guard samples max |y| = "
              f"{g_max:.1e} ({L - n_on} of {L} samples per frame)")
    log()


# ---------------------------------------------------------------------------
def check3(quick):
    log("## 3. Mean power |X|^2 vs the on/off pattern")
    p = base().with_(a_true=0.0)
    n_samp = 64000 if quick else 640000
    wfs = [OCDMWaveform(), OTFSWaveform(guard="symbol"),
           OTFSWaveform(guard="frame")]
    curves = {}
    for wf in wfs:
        ws = wf.signal(p, 1000)
        P = ws.P_samp
        n_per = n_samp // P
        u = (np.arange(n_per * P) + 1e-6) * p.T_samp_syn
        x2 = (np.abs(ws.tx(u)) ** 2).reshape(n_per, P)
        mp = x2.mean(0)
        se = x2.std(0, ddof=1) / math.sqrt(n_per)
        pat = ws.mean_power(u[:P])
        on = pat > 0
        dev = np.abs(mp - pat)
        z = float(np.max(dev[on] / se[on]))
        check(f"{wf.name} (P_samp = {P})",
              bool(np.all(mp[~on] == 0.0)) and z < 5,
              f"{int(on.sum())} on / {int((~on).sum())} off; active: max "
              f"|mean - sigma_D^2| = {dev[on].max():.1e} = {z:.1f} std err "
              f"({n_per} periods); guard: max mean = {mp[~on].max():.1e}")
        curves[wf.name] = (mp, pat, P)
    log("   OTFS-symbol: the deviation is 0 up to rounding because the "
        "average runs over whole frames: sample i of slot n is a DFT over "
        "Doppler of delay row i, so by Parseval the power at each position, "
        "summed over the N slots of a frame, is exactly N sigma_D^2. (The "
        "std err assumes independent periods and is therefore not meaningful "
        "for it.)")
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for ax, (name, (mp, pat, P)) in zip(axes, curves.items()):
        n_show = 2 * 320
        reps = n_show // P
        ax.step(np.arange(n_show), np.tile(mp, reps), where="post", lw=1.2,
                label="simulated mean of |X|²")
        ax.step(np.arange(n_show), np.tile(pat, reps), where="post", lw=0.8,
                ls="--", color="k", label="on/off pattern")
        ax.set_ylabel("E|X|² / σ_D²")
        ax.set_title(f"{name}: period P_samp = {P} samples", fontsize=10)
        ax.set_ylim(-0.1, 1.3)
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_wf_mean_power.png", dpi=150)
    plt.close(fig)
    log("   figure: figures/fig_wf_mean_power.png")
    log()


# ---------------------------------------------------------------------------
def qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def check4(quick):
    log("## 4. Working point: BPSK SER after demodulation (a = 0, single "
        "path, perfect timing)")
    p = base().with_(a_true=0.0)
    g = abs(p.A_p0) ** 2 / 4.0
    # theory: Es/N0 = |A0|^2/4 * sigma_D^2/P_n, SER = Q(sqrt(2 Es/N0))
    lo, hi = -10.0, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if qfunc(math.sqrt(2 * g * 10 ** (mid / 10))) > 0.1:
            lo = mid
        else:
            hi = mid
    wp_theory = 0.5 * (lo + hi)
    snrs = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    nf = 40 if quick else 400
    h = p.A_p0 * np.exp(-2j * np.pi * p.f_Tx * p.tau_p0) / 2.0
    res = {}
    for name in ("OTFS-symbol", "OTFS-frame", "OCDM"):
        rows = []
        for j, snr in enumerate(snrs):
            P_n = p.sigma_D2 / 10 ** (snr / 10)
            rng = np.random.default_rng([2000, j, len(name)])

            def noise(n):
                return math.sqrt(P_n / 2) * (rng.standard_normal(n)
                                             + 1j * rng.standard_normal(n))
            if name.startswith("OTFS"):
                ws = OTFSWaveform(guard=name.split("-")[1]).signal(p, 3000 + j)
                frames = np.arange(nf)
                y = otfs_sample(ws, p, frames, p.tau_p0, noise=noise)
                d_hat = ws.demod(y)
                D = ws.data(frames)
                y0 = otfs_sample(ws, p, frames, p.tau_p0)
                p_noise = float(np.mean(np.abs((y - y0) * h) ** 2))
            else:
                ws = OCDMWaveform().signal(p, 3000 + j)
                n_sym = nf * 16
                m = np.arange(n_sym)[:, None]
                i = np.arange(p.N_is)[None, :]
                t = (p.tau_p0 + (m * p.N_sym + i + EPS)
                     * p.T_samp_syn).ravel()
                derot = np.exp(-1j * (2 * np.pi * (p.f_Tx - p.f_Rx) * t
                                      + p.phi_Tx - p.phi_Rx))
                y0f = received(ws, single_path(p), p, t, ConstantDoppler(0.0))
                yf = y0f + noise(t.size)
                Phi = chirp_matrix(p.N_is)
                d_hat = ((yf * derot / h).reshape(n_sym, p.N_is)
                         @ Phi.conj())
                D = ws.long.D[np.arange(n_sym), :]
                p_noise = float(np.mean(np.abs(yf - y0f) ** 2))
            dec = np.where(d_hat.real >= 0, 1.0, -1.0)
            ser = float(np.mean(dec != D.real))
            th = qfunc(math.sqrt(2 * g * 10 ** (snr / 10)))
            rows.append((snr, ser, th, p_noise / P_n, D.size))
        res[name] = rows
        ls = np.log([r[1] for r in rows])
        x = np.array(snrs)
        k = int(np.flatnonzero(ls < math.log(0.1))[0])
        wp = x[k - 1] + (math.log(0.1) - ls[k - 1]) * (x[k] - x[k - 1]) / (
            ls[k] - ls[k - 1])
        res[name + "_wp"] = wp
    log(f"   theory: Es/N0 = |A0|^2/4 * sigma_D^2/P_n; 10 % BPSK SER at "
        f"sigma_D^2/P_n = {wp_theory:.2f} dB")
    log("   | sigma_D²/P_n [dB] | SER OTFS-symbol | SER OTFS-frame | SER OCDM "
        "| theory | measured P_n / nominal |")
    log("   |---|---|---|---|---|---|")
    for j, snr in enumerate(snrs):
        a, b, c = (res[n][j] for n in ("OTFS-symbol", "OTFS-frame", "OCDM"))
        log(f"   | {snr:g} | {a[1]:.4f} | {b[1]:.4f} | {c[1]:.4f} | "
            f"{a[2]:.4f} | {a[3]:.3f} / {b[3]:.3f} / {c[3]:.3f} |")
    log(f"   ({res['OTFS-symbol'][0][4]} symbols per point and waveform)")
    for name in ("OTFS-symbol", "OTFS-frame", "OCDM"):
        wp = res[name + "_wp"]
        check(f"10 % working point, {name}", abs(wp - wp_theory) < 0.15,
              f"{wp:.2f} dB (theory {wp_theory:.2f} dB, Step 4 OCDM: 4.0 dB)")
    log()


# ---------------------------------------------------------------------------
def check5(quick):
    log("## 5. Multipath at a = 0: received mean power")
    p = base().with_(a_true=0.0)
    n_samp = 64000 if quick else 320000
    chans = {
        "4 paths, integer delays 0,1,2,3 (+2)": exponential_pdp(3, 4),
        "4 paths, integer delays 0,2,4,6 (+2)": exponential_pdp(6, 4),
        "4 paths, fractional delays 0,2/3,4/3,2 (+2)": exponential_pdp(2, 4),
        "2 paths, delay 0 and 2.5 (+2), equal power":
            MultipathChannel([CPath(1.0, 2.0), CPath(1j, 4.5)]),
    }
    for wf in (OCDMWaveform(), OTFSWaveform(guard="symbol"),
               OTFSWaveform(guard="frame")):
        for cname, ch in chans.items():
            ws = wf.signal(p, 1000)
            P = ws.P_samp
            n_per = max(n_samp // P, 200)
            t = (np.arange(n_per * P) + 0.37) * p.T_samp_syn
            y = received(ws, ch, p, t, ConstantDoppler(0.0))
            y2 = (np.abs(y) ** 2).reshape(n_per, P)
            emp = y2.mean(0)
            se = y2.std(0, ddof=1) / math.sqrt(n_per)
            # reference instants several periods into the record, so that
            # t - tau_p never falls before the start of the signal
            t0 = t[:P] + 10 * P * p.T_samp_syn
            hs =[p.A_p0 * q.gain * np.exp(-2j * np.pi * p.f_Tx * q.delay
                                           * p.T_samp_syn) / 2.0
                  for q in ch.paths]
            taus = [q.delay * p.T_samp_syn for q in ch.paths]
            exact = np.zeros(P)
            for hp, tp in zip(hs, taus):
                for hq, tq in zip(hs, taus):
                    exact += np.real(hp * np.conj(hq)
                                     * ws.autocorr(t0 - tp, t0 - tq))
            sump = sum(abs(hp) ** 2 * ws.mean_power(t0 - tp)
                       for hp, tp in zip(hs, taus))
            z = float(np.max(np.abs(emp - exact) / np.maximum(se, 1e-12)))
            cross = float(np.max(np.abs(exact - sump)) / np.max(sump))
            check(f"{wf.name}, {cname}", z < 5,
                  f"empirical vs exact: max {z:.1f} std err; exact vs sum of "
                  f"shifted patterns: max rel. diff {cross:.1e}")
    log("   (sample instants offset by 0.37 T_samp so that no sample sits on "
        "a pattern edge)")
    log()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    log("# Step 7, Part A: verification of the waveform, channel and runner "
        "layers")
    log()
    log("Generated by `verify_step7.py`" + (" --quick" if args.quick else "")
        + ".")
    log()
    check1(args.quick)
    check2(args.quick)
    check3(args.quick)
    check4(args.quick)
    check5(args.quick)
    log(f"SUMMARY: {len(FAILS)} failed"
        + (f" ({', '.join(FAILS)})" if FAILS else " -- all checks passed")
        + f"  [{time.time() - t0:.0f}s]")
    out = ROOT / "results" / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify_step7.md").write_text("\n".join(LINES) + "\n",
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
