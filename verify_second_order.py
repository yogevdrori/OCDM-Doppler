"""
Step 6, Part A.3: verification of the loop filters (loop_filters.py) and the
filter runner (loop_tv2.py). All numbers are printed and written to
results/verify_second_order.md.

1. FirstOrderFilter in run_loop_filters vs loop_tv.run_loop_profile (constant
   non-integer a, ramp, sinusoid; noiseless and 4 dB): bit for bit.
2. SecondOrderFilter(mu1, mu2 = 0) vs the first-order loop with mu1: bit
   for bit.
3. Lockstep (several filters) vs separate runs, and a run in segments
   (states passed on) vs one run: bit for bit.
4. Analytical helpers vs an independent direct linear recursion: step
   (n95, overshoot), ramp (0 / -beta/g), parabola (-gamma/g2), B_L
   (Lyapunov vs sum of h^2), H2 vs the recursion's response to a sinusoid;
   and the design table of the task file, recomputed.
5. Noiseless closed loop on a ramp (r = 10, up and down over the same
   interval): bias-free lag of the second-order loop vs 0 and of the
   first-order loop vs -beta/g.

Usage:  python verify_second_order.py      (~5 min)
"""

from __future__ import annotations

import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ocdm_doppler import preset_noisy_22jan
from ocdm_doppler import loop_filters as lf
from ocdm_doppler.doppler_profile import (ConstantDoppler, RampDoppler,
                                          SinusoidalDoppler)
from ocdm_doppler.loop_tv import run_loop_profile
from ocdm_doppler.loop_tv2 import run_loop_filters

ROOT = Path(__file__).parent
KD0 = 175.5
T_BLK = 24000 / 4e6
A_NI = 5.1542e-4                  # a N_c = 12.37: non-integer shift per block
SNRS = (float("inf"), 4.0)
SIGMA_SS = 4.59e-5 * math.sqrt(0.01 / 1.99)

OUT: list[str] = []
FAILS: list[str] = []


def log(msg=""):
    print(msg, flush=True)
    OUT.append(msg)


def check(name, ok, detail):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def base():
    return preset_noisy_22jan(float("inf")).with_(cbsfs_deltas=(0,))


PROFILES = {
    "constant a = 5.1542e-4": ConstantDoppler(A_NI),
    "ramp 5.4e-5 /s": RampDoppler(-3e-4, 5.4e-5 * 10),
    "sinusoid 5e-4 at 1 Hz": SinusoidalDoppler(0.0, 5e-4, 1.0),
}


# ---------------------------------------------------------------------------
def check1_2():
    log("\n## 1-2. Equivalence with the first-order loop of Step 5")
    mus = [0.1 / KD0, 0.01 / KD0]
    nb = 200
    w1, w2, w2s = 0.0, 0.0, 0.0
    for name, prof in PROFILES.items():
        a0 = float(prof.a(0.0))
        ref = run_loop_profile(base(), prof, mu=mus, n_blocks=nb, snrs=SNRS,
                               a_hat0=a0, seed=1000)
        filt = [lf.FirstOrderFilter(m, a0) for m in mus] + \
               [lf.SecondOrderFilter(m, 0.0, a0, 0.0) for m in mus]
        new = run_loop_filters(base(), prof, filt, nb, snrs=SNRS, seed=1000)
        w1 = max(w1, np.max(np.abs(new.a_hat[:2] - ref.a_hat)))
        w2 = max(w2, np.max(np.abs(new.a_hat[2:] - ref.a_hat)))
        w2s = max(w2s, np.max(np.abs(new.b_hat[2:])))
    check("FirstOrderFilter (run_loop_filters) == run_loop_profile, 3 "
          "profiles x 2 mu x {noiseless, 4 dB}", w1 == 0.0,
          f"max |a_hat diff| = {w1:.1e} over {nb} blocks")
    check("SecondOrderFilter(mu1, mu2 = 0) == first-order loop (mu1)",
          w2 == 0.0 and w2s == 0.0,
          f"max |a_hat diff| = {w2:.1e}, max |b_hat| = {w2s:.1e}")


def check3():
    log("\n## 3. Lockstep vs separate runs; segmented vs one run")
    d_cd = lf.equal_BL_design(0.95, 1.0)
    d_z = lf.equal_BL_design(0.95, 0.707)
    prof = PROFILES["sinusoid 5e-4 at 1 Hz"]
    designs = [lf.first_order_design(0.95), d_cd, d_z]
    filt = [d.filter(KD0, 0.0, 0.0) for d in designs]
    nb = 240
    lock = run_loop_filters(base(), prof, filt, nb, snrs=SNRS, seed=1001)
    worst = 0.0
    for m, f in enumerate(filt):
        sep = run_loop_filters(base(), prof, [f], nb, snrs=SNRS, seed=1001)
        worst = max(worst, np.max(np.abs(sep.a_hat[0] - lock.a_hat[m])),
                    np.nanmax(np.abs(np.nan_to_num(sep.b_hat[0])
                                     - np.nan_to_num(lock.b_hat[m]))))
    check("lockstep [1st 0.95, 2nd cd, 2nd zeta 0.707] == separate runs",
          worst == 0.0, f"max |diff| (a_hat, b_hat) = {worst:.1e}")
    seg_a, seg_b, st = [lock.a_hat[..., :1]], [lock.b_hat[..., :1]], None
    for s0 in (0, 80, 160):
        r = run_loop_filters(base(), prof, filt, 80, snrs=SNRS, seed=1001,
                             start_block=s0, states=st)
        seg_a.append(r.a_hat[..., 1:])
        seg_b.append(r.b_hat[..., 1:])
        st = r.states
    da = np.max(np.abs(np.concatenate(seg_a, -1) - lock.a_hat))
    db = np.nanmax(np.abs(np.concatenate(seg_b, -1) - lock.b_hat))
    check("3 segments (states passed on) == one run", da == 0.0 and db == 0.0,
          f"max |a_hat diff| = {da:.1e}, max |b_hat diff| = {db:.1e}")


# ---------------------------------------------------------------------------
def recursion(g1, g2, a_bar, a0=0.0, b0=0.0):
    """Independent direct implementation of the linear loop."""
    a, b, out = a0, b0, [a0]
    for x_bar in a_bar:
        err = x_bar - a
        b += g2 * err
        a += b + g1 * err
        out.append(a)
    return np.array(out)


TABLE = {   # task file: p1 -> (B_L, p, g1, g2, n95 2nd, n95 1st, overshoot)
    0.9: (0.0526, 0.95942, 0.0795, 1.65e-3, 101, 28, 0.14),
    0.95: (0.0256, 0.97986, 0.0399, 4.06e-4, 205, 58, 0.14),
    0.99: (0.00503, 0.99599, 0.00800, 1.60e-5, 1033, 298, 0.14),
}


def check4():
    log("\n## 4. Analytical helpers vs direct recursion; design table")
    designs = [lf.first_order_design(0.95), lf.equal_BL_design(0.95, 1.0),
               lf.equal_BL_design(0.95, 0.707), lf.equal_BL_design(0.99, 1.0)]
    worst_bl, worst_ramp, worst_par, worst_h = 0.0, 0.0, 0.0, 0.0
    for d in designs:
        g1, g2 = d.g1, d.g2
        n = lf._horizon(g1, g2) * 3
        h = np.diff(recursion(g1, g2, np.ones(n)))       # impulse = diff(step)
        bl_sum = float(np.sum(h ** 2))
        worst_bl = max(worst_bl, abs(bl_sum - d.BL) / d.BL)
        beta = 3.25e-7
        ramp = recursion(g1, g2, beta * np.arange(n))
        err_r = ramp[n - 1] - beta * (n - 1)
        worst_ramp = max(worst_ramp,
                         abs(err_r - lf.ss_error_ramp(beta, g1, g2)) / beta)
        if g2 > 0:
            gam = 1e-9
            i = np.arange(n)
            par = recursion(g1, g2, 0.5 * gam * i * (i + 1))
            err_p = par[n - 1] - 0.5 * gam * (n - 1) * n
            worst_par = max(worst_par, abs(err_p / lf.ss_error_parabola(
                gam, g1, g2) - 1))
        f3 = lf.f_3db(g1, g2, T_BLK)
        for f in (0.3 * f3, f3, 3 * f3):
            m = 40 * int(1 / (f * T_BLK)) + n
            k = np.arange(m)
            x = np.sin(2 * np.pi * f * T_BLK * k)
            y = recursion(g1, g2, x)[:m]
            tail = slice(m - 20 * int(1 / (f * T_BLK)), m)
            X = np.column_stack([np.sin(2 * np.pi * f * T_BLK * k[tail]),
                                 np.cos(2 * np.pi * f * T_BLK * k[tail])])
            cs, cc = np.linalg.lstsq(X, y[tail], rcond=None)[0]
            # x = sin(wk) -> y = |H| sin(wk + arg H) = cs sin + cc cos
            Hm = complex(cs, cc)
            Hp = complex(lf.H2(f, g1, g2, T_BLK))
            worst_h = max(worst_h, abs(Hm - Hp))
        log(f"   {d.name:22s} g1={g1:.5f} g2={g2:.3e} B_L={d.BL:.5f} "
            f"(sum h^2 {bl_sum:.5f}) n95={lf.n95(g1, g2)} "
            f"overshoot={lf.overshoot(g1, g2):.1%} f_3dB={f3:.4f} Hz "
            f"|H2(f_3dB)|={abs(lf.H2(f3, g1, g2, T_BLK)):.4f}")
    check("B_L: Lyapunov vs sum of h^2 (recursion)", worst_bl < 1e-7,
          f"max rel. diff {worst_bl:.1e} (float accumulation in the sum)")
    check("ramp: steady-state error vs 0 (2nd) / -beta/g (1st)",
          worst_ramp < 1e-6, f"max |diff| / beta = {worst_ramp:.1e}")
    check("parabola: steady-state error vs -gamma/g2", worst_par < 1e-6,
          f"max rel. diff {worst_par:.1e}")
    check("H2 vs the recursion's response to a sinusoid (0.3, 1, 3 f_3dB)",
          worst_h < 1e-6, f"max |H_meas - H2| = {worst_h:.1e}")

    log("\n   Design table of the task file, recomputed "
        "(equal-B_L critically damped designs):")
    log("   | p1 | B_L | p (double pole) | g1 | g2 | n95 2nd / 1st | "
        "overshoot | task file |")
    def sig(x, n):
        return float(f"{x:.{n}g}")

    mismatches = []
    for p1, ref in TABLE.items():
        bl = lf.first_order_BL(1 - p1)
        d = lf.equal_BL_design(p1, 1.0)
        p = math.sqrt(1 - d.g1)
        n2, n1 = lf.n95(d.g1, d.g2), lf.n95(1 - p1, 0.0)
        n1c = math.log(0.05) / math.log(p1)
        ov = lf.overshoot(d.g1, d.g2)
        log(f"   | {p1} | {bl:.5f} | {p:.5f} | {d.g1:.5f} | {d.g2:.3e} | "
            f"{n2} / {n1} ({n1c:.1f}) | {ov:.1%} | {ref} |")
        # compare at the precision printed in the task file
        cmp = [("B_L", sig(bl, 3), ref[0]), ("p", round(p, 5), ref[1]),
               ("g1", sig(d.g1, 3), ref[2]), ("g2", sig(d.g2, 3), ref[3]),
               ("n95 2nd", n2, ref[4]), ("n95 1st (floor of continuous)",
                                         int(n1c), ref[5]),
               ("overshoot", round(ov, 2), ref[6])]
        mismatches += [f"p1={p1} {k}: {a} vs {b}" for k, a, b in cmp
                       if abs(a - b) > 1e-12 * max(1, abs(b))]
    check("task-file table, at its printed precision", not mismatches,
          "all values agree" if not mismatches else "; ".join(mismatches))
    log("   (n95 1st order: the integer settling block is one more than the "
        "table's 28 / 58 / 298, which are the continuous values "
        "ln 0.05 / ln p1 = 28.4 / 58.4 / 298.1 truncated)")


# ---------------------------------------------------------------------------
def _ramp_job(args):
    direction, r_i, nb, beta, a_c, D, L = args
    sgn = 1 if direction == "up" else -1
    a0 = a_c - sgn * (D + L / 2) * beta
    prof = RampDoppler(a0, sgn * beta / T_BLK)
    d1 = lf.first_order_design(0.95)
    d2 = lf.equal_BL_design(0.95, 1.0)
    filt = [d1.filter(KD0, a0), d2.filter(KD0, a0, sgn * beta)]
    res = run_loop_filters(base(), prof, filt, nb, snrs=(float("inf"),),
                           seed=1000 + r_i)
    return direction, res.eps[:, 0, D:]                     # [2, L]


def check5():
    log("\n## 5. Noiseless closed loop on a ramp (r = 10), up and down")
    beta = 10 * 0.01 * SIGMA_SS
    D, L, R, a_c = 900, 1500, 4, 0.0
    jobs = [(d, r, D + L, beta, a_c, D, L) for d in ("up", "down")
            for r in range(R)]
    with Pool(4) as pool:
        res = pool.map(_ramp_job, jobs)
    up = np.array([x[1] for x in res if x[0] == "up"])      # R x 2 x L
    dn = np.array([x[1] for x in res if x[0] == "down"])
    log(f"   beta = {beta:.3e} per block; designs: 1st p = 0.95 (g = 0.05), "
        "2nd critically damped with the same B_L (b_hat0 = beta); "
        f"R = {R}, discard {D}, L = {L}; a over [{a_c - L * beta / 2:.1e}, "
        f"{a_c + L * beta / 2:.1e}]")
    for m, name, pred in ((0, "1st p = 0.95", -beta / 0.05),
                          (1, "2nd cd (equal B_L)", 0.0)):
        mu_, md_ = up[:, m].mean(1), dn[:, m].mean(1)
        lag = 0.5 * (mu_.mean() - md_.mean())
        se = 0.5 * math.hypot(mu_.std(ddof=1), md_.std(ddof=1)) / math.sqrt(R)
        bias = 0.5 * (mu_.mean() + md_.mean())
        # drift check on the same a sub-interval: the first half of the up
        # ramp covers the same a as the second half of the down ramp
        h1 = 0.5 * (up[:, m, :L // 2].mean() - dn[:, m, L // 2:].mean())
        h2 = 0.5 * (up[:, m, L // 2:].mean() - dn[:, m, :L // 2].mean())
        log(f"   {name}: bias-free lag {lag:+.2e} +/- {se:.1e} (pred "
            f"{pred:+.2e}); bias {bias:+.2e}; bias-free lag on the lower / "
            f"upper half of the a interval {h1:+.2e} / {h2:+.2e}")
        ok = abs(lag - pred) < max(3 * se, 0.03 * abs(pred) + 1e-8)
        check(f"ramp lag, {name}", ok,
              f"{lag:+.2e} vs {pred:+.2e} ({(lag - pred) / se:+.1f} sigma)")


def main():
    t0 = time.time()
    log("# Step 6, Part A: verification of the loop filters")
    log("")
    log("Generated by `verify_second_order.py`.")
    check1_2()
    check3()
    check4()
    check5()
    log(f"\nSUMMARY: {len(FAILS)} failed"
        + (": " + ", ".join(FAILS) if FAILS else " -- all checks passed")
        + f"  [{time.time() - t0:.0f}s]")
    (ROOT / "results" / "verify_second_order.md").write_text(
        "\n".join(OUT) + "\n", encoding="utf-8")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
