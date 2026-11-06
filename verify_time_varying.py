"""
Step 5, Part A.4: verification of the time-varying Doppler infrastructure
(doppler_profile.py, signal_tv.py, loop_tv.py). All numbers are printed and
written to results/verify_time_varying.md.

1. Equivalence with the existing code (constant a)
   a. rx_profile(ConstantDoppler(a)) vs OCDMSignal.rx(a) (delegated: must be
      exactly equal) and the GENERAL formula with a constant profile
      (RampDoppler(a, 0)) vs OCDMSignal.rx(a) (floating-point level), for
      a in {0, +/-5e-4, +/-1e-3}: noisy on an OCDMSignal for t <= 1 s,
      noiseless on a loop.LongSignal for t up to 120 s (the longest run).
   b. run_loop_profile(ConstantDoppler(a)) vs loop.run_loop(a), noiseless and
      4 dB: bit-for-bit; lockstep (several mu) vs separate single-mu runs:
      bit-for-bit; the general formula (RampDoppler(a, 0)): float level.
2. Integrals: A(t) of every profile vs a composite Gauss-Legendre integral of
   a(t) (subintervals split at the knots), error in s and in carrier phase.
3. Continuity: one long record vs the same record block by block; a loop run
   in segments (start_block, a_hat0 = last a_hat) vs one run.
4. Old results unchanged: git diff of the protected modules; recompute
   committed raw data (Part B: one a x 64 blocks; Part D: first 300 blocks of
   one trajectory) and compare bit for bit.
5. Detector on a ramp (noiseless, open loop, beta = 3.25e-7 per block):
   mean of e/K_d0 - (a_bar - a_hat) vs the same at constant a.

Usage:  python verify_time_varying.py      (~3-4 min)
"""

from __future__ import annotations

import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ocdm_doppler import OCDMSignal, preset_noisy_22jan
from ocdm_doppler import loop as lp
from ocdm_doppler.doppler_profile import (ConstantDoppler, RampDoppler,
                                          SinusoidalDoppler, FlyByDoppler,
                                          PiecewiseLinearDoppler,
                                          CallableDoppler)
from ocdm_doppler.signal_tv import rx_profile, rx_nominal_profile
from ocdm_doppler.loop_tv import run_loop_profile

ROOT = Path(__file__).parent
N_C, N_SEG = 24000, 2048
KD0 = 175.5
T_BLK = N_C / 4e6
A_VALS = (0.0, 5e-4, -5e-4, 1e-3, -1e-3)
PROTECTED = ["params.py", "signal_model.py", "analytical_af.py", "cbsfs.py",
             "af_estimation.py", "curve_fit.py", "discriminator.py"]

OUT: list[str] = []
FAILS: list[str] = []


def log(msg=""):
    print(msg, flush=True)
    OUT.append(msg)


def check(name, ok, detail):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def base(snr=float("inf"), lags=(0,)):
    return preset_noisy_22jan(snr).with_(cbsfs_deltas=lags)


# ---------------------------------------------------------------------------
def check1a():
    log("\n## 1a. rx_profile vs OCDMSignal.rx (constant a)")
    # noisy, OCDMSignal, t <= 1 s, compact windows (small noise cache)
    p = base(4.0)
    n_sym = int(1.0 / p.T_sym) + 10
    starts = [0, 40_000, 400_000, 1_600_000, 3_900_000]      # nominal samples
    k = np.concatenate([s + np.arange(400) for s in starts])
    t = k * p.T_samp_ns
    worst_d, worst_g = 0.0, 0.0
    for a in A_VALS:
        sig = OCDMSignal(p.with_(a_true=a), n_symbols=n_sym, seed=11)
        ref = sig.rx(t, a=a)
        d = np.max(np.abs(rx_profile(sig, t, ConstantDoppler(a)) - ref))
        g = np.max(np.abs(rx_profile(sig, t, RampDoppler(a, 0.0)) - ref))
        worst_d, worst_g = max(worst_d, d), max(worst_g, g)
    peak = np.max(np.abs(ref))
    check("noisy, t <= 1 s, ConstantDoppler (delegated)", worst_d == 0.0,
          f"max |diff| = {worst_d:.1e}")
    log(f"   general formula RampDoppler(a, 0) vs OCDMSignal.rx, noisy, "
        f"t <= 1 s: max |diff| = {worst_g:.2e} (|Y| up to {peak:.2f})")
    # noiseless, LongSignal, t up to 120 s
    p0 = base()
    k = np.concatenate([s + np.arange(2000) for s in
                        (0, 4_000_000, 40_000_000, 240_000_000,
                         480_000_000 - 2000)])
    t = k * p0.T_samp_ns
    worst_d, worst_g = 0.0, 0.0
    for a in A_VALS:
        sig = lp.LongSignal(p0.with_(a_true=a), seed=12)
        ref = sig.rx(t, a=a, noiseless=True)
        d = np.max(np.abs(rx_profile(sig, t, ConstantDoppler(a),
                                     noiseless=True) - ref))
        g = np.max(np.abs(rx_profile(sig, t, RampDoppler(a, 0.0),
                                     noiseless=True) - ref))
        worst_d, worst_g = max(worst_d, d), max(worst_g, g)
    check("noiseless, t up to 120 s, ConstantDoppler (delegated)",
          worst_d == 0.0, f"max |diff| = {worst_d:.1e}")
    log(f"   general formula RampDoppler(a, 0) vs OCDMSignal.rx, noiseless, "
        f"t up to 120 s: max |diff| = {worst_g:.2e} (|Y| ~ 0.5); see the "
        "exact-phase comparison below")
    exact_phase_check(p0)


def exact_phase_check(p):
    """Carrier phase and time warp of the old formula ((1+a) f_Tx - f_Rx,
    (1+a) t) and of the general one ((f_Tx - f_Rx) t + f_Tx a t, t + a t)
    against a 50-digit Decimal reference (a, t taken as the exact binary
    values of the floats)."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    PI2 = Decimal("6.28318530717958647692528676655900576839433879875021164")
    fTx, fRx, tau = Decimal(p.f_Tx), Decimal(p.f_Rx), Decimal(p.tau_p0)

    def wrap(x):                                # Decimal -> (-pi, pi]
        y = x % PI2
        return float(y - PI2 if y > PI2 / 2 else y)

    worst = {"old_ph": 0.0, "new_ph": 0.0, "old_u": 0.0, "new_u": 0.0}
    for a in A_VALS:
        for t in (0.5, 1.8, 30.0, 74.0, 120.0):
            da, dt = Decimal(a), Decimal(t)
            ph_ex = PI2 * (((1 + da) * fTx - fRx) * dt)
            u_ex = (1 + da) * dt - tau
            ph_old = 2 * np.pi * p.Delta_f_of_a(a) * t
            ph_new = 2 * np.pi * ((p.f_Tx - p.f_Rx) * t + p.f_Tx * (a * t))
            u_old = (1.0 + a) * t - p.tau_p0
            u_new = t + a * t - p.tau_p0
            worst["old_ph"] = max(worst["old_ph"],
                                  abs(wrap(Decimal(ph_old) - ph_ex)))
            worst["new_ph"] = max(worst["new_ph"],
                                  abs(wrap(Decimal(ph_new) - ph_ex)))
            worst["old_u"] = max(worst["old_u"],
                                 abs(float(Decimal(u_old) - u_ex)))
            worst["new_u"] = max(worst["new_u"],
                                 abs(float(Decimal(u_new) - u_ex)))
    log(f"   vs exact (50-digit) reference, a in {A_VALS}, t up to 120 s:")
    log(f"     carrier phase error: old formula {worst['old_ph']:.1e} rad, "
        f"general formula {worst['new_ph']:.1e} rad")
    log(f"     time-warp error:     old formula {worst['old_u']:.1e} s, "
        f"general formula {worst['new_u']:.1e} s")
    check("general formula at least as accurate as the old one (the "
          "difference is the old formula's float rounding)",
          worst["new_ph"] <= worst["old_ph"] and worst["new_ph"] < 1e-6
          and worst["new_u"] < 1e-13,
          f"phase {worst['new_ph']:.1e} vs {worst['old_ph']:.1e} rad; "
          f"warp {worst['new_u']:.1e} vs {worst['old_u']:.1e} s")


def check1b():
    log("\n## 1b. run_loop_profile vs loop.run_loop (constant a)")
    mu99, mu90 = 0.01 / KD0, 0.1 / KD0
    snrs = (float("inf"), 4.0)
    nb = 300
    worst_c, worst_lock, worst_g = 0.0, 0.0, 0.0
    for a in (5e-4, -1e-3):
        ref = lp.run_loop(base(), a, mu=mu99, n_blocks=nb, snrs=snrs,
                          seed=1000)
        new = run_loop_profile(base(), ConstantDoppler(a), mu=mu99,
                               n_blocks=nb, snrs=snrs, seed=1000)
        worst_c = max(worst_c, np.max(np.abs(new.a_hat - ref.a_hat)))
        lock = run_loop_profile(base(), ConstantDoppler(a),
                                mu=[mu90, mu99], n_blocks=nb, snrs=snrs,
                                seed=1000)
        sep90 = run_loop_profile(base(), ConstantDoppler(a), mu=mu90,
                                 n_blocks=nb, snrs=snrs, seed=1000)
        worst_lock = max(worst_lock,
                         np.max(np.abs(lock.a_hat[0] - sep90.a_hat)),
                         np.max(np.abs(lock.a_hat[1] - new.a_hat)))
        gen = run_loop_profile(base(), RampDoppler(a, 0.0), mu=mu99,
                               n_blocks=nb, snrs=snrs, seed=1000)
        dg = np.max(np.abs(gen.a_hat - ref.a_hat), axis=-1)   # per SNR
        worst_g = np.maximum(worst_g, dg)
    check("scalar mu, ConstantDoppler == run_loop (noiseless, 4 dB)",
          worst_c == 0.0, f"max |a_hat diff| = {worst_c:.1e} over "
                          f"{nb} blocks, a = 5e-4 and -1e-3")
    check("lockstep [mu(0.9), mu(0.99)] == separate single-mu runs",
          worst_lock == 0.0, f"max |a_hat diff| = {worst_lock:.1e}")
    s_ss = 4.59e-5 * np.sqrt(0.01 / 1.99)
    check("general formula RampDoppler(a, 0) vs run_loop (float-level "
          "signal differences propagated through the loop)",
          np.all(worst_g < 0.01 * s_ss),
          f"max |a_hat diff| = {worst_g[0]:.1e} noiseless, "
          f"{worst_g[1]:.1e} at 4 dB (vs sigma_ss = {s_ss:.2e} at the WP)")


# ---------------------------------------------------------------------------
def gl_integral(fn, t, breaks=(), n_sub=4000, order=20):
    """int_0^t fn(s) ds, composite Gauss-Legendre, split at `breaks`."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.unique(np.concatenate([[0.0, t], [b for b in breaks
                                                 if 0 < b < t]]))
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = np.linspace(lo, hi, max(2, int(n_sub * (hi - lo) / t) + 2))
        a_, b_ = sub[:-1, None], sub[1:, None]
        s = 0.5 * (b_ - a_) * x[None, :] + 0.5 * (b_ + a_)
        total += float(np.sum(0.5 * (b_ - a_) * w[None, :] * fn(s)))
    return total


def check2():
    log("\n## 2. A(t) closed form vs numerical integral of a(t)")
    rng = np.random.default_rng(5)
    kt = np.sort(np.concatenate([[-3.0], rng.uniform(0, 120, 25), [130.0]]))
    kv = rng.uniform(-1e-3, 1e-3, kt.size)
    profiles = {
        "Constant(-7e-4)": (ConstantDoppler(-7e-4), ()),
        "Ramp(-8e-4, 5.4e-5/s)": (RampDoppler(-8e-4, 5.4e-5), ()),
        "Sinusoid(1e-4, 5e-4, 8.4 Hz, 0.7)":
            (SinusoidalDoppler(1e-4, 5e-4, 8.4, 0.7), ()),
        "Sinusoid(0, 5e-4, 0.0265 Hz)": (SinusoidalDoppler(0, 5e-4, 0.0265),
                                         ()),
        "FlyBy(5e-4, 36.9, 9.23)": (FlyByDoppler(5e-4, 36.9, 9.23), ()),
        "FlyBy(5e-4, 3.69, 0.923)": (FlyByDoppler(5e-4, 3.69, 0.923), ()),
        "PiecewiseLinear(27 random knots)":
            (PiecewiseLinearDoppler(kt, kv), tuple(kt)),
        "Callable(3e-4 cos(0.3 t))":
            (CallableDoppler(lambda t: 3e-4 * np.cos(0.3 * t),
                             lambda t: 1e-3 * np.sin(0.3 * t)), ()),
    }
    ts = (0.003, 0.5, 6.0, 37.0, 74.0, 120.0)
    worst = 0.0
    for name, (prof, br) in profiles.items():
        errs = [abs(prof.A(np.array([t]))[0]
                    - gl_integral(prof.a, t, br, n_sub=20000)) for t in ts]
        e = max(errs)
        worst = max(worst, e)
        log(f"   {name:34s} max |A - int a| = {e:.1e} s  (carrier phase "
            f"{2 * np.pi * 1e9 * e:.1e} rad)")
    check("A(t) closed form, all profiles, t up to 120 s", worst < 1e-13,
          f"worst {worst:.1e} s = {2 * np.pi * 1e9 * worst:.1e} rad of "
          "carrier phase (requirement ~1e-13 s)")


# ---------------------------------------------------------------------------
def check3():
    log("\n## 3. Continuity")
    p0 = base()
    for name, prof in (("Sinusoid", SinusoidalDoppler(0, 5e-4, 2.0)),
                       ("FlyBy", FlyByDoppler(5e-4, 0.05, 0.02))):
        sig = lp.LongSignal(p0, seed=21)
        whole = rx_nominal_profile(sig, 6 * N_C, prof, noiseless=True,
                                   start_sample=3 * N_C)
        parts = np.concatenate([rx_nominal_profile(
            sig, N_C, prof, noiseless=True, start_sample=(3 + b) * N_C)
            for b in range(6)])
        d = np.max(np.abs(whole - parts))
        check(f"one record vs block by block ({name}, 6 blocks)", d == 0.0,
              f"max |diff| = {d:.1e}")
    p = base(4.0)
    sig = OCDMSignal(p, n_symbols=6 * N_C // p.N_sym + 10, seed=22)
    prof = RampDoppler(-2e-4, 1e-2)
    whole = rx_nominal_profile(sig, 4 * N_C, prof)
    parts = np.concatenate([rx_nominal_profile(sig, N_C, prof,
                                               start_sample=b * N_C)
                            for b in range(4)])
    d = np.max(np.abs(whole - parts))
    check("one record vs block by block (noisy OCDMSignal, ramp)", d == 0.0,
          f"max |diff| = {d:.1e}")
    prof = SinusoidalDoppler(2e-4, 5e-4, 0.5)
    mus = [0.1 / KD0, 0.01 / KD0]
    one = run_loop_profile(base(), prof, mu=mus, n_blocks=240,
                           snrs=(float("inf"), 4.0), seed=1001)
    a0 = one.a_hat[..., 0]
    seg_a, cur = [a0[..., None]], a0
    for s0 in (0, 80, 160):
        r = run_loop_profile(base(), prof, mu=mus, n_blocks=80,
                             snrs=(float("inf"), 4.0), seed=1001,
                             start_block=s0, a_hat0=cur)
        seg_a.append(r.a_hat[..., 1:])
        cur = r.a_hat[..., -1]
    d = np.max(np.abs(np.concatenate(seg_a, axis=-1) - one.a_hat))
    check("loop in 3 segments vs one run (lockstep 2 mu x 2 SNR)", d == 0.0,
          f"max |a_hat diff| = {d:.1e} over 240 blocks")


# ---------------------------------------------------------------------------
def _charB(seed):
    sys.path.insert(0, str(ROOT))
    import study_characteristic as sc
    return sc.trial((5e-4, seed))


def check4():
    log("\n## 4. Old results unchanged")
    files = [f"ocdm_doppler/{f}" for f in PROTECTED]
    diff = subprocess.run(["git", "diff", "--stat", "HEAD", "--"] + files,
                          capture_output=True, text=True, cwd=ROOT).stdout
    check("git diff of the protected modules is empty", diff.strip() == "",
          "no changes" if not diff.strip() else diff.strip())
    d_loop = subprocess.run(["git", "diff", "-U0", "HEAD", "--",
                             "ocdm_doppler/loop.py"],
                            capture_output=True, text=True, cwd=ROOT).stdout
    changed = [ln for ln in d_loop.splitlines()
               if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    only_msg = all(("run_loop takes a constant a" in ln
                    or "loop_tv.run_loop_profile" in ln
                    or "time-varying a needs" in ln
                    or "Doppler); not allowed" in ln) for ln in changed)
    check("loop.py: only the NotImplementedError text changed",
          only_msg and len(changed) == 4,
          f"{len(changed)} changed lines: " + " | ".join(
              ln.strip()[:70] for ln in changed))

    ref = np.load(ROOT / "results/char_raw/a_+0.000500.npz")["e"]
    with Pool(4) as pool:
        new = np.array(pool.map(_charB, [1000 + s for s in range(64)]))
    d = np.max(np.abs(new - ref))
    check("Part B raw data recomputed (a = 5e-4, 64 blocks, 9 SNR, 2 lag "
          "sets)", d == 0.0, f"max |e diff| = {d:.1e}")

    raw = np.load(ROOT / "results/loop_raw/lag0_a+5e-04.npz")
    snrs = tuple(float(s) for s in raw["snrs"])
    mu = float(raw["mu"])
    nb = 300
    r_old = lp.run_loop(base(), 5e-4, mu=mu, n_blocks=nb, snrs=snrs,
                        seed=1000)
    r_new = run_loop_profile(base(), ConstantDoppler(5e-4), mu=mu,
                             n_blocks=nb, snrs=snrs, seed=1000)
    d1 = np.max(np.abs(r_old.a_hat - raw["a_hat"][0, :, :nb + 1]))
    d2 = np.max(np.abs(r_new.a_hat - raw["a_hat"][0, :, :nb + 1]))
    check("Part D raw trajectory recomputed with loop.run_loop (first 300 "
          "blocks, 9 SNR)", d1 == 0.0, f"max |a_hat diff| = {d1:.1e}")
    check("... and with loop_tv.run_loop_profile(ConstantDoppler)",
          d2 == 0.0, f"max |a_hat diff| = {d2:.1e}")


# ---------------------------------------------------------------------------
def staircase(a0, rate, T):
    """Per-block constant Doppler equal to the block means of
    RampDoppler(a0, rate): a = a0 + rate T (k + 1/2) in block k, with the
    exact (continuous, piecewise-linear) integral
    A(t) = a0 t + rate T^2 k^2 / 2 + rate T (k + 1/2)(t - k T)."""
    def a_fn(t):
        k = np.floor(t / T)
        return a0 + rate * T * (k + 0.5)

    def A_fn(t):
        k = np.floor(t / T)
        return (a0 * t + 0.5 * rate * T * T * k * k
                + rate * T * (k + 0.5) * (t - k * T))
    return CallableDoppler(a_fn, A_fn)


def _ramp_job(args):
    """Open loop at a_hat = 5e-4 over a ramp centred on a_hat, and the same
    blocks with the staircase (same block means, no in-block change).
    The two time warps differ by < beta T_blk / 8, so both see the same data
    and noise: their difference isolates the in-block Doppler change."""
    seed, nb, beta = args
    a_hat = 5e-4
    rate = beta / T_BLK
    a0 = a_hat - 0.5 * nb * beta
    out = []
    for prof in (RampDoppler(a0, rate), staircase(a0, rate, T_BLK),
                 ConstantDoppler(a_hat)):
        r = run_loop_profile(base(), prof, mu=0.0, n_blocks=nb,
                             a_hat0=a_hat, seed=seed)
        out.append((r.e[0], r.a_bar))
    return out


def check5():
    log("\n## 5. Detector on a steep ramp (noiseless, open loop)")
    # beta = r g sigma_ss with r = 10, g = 0.01, sigma_ss = 3.25e-6
    beta = 10 * 0.01 * 4.59e-5 * np.sqrt(0.01 / 1.99)
    nb, R, a_hat = 400, 4, 5e-4
    with Pool(4) as pool:
        res = pool.map(_ramp_job, [(1000 + r, nb, beta) for r in range(R)])
    e_r = np.concatenate([x[0][0] for x in res])
    ab_r = np.concatenate([x[0][1] for x in res])
    e_s = np.concatenate([x[1][0] for x in res])
    ab_s = np.concatenate([x[1][1] for x in res])
    e_c = np.concatenate([x[2][0] for x in res])

    def stat(v):
        return v.mean(), v.std() / np.sqrt(v.size)

    res_r = stat(e_r / KD0 - (ab_r - a_hat))
    res_s = stat(e_s / KD0 - (ab_s - a_hat))
    res_c = stat(e_c / KD0)
    d = stat((e_r - e_s) / KD0)
    log(f"   beta = {beta:.3e} per block ({beta / T_BLK:.3e} 1/s), "
        f"{R} x {nb} blocks, a_bar - a_hat in "
        f"[{-0.5 * nb * beta:.1e}, {0.5 * nb * beta:.1e}]; block means of "
        f"ramp and staircase agree to {np.max(np.abs(ab_r - ab_s)):.1e}")
    log(f"   ramp:      mean(e/K_d0 - (a_bar - a_hat)) = {res_r[0]:+.2e} "
        f"+/- {res_r[1]:.1e}")
    log(f"   staircase: mean(e/K_d0 - (a_bar - a_hat)) = {res_s[0]:+.2e} "
        f"+/- {res_s[1]:.1e}")
    log(f"   constant a = a_hat: mean(e/K_d0)          = {res_c[0]:+.2e} "
        f"+/- {res_c[1]:.1e}   (the constant-a detector bias, Step 4)")
    check("in-block Doppler change: paired ramp - staircase",
          abs(d[0]) < max(3 * d[1], 0.01 * beta),
          f"mean (e_ramp - e_stair)/K_d0 = {d[0]:+.2e} +/- {d[1]:.1e} "
          f"({d[0] / d[1]:+.1f} sigma), vs in-block change beta = "
          f"{beta:.1e}")


def main():
    t0 = time.time()
    log("# Step 5, Part A.4: verification of the time-varying Doppler code")
    log("")
    log("Generated by `verify_time_varying.py`.")
    check1a()
    check1b()
    check2()
    check3()
    check4()
    check5()
    log(f"\nSUMMARY: {len(FAILS)} failed"
        + (": " + ", ".join(FAILS) if FAILS else " -- all checks passed")
        + f"  [{time.time() - t0:.0f}s]")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "verify_time_varying.md").write_text(
        "\n".join(OUT) + "\n", encoding="utf-8")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
