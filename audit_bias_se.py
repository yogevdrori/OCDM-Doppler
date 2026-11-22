"""
Audit of the standard errors of every closed-loop bias reported so far
(Yogev, 22.11). No new simulation: everything is recomputed from the saved
raw data; where none was saved, the inflation factor is estimated from a
comparable run.

The loop output is strongly correlated: for a loop with unit DC gain driven
by white detector noise, the integrated autocorrelation time of a_hat is
tau_int = 1 / B_L (first order: (2 - g)/g = 199 blocks for g = 0.01), so the
std err of a time mean is

    se = std(eps) * sqrt(tau_int / N) = sqrt(B_L) sigma_n * sqrt(1/(B_L N))
       = (sigma_e / K_d) / sqrt(N)        (N = R * L blocks)

Methods compared per bias figure:
  old      the method of the original script (realisation means: valid but
           with R - 1 degrees of freedom; batch means of b blocks: biased low
           when b < ~10 tau)
  tau_int  data-driven: integrated autocorrelation time from the pooled
           autocovariance of eps, Sokal's automatic window (c = 5)
  model    std(eps) * sqrt(1 / (B_L N)), B_L from the lag-1 autocorrelation
           (first order: B_L = (1 - rho)/(1 + rho)) or from the design
  batch    batch means with b >= 10 tau (tau = 1/g), only if >= 4 batches
The corrected se is tau_int when its window is < L/4, else the model value.
Significance flags: |bias| / se crosses 2 (or 3) between old and corrected.

Output: results/bias_se_audit.md.  Usage: .venv\\Scripts\\python.exe
audit_bias_se.py [--quick]
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RES = ROOT / "results"
ROWS: list[dict] = []


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def acov(x: np.ndarray) -> np.ndarray:
    """Pooled (biased) autocovariance of x[R, L] about the grand mean."""
    x = np.atleast_2d(x)
    xc = x - x.mean()
    L = x.shape[1]
    nfft = 1 << int(math.ceil(math.log2(2 * L)))
    F = np.fft.rfft(xc, nfft, axis=1)
    c = np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, :L].mean(0) / L
    return c


def tau_sokal(x: np.ndarray, c: float = 5.0):
    """(tau_int, window W, var) by Sokal's automatic windowing."""
    a = acov(x)
    rho = a / a[0]
    tau = 1.0
    for W in range(1, rho.size):
        tau = 1.0 + 2.0 * rho[1:W + 1].sum()
        if W >= c * tau:
            return float(tau), W, float(a[0])
    return float(tau), rho.size, float(a[0])


def lag1(x):
    a = acov(x)
    return float(a[1] / a[0])


def se_real(x):
    x = np.atleast_2d(x)
    return float(x.mean(1).std(ddof=1) / math.sqrt(x.shape[0])) \
        if x.shape[0] > 1 else float("nan")


def se_batch(x, b):
    x = np.atleast_2d(x)
    nb = x.shape[1] // b
    if nb < 1:
        return float("nan"), 0
    m = x[:, :nb * b].reshape(x.shape[0], nb, b).mean(-1).ravel()
    return (float(m.std(ddof=1) / math.sqrt(m.size)) if m.size > 1
            else float("nan")), int(m.size)


def audit(x, *, old_method, old_b=None, tau_design=None, B_L=None):
    """x[R, L]: the series whose mean is the reported bias."""
    x = np.atleast_2d(x)
    R, L = x.shape
    N = R * L
    bias = float(x.mean())
    if old_method == "real":
        se_old, b_old = se_real(x), L
    else:
        se_old, b_old = se_batch(x, old_b)[0], old_b
    tau_i, W, var = tau_sokal(x)
    se_tau = math.sqrt(var * tau_i / N)
    rho = lag1(x)
    BL = B_L if B_L is not None else (1 - rho) / (1 + rho)
    se_mod = math.sqrt(var / (BL * N)) if BL > 0 else float("nan")
    tau_d = tau_design if tau_design is not None else 1.0 / max(1 - rho,
                                                                1e-9)
    b10 = int(math.ceil(10 * tau_d))
    se_b10, nb10 = se_batch(x, b10)
    if nb10 < 4:
        se_b10 = float("nan")
    # corrected se: batch means of >= 10 tau if >= 8 batches; else tau_int
    # if its window is reliable (W < L/4); else the larger of model and
    # tau_int (the detector noise is not always white: conservative)
    if nb10 >= 8 and np.isfinite(se_b10):
        se_new, src = se_b10, "batch≥10τ"
    elif W < L / 4:
        se_new, src = se_tau, "τ_int"
    else:
        se_new, src = max(se_mod, se_tau), ("model" if se_mod >= se_tau
                                            else "τ_int*")
    return dict(bias=bias, R=R, L=L, b_old=b_old, tau=tau_d,
                b_over_tau=b_old / tau_d, se_old=se_old, tau_int=tau_i,
                W=W, se_tau=se_tau, se_mod=se_mod, se_b10=se_b10, nb10=nb10,
                se_new=se_new, new_from=src,
                z_old=abs(bias) / se_old if se_old > 0 else float("nan"),
                z_new=abs(bias) / se_new if se_new > 0 else float("nan"))


def flag(r):
    zo, zn = r["z_old"], r["z_new"]
    if not (np.isfinite(zo) and np.isfinite(zn)):
        return ""
    for lev in (3, 2):
        if (zo >= lev) != (zn >= lev):
            return (f"**{'lost' if zo >= lev else 'gained'} {lev}σ**")
    return ""


def row(section, case, snr, x, **kw):
    r = audit(x, **kw)
    r.update(section=section, case=case, snr=snr)
    ROWS.append(r)
    return r


def snr_s(x):
    return "∞" if np.isinf(x) else f"{x:g}"


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def step4():
    """Step 4 Part D (results/loop_sim.md): R = 8 x L = 2000, pole 0.99,
    realisation means."""
    for f in sorted((RES / "loop_raw").glob("*.npz")):
        z = np.load(f)
        a, D = float(z["a"]), int(z["discard"])
        for s, snr in enumerate(z["snrs"]):
            x = z["a_hat"][:, s, D + 1:] - a
            row("Step 4 Part D (loop_sim)", f"{f.stem}", float(snr), x,
                old_method="real", tau_design=100.0)


def step5():
    """Step 5 Part B ramps: bias = (eps_up + eps_down)/2, lag = (eps_up -
    eps_down)/2; R = 8; the old se combined the up / down realisation-mean
    se as if independent."""
    import study_tracking as st
    design = st.part_b_design(False)
    for r, d in design.items():
        eps = {}
        for direction, sgn in (("up", 1), ("down", -1)):
            tr = st.load_case(RES / "track_raw", f"B_r{r:g}_{direction}",
                              d["R"])
            a0 = d["a_c"] - sgn * d["half"]
            prof = st.RampDoppler(a0, sgn * d["beta"] / st.T_BLK)
            tb = np.arange(d["D"] + d["L"]) * st.T_BLK
            a_bar = prof.block_mean(tb, tb + st.T_BLK)
            eps[direction] = (tr[..., :d["D"] + d["L"]] - a_bar)[..., d["D"]:]
        for j, p in enumerate(d["poles"]):
            for s_i, snr in enumerate(st.SNRS):
                up, dn = eps["up"][:, j, s_i], eps["down"][:, j, s_i]
                for kind, x in (("bias", 0.5 * (up + dn)),
                                ("lag", 0.5 * (up - dn))):
                    rr = row("Step 5 Part B (tracking_ramp)",
                             f"r={r:g} p={p:g} {kind}", snr, x,
                             old_method="real", tau_design=1 / (1 - p))
                    rr["se_old"] = 0.5 * math.hypot(se_real(up), se_real(dn))
                    rr["z_old"] = abs(rr["bias"]) / rr["se_old"]


def step6():
    """Step 6 Part B steady state (so_constant): R = 2 x L = 2000,
    realisation means."""
    import study_second_order as so2
    from ocdm_doppler import loop_filters as lf
    d = so2.part_b_design(False)
    des = so2.main_designs()
    for a in so2.A_SS:
        A, _ = so2.load(RES / "so_raw", f"B_ss_a{a:+.4e}", d["R"])
        for m, ds in enumerate(des):
            for s_i, snr in enumerate(so2.SNRS):
                x = A[:, m, s_i, d["D"]:d["D"] + d["L"]] - a
                BL = lf.B_L(ds.g1, ds.g2)
                row("Step 6 Part B (so_constant)", f"a={a:+.2e} {ds.name}",
                    snr, x, old_method="real", tau_design=1 / (2 * BL),
                    B_L=BL)


def step7():
    import study_otfs as so
    dB = so.design_b(False)
    for w in so.WF_B:
        for a in so.A_LOOP:
            nm = f"B_loop_{w}_a{a:+.4e}"
            A = np.array([np.load(RES / "otfs_raw" / f"{nm}__r{r}.npz")
                          ["a_hat"] for r in range(dB["R"])])
            for s, snr in enumerate(so.SNRS_LOOP):
                x = A[:, 0, s, dB["D"]:dB["D"] + dB["L"]] - a
                row("Step 7 Part B (otfs_B)", f"{w} a={a:+.2e}", snr, x,
                    old_method="real", tau_design=100.0)
    for a in so.A_LOOP:
        nm = f"C_loop_OTFS-frame_a{a:+.4e}"
        A = np.array([np.load(RES / "otfs_raw" / f"{nm}__r{r}.npz")["a_hat"]
                      for r in range(dB["R"])])
        for m, k in enumerate((1, 2, 3, 4)):
            for s, snr in enumerate(so.SNRS_LOOP):
                x = A[:, m, s, dB["D"]:dB["D"] + dB["L"]] - a
                row("Step 7 Part C (otfs_C)", f"k={k} a={a:+.2e}", snr, x,
                    old_method="real")
    # Part D: batches of 150 blocks
    dD = so.design_d(False)
    groups = {}
    for w, si, D_max, g, r, seed in so.d_cases(dD):
        groups.setdefault((w, si, D_max, g), []).append(r)
    for (w, si, D_max, g), rs in groups.items():
        A = np.array([np.load(RES / "otfs_raw" /
                              f"D_loop_{so._d_name(w, si, g, r)}.npz")
                      ["a_hat"] for r in rs])
        for di, xdet in enumerate(so.WAVE_D[w]["dets"]):
            for s, snr in enumerate(so.SNRS_D):
                x = A[:, di, s, dD["D"]:dD["D"] + dD["L"]] - so.A_D
                row("Step 7 Part D (otfs_D)",
                    f"{w} {so.DET_NAME[xdet]} D_max={D_max:g} {g}", snr, x,
                    old_method="batch", old_b=150, tau_design=100.0)
    # follow-up 3: all 4 seeds, batches of 150
    for si, D_max in ((0, 0), (1, 2), (2, 4)):
        arrs = [np.load(RES / "otfs_raw" /
                        f"D_loop_{so._d_name('OCDM', si, 'fixed', 0)}.npz")
                ["a_hat"]] + [np.load(RES / "followup_raw" /
                                      f"p3_s{si}_seed{s}.npz")["a_hat"]
                              for s in (1001, 1002, 1003)]
        A = np.array(arrs)
        for di, xdet in enumerate(("dc", "cur")):
            for s, snr in enumerate(so.SNRS_D):
                x = A[:, di, s, dD["D"]:dD["D"] + dD["L"]] - so.A_D
                row("Step 7 follow-up 3 (4 seeds)",
                    f"OCDM {so.DET_NAME[xdet]} D_max={D_max} fixed", snr, x,
                    old_method="batch", old_b=150, tau_design=100.0)


def nseg_closed():
    """N_seg check closed loop: R = 2 x L = 1500, batches of 300; and the
    paired 1920 - 2048 difference."""
    D, L, R = 900, 1500, 2
    snrs = (float("inf"), 4.0, 0.0)
    for j, a in enumerate((5.1542e-4, -4.8458e-4)):
        A = {N: np.array([np.load(RES / "nseg_raw" /
                                  f"ocdm_loop_{j}_{N}_{r}.npz")["a_hat"]
                          for r in range(R)]) for N in (2048, 1920)}
        for s, snr in enumerate(snrs):
            e = {N: A[N][:, s, D:D + L] - a for N in A}
            for nm, x in (("2048", e[2048]), ("1920", e[1920]),
                          ("1920 − 2048 (paired)", e[1920] - e[2048])):
                row("N_seg check, closed loop (nseg_leakage 2b)",
                    f"a={a:+.2e} {nm}", snr, x, old_method="batch",
                    old_b=300, tau_design=100.0)


def open_loop_checks():
    """Open-loop bias figures: are consecutive blocks independent? (tau_int
    of the per-block bias estimate e(a_hat = a)/K_d)."""
    out = []
    D_OFF = 2e-4
    # DC-removal check: e[det, nseg, snr, i, off]
    for j, a in enumerate((5.1542e-4, -4.8458e-4)):
        E = np.concatenate([np.load(RES / "dc_raw" / f"ocdm_{j}_{s}.npz")["e"]
                            for s in range(0, 1500, 125)], axis=3)
        for ni, N in enumerate((2048, 1920)):
            for s, snr in enumerate((float("inf"), 4.0, 0.0, -5.0)):
                b = []
                for di in range(2):
                    e = E[di, ni, s]
                    K = (e[:, 0] - e[:, 2]).mean() / (2 * D_OFF)
                    b.append(e[:, 1] / K)
                for nm, x in (("current", b[0]), ("DC removed", b[1]),
                              ("DC − cur (paired)", b[1] - b[0])):
                    t, W, var = tau_sokal(x[None, :])
                    se_old = x.std(ddof=1) / math.sqrt(x.size)
                    se_new = se_old * math.sqrt(max(t, 1.0))
                    out.append(("DC-removal check (open loop)",
                                f"a={a:+.2e} N_seg={N} {nm}", snr, x.mean(),
                                se_old, t, se_new))
    return out


# ---------------------------------------------------------------------------
def bias_phase_estimate():
    """bias_phase_check.md: no raw data. Its se came from R = 4 realisation
    means of L = 3000 blocks (pole 0.99). The method is valid but has 3
    degrees of freedom; the model se (sigma_e/K_d)/sqrt(R L) is estimated
    from the comparable Step 4 runs (same loop, same detector)."""
    txt = (RES / "bias_phase_check.md").read_text(encoding="utf-8")
    rows = re.findall(r"\| ([+-][\d.e+-]+) \| ([+-][\d.]+) \| ([\d.]+) \| "
                      r"([+-][\d.e+-]+) ± ([\d.e+-]+) \| ([+-][\d.e+-]+) ± "
                      r"([\d.e+-]+) \|", txt)
    # sigma per block of the loop error and B_L from Step 4 (Delta = {0})
    ref = {}
    for f in (RES / "loop_raw").glob("lag0_*.npz"):
        z = np.load(f)
        D = int(z["discard"])
        for s, snr in enumerate(z["snrs"]):
            if float(snr) in (float("inf"), 4.0):
                x = z["a_hat"][:, s, D + 1:] - float(z["a"])
                rho = lag1(x)
                ref.setdefault(float(snr), []).append(
                    math.sqrt(acov(x)[0] / ((1 - rho) / (1 + rho))))
    s_n = {k: float(np.median(v)) for k, v in ref.items()}   # sigma_e/K_d
    N = 4 * 3000
    out = []
    for a, aNc, frac, b0, se0, b4, se4 in rows:
        for snr, b, se in ((float("inf"), b0, se0), (4.0, b4, se4)):
            se_m = s_n.get(snr, float("nan")) / math.sqrt(N)
            out.append((float(a), snr, float(b), float(se), se_m))
    return out, s_n


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Step 4 and the N_seg check only (timing)")
    args = ap.parse_args()
    t0 = time.time()
    step4()
    nseg_closed()
    if not args.quick:
        step5()
        step6()
        step7()
    ol = open_loop_checks()
    bp, s_n = bias_phase_estimate()

    L = ["# Audit of the bias standard errors", "",
         "Generated by `audit_bias_se.py`" + (" --quick" if args.quick else "")
         + ". Method in the script docstring. Columns: L = blocks per "
         "realisation used for the bias; batch = block length of the old "
         "method's averaging unit (realisation means: L); τ = 1/g (design; "
         "second order: 1/(2B_L)); old se = the original method; τ_int = "
         "integrated autocorrelation time of ε from the data (Sokal window "
         "W); corrected se = from τ_int if W < L/4, else the model "
         "std(ε)·√(1/(B_L N)); batch≥10τ = batch means of ≥ 10τ blocks "
         "(only if ≥ 4 batches). Corrected se = batch≥10τ if ≥ 8 batches, "
         "else τ_int if W < L/4, else max(model, τ_int) (marked τ_int* when "
         "the data-driven value is the larger: correlated detector noise). "
         "Flag: |bias|/se crosses 2σ or 3σ.", ""]
    # summary of flags
    flagged = [r for r in ROWS if flag(r)]
    L += ["## Summary", "",
          f"{len(ROWS)} closed-loop bias figures audited; {len(flagged)} "
          "change significance at the 2σ or 3σ level (listed per section "
          "below). Ratio corrected / old se, median per section:", "",
          "| section | figures | old method | median se_new / se_old | "
          "range | flags |", "|---|---|---|---|---|---|"]
    secs = []
    for r in ROWS:
        if r["section"] not in secs:
            secs.append(r["section"])
    for sec in secs:
        rs = [r for r in ROWS if r["section"] == sec]
        ratio = np.array([r["se_new"] / r["se_old"] for r in rs
                          if r["se_old"] > 0])
        meth = ("realisation means (R = %d)" % rs[0]["R"]
                if rs[0]["b_old"] == rs[0]["L"] else
                f"batch means, b = {rs[0]['b_old']}")
        L.append(f"| {sec} | {len(rs)} | {meth} | {np.median(ratio):.2f} | "
                 f"{ratio.min():.2f}–{ratio.max():.2f} | "
                 f"{sum(1 for r in rs if flag(r))} |")
    L.append("")
    for sec in secs:
        L += [f"## {sec}", "",
              "| case | SNR | bias | R × L | batch / τ | old se | τ_int (W) "
              "| corrected se | model se | batch≥10τ se | z old → new | "
              "flag |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in ROWS if r["section"] == sec]:
            L.append(
                f"| {r['case']} | {snr_s(r['snr'])} | {r['bias']:+.2e} | "
                f"{r['R']} × {r['L']} | {r['b_old']}/{r['tau']:.0f} = "
                f"{r['b_over_tau']:.1f} | {r['se_old']:.1e} | "
                f"{r['tau_int']:.0f} ({r['W']}) | {r['se_new']:.1e} "
                f"({r['new_from']}) | {r['se_mod']:.1e} | "
                + (f"{r['se_b10']:.1e} ({r['nb10']})" if np.isfinite(
                    r['se_b10']) else "–")
                + f" | {r['z_old']:.1f} → {r['z_new']:.1f} | {flag(r)} |")
        L.append("")
    L += ["## Open-loop bias figures (DC-removal check)", "",
          "Open loop, â fixed: consecutive blocks have independent data and "
          "noise, so the block-to-block std err is valid if τ_int ≈ 1. "
          "corrected = old·√τ_int.", "",
          "| case | SNR | bias | old se | τ_int | corrected se | z old → new "
          "|", "|---|---|---|---|---|---|---|"]
    for sec, case, snr, b, so_, t, sn in ol:
        L.append(f"| {case} | {snr_s(snr)} | {b:+.2e} | {so_:.1e} | {t:.2f} "
                 f"| {sn:.1e} | {abs(b) / so_:.1f} → {abs(b) / sn:.1f} |")
    L += ["", "## bias_phase_check.md (no raw data saved)", "",
          "R = 4 × L = 3000, pole 0.99, realisation means (batch = 30τ: the "
          "method is valid, but each se has only 3 degrees of freedom — its "
          "own 95 % range is ×0.57 … ×3.7). Model se = (σ_e/K_d)/√(R L) "
          "with σ_e/K_d estimated from the Step 4 loop runs (same loop, same "
          f"detector): noiseless {s_n.get(float('inf'), float('nan')):.2e}, "
          f"4 dB {s_n.get(4.0, float('nan')):.2e}.", "",
          "| a | SNR | bias | reported se | model se | z reported → model | "
          "flag |", "|---|---|---|---|---|---|---|"]
    for a, snr, b, se, se_m in bp:
        zo, zn = abs(b) / se, abs(b) / se_m
        fl = ""
        for lev in (3, 2):
            if (zo >= lev) != (zn >= lev):
                fl = f"**{'lost' if zo >= lev else 'gained'} {lev}σ**"
                break
        L.append(f"| {a:+.4e} | {snr_s(snr)} | {b:+.2e} | {se:.1e} | "
                 f"{se_m:.1e} | {zo:.1f} → {zn:.1f} | {fl} |")
    # Step 6 Part A check 5 (verify_second_order.md): no raw data
    if not args.quick:
        import study_second_order as so2
        from ocdm_doppler import loop_filters as lf
        d = so2.part_b_design(False)
        A, _ = so2.load(RES / "so_raw", f"B_ss_a{so2.A_SS[0]:+.4e}", d["R"])
        des = so2.main_designs()
        L += ["## Step 6 Part A, check 5 (no raw data saved)", "",
              "Noiseless ramp r = 10, R = 4 × L = 1500, bias-free lag = "
              "(ε_up − ε_down)/2, se from realisation means (3 degrees of "
              "freedom). Model se = std(ε)·√(1/(B_L N))/√2 (up and down "
              "independent), std(ε) and B_L from the Part B noiseless steady "
              "state of the same design.", "",
              "| design | reported lag ± se | model se | z reported → model |",
              "|---|---|---|---|"]
        for name, lag, se in (("1st p=0.95", -6.63e-6, 6.4e-8),
                              ("2nd cd ~p1=0.95", -1.09e-7, 6.6e-8)):
            m = [i for i, x in enumerate(des) if x.name == name][0]
            x = A[:, m, 0, d["D"]:d["D"] + d["L"]] - so2.A_SS[0]
            BL = lf.B_L(des[m].g1, des[m].g2)
            se_m = math.sqrt(acov(x)[0] / (BL * 4 * 1500)) / math.sqrt(2)
            pred = -6.51e-6 if name.startswith("1st") else 0.0
            L.append(f"| {name} | {lag:+.2e} ± {se:.1e} (pred {pred:+.2e}) "
                     f"| {se_m:.1e} | {abs(lag - pred) / se:.1f} → "
                     f"{abs(lag - pred) / se_m:.1f} |")
        L.append("")
    L += ["", f"Runtime {time.time() - t0:.0f} s.", ""]
    out = RES / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bias_se_audit.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
