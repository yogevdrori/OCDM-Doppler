"""
Monte-Carlo NMSE / normalised bias of the Section-4 curve-fitting estimator.

Reproduces the result tables of the results documents:

  31jan   31-Jan Exp. 1: noiseless, a_sr = a, 31 lags inside the main lobe
  22jan   22-Jan Settings 1-5 at 10 / 20 dB: CB-SFS Doppler NMSE and the
          AF-magnitude NMSEs for |A0|^2, tau_p0 and Doppler
  dense   22-Jan p. 3: N_lag = 2^8 lags inside the known window, 20 dB
  bias    22-Jan p. 3: normalised bias, Setting 1, 20 dB, N_mc = 2^12

a_sr for 22-Jan comes from CB-SFS with the settings of mc_cbsfs_22jan.py
(N_c = 24000, N_seg = 2048, harmonic 1, +/-0.05 alpha_1, 128 points).

Results are appended to results/mc_curvefit.csv as each case finishes, so an
interrupted run loses only the case in progress (--resume skips finished
cases). results/mc_curvefit.md is rewritten from the CSV at the end, with the
documents' values next to ours. Seeds are fixed (1000 + trial index), so the
numbers are reproducible.

Usage (from this directory):
    python mc_curvefit.py                          # everything (~4 h)
    python mc_curvefit.py --cases 31jan 22jan --settings 1 --n-mc 32
    python mc_curvefit.py --resume                 # continue a killed run
    python mc_curvefit.py --report                 # only rebuild the .md
"""

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ocdm_doppler import cbsfs, preset_noiseless_31jan, preset_noisy_22jan
from ocdm_doppler import af_estimation as afe
from ocdm_doppler import curve_fit as cf

OUT = Path(__file__).parent / "results"
CSV_PATH = OUT / "mc_curvefit.csv"
MD_PATH = OUT / "mc_curvefit.md"

N_C, N_SEG, K = 24000, 2048, 1
SETTINGS_22JAN = {1: dict(rho=2 ** 5), 2: dict(rho=2 ** 6),
                  3: dict(rho=2 ** 5, a_max=1e-2),
                  4: dict(rho=2 ** 6, a_max=1e-2),
                  5: dict(rho=2 ** 6, N_is=64)}
QUANTITIES = ("cbsfs", "A0_sq", "tau_p0", "doppler")

# Values reported in the results documents: NMSE of
# (CB-SFS Doppler, |A0|^2, tau_p0, AF-magnitude Doppler).
DOC_NMSE = {
    ("22jan", 1, 10): (0.064549, 0.095286, 0.99307, 1.1718),
    ("22jan", 1, 20): (0.0044046, 0.0017935, 0.030366, 1.4881),
    ("22jan", 2, 10): (0.064549, 0.098729, 1.6874, 1.3339),
    ("22jan", 2, 20): (0.0044046, 0.0020082, 0.030881, 1.6436),
    ("22jan", 3, 10): (0.00065131, 0.098414, 1.8472, 0.59513),
    ("22jan", 3, 20): (4.4443e-5, 0.0018077, 0.03312, 0.5729),
    ("22jan", 4, 10): (0.00065131, 0.097723, 1.5082, 0.57558),
    ("22jan", 4, 20): (4.4443e-5, 0.0019098, 0.10499, 0.4933),
    ("22jan", 5, 10): (0.064549, 0.096492, 59.7181, 7.3534),
    ("22jan", 5, 20): (0.0044046, 0.002373, 19.918, 8.4816),
    ("dense", 1, 20): (0.0070776, 0.0019013, 0.00096926, 4.772),
}
DOC_BIAS_S1_20DB = (0.0012, 0.0308, 0.0930, 0.6735)   # N_mc = 2^12
DOC_31JAN = (4.9102, -1.0138)                         # (NMSE, norm. bias)


# ---------------------------------------------------------------------------
# One Monte-Carlo trial per experiment type
# ---------------------------------------------------------------------------
def trial_31jan(seed):
    p = preset_noiseless_31jan()
    a = p.a_true_eff
    sig, t0 = afe.make_signal(p, seed=seed)
    w = afe.estimate_windowed_af(
        sig, t0, a_sr=a,
        t_candidates=(0.0, p.T_sym / (2.0 * (1.0 + a))),
        lag_half_width=p.T_is / (p.N_is * (1.0 + a)))
    # window = whole lag set, Gamma = 1 on it: tau_p0 not identifiable from
    # the window edge, so the true value is used
    fit = cf.fit_doppler(w, p, tau_p0=p.tau_p0)
    return a, fit.A0_sq, fit.tau_p0, fit.a_hat


def params_22jan(kind, setting, snr):
    p = preset_noisy_22jan(snr).with_(**SETTINGS_22JAN[setting])
    if kind == "dense":
        p = p.with_(N_lag=2 ** 8)
    return p


def trial_22jan(args):
    """kind "22jan": lag grid over (-T_is/(1+a_min), T_is/(1+a_min)) with
    window detection; kind "dense": N_lag lags inside the known window."""
    kind, setting, snr, seed = args
    p = params_22jan(kind, setting, snr)
    M = N_C // p.N_sym
    sig, t0 = afe.make_signal(p, n_extra_symbols=M + 2, seed=seed)
    y = sig.rx_nominal(M * p.N_sym)
    ak, _, _ = cbsfs.locate_harmonic(y, p, K, n_alpha=128,
                                     half_width_factor=0.05, N_seg=N_SEG)
    a_sr = ak / K / p.alpha_1_syn - 1.0

    t_cand = (0.0, p.T_sym / (3.0 * (1.0 + p.a_max)))
    lags = None
    if kind == "dense":
        # t'_1 = 0 lies in the guard interval for these parameters, so the
        # variance test selects t'_2; the known window is taken there
        lo, hi = afe.true_window(t0 + t_cand[1], p)
        lags = np.linspace(lo, hi, p.N_lag + 1)[1:]          # (lo, hi]
    w = afe.estimate_windowed_af(sig, t0, a_sr, t_candidates=t_cand,
                                 lags=lags)
    fit = cf.fit_doppler(w, p)
    return a_sr, fit.A0_sq, fit.tau_p0, fit.a_hat


# ---------------------------------------------------------------------------
# Statistics and result files
# ---------------------------------------------------------------------------
def _stats(est, true):
    err = np.asarray(est, dtype=float) - true
    return float(np.mean(err ** 2) / true ** 2), float(err.mean() / true)


def _case_key(kind, setting, snr, n_mc):
    if kind == "31jan":
        return "31jan"
    if kind == "bias":
        return f"bias-S{setting}-{snr:g}dB-N{n_mc}"
    return f"{kind}-S{setting}-{snr:g}dB"


def _done_keys():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="") as f:
        return {row["case"] for row in csv.DictReader(f)}


def _append_row(row: dict):
    OUT.mkdir(exist_ok=True)
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)


def _row(key, kind, setting, snr, n_mc, r, truth, extra="", seconds=0.0):
    row = dict(case=key, kind=kind, setting=setting, snr_db=snr, n_mc=n_mc)
    for j, q in enumerate(QUANTITIES):
        if truth[j] is None:
            row[f"nmse_{q}"], row[f"bias_{q}"] = "", ""
        else:
            row[f"nmse_{q}"], row[f"bias_{q}"] = _stats(r[:, j], truth[j])
    row["note"] = extra
    row["seconds"] = round(seconds)
    return row


def _fmt(x):
    return "–" if x in ("", None) else f"{float(x):.4g}"


def write_report():
    """results/mc_curvefit.md from the CSV, with the documents' values."""
    with open(CSV_PATH, newline="") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    L = ["# Curve-fitting estimator: Monte-Carlo results",
         "",
         "Generated by `mc_curvefit.py` from `mc_curvefit.csv`. Each cell is "
         "**ours** / doc (the results document's value). Seeds 1000 + trial "
         "index. CB-SFS: N_c = 24000, N_seg = 2048, harmonic 1 (the documents "
         "do not state their CB-SFS parameters).",
         ""]

    r = rows.get("31jan")
    if r:
        L += ["## 31-Jan, Experiment 1 (noiseless, a_sr = a)", "",
              "| N_mc | Doppler NMSE | Doppler norm. bias | note |",
              "|---|---|---|---|",
              f"| {r['n_mc']} | **{_fmt(r['nmse_doppler'])}** / "
              f"{DOC_31JAN[0]} | **{_fmt(r['bias_doppler'])}** / "
              f"{DOC_31JAN[1]} | {r['note']} |", ""]

    hdr = ("| setting | σ_D²/P_n (dB) | CB-SFS Doppler | \\|A0\\|² | τ_p0 | "
           "Doppler (AF magnitude) |")
    sections = [("22jan", "## 22-Jan, Settings 1–5: NMSE (N_mc = {n})"),
                ("dense", "## 22-Jan p. 3: N_lag = 2^8 inside the known "
                          "window, NMSE (N_mc = {n})")]
    for kind, title in sections:
        sel = [v for v in rows.values() if v["kind"] == kind]
        if not sel:
            continue
        sel.sort(key=lambda v: (int(v["setting"]), float(v["snr_db"])))
        L += [title.format(n=sel[0]["n_mc"]), "", hdr,
              "|---|---|---|---|---|---|"]
        for v in sel:
            ref = DOC_NMSE.get((kind, int(v["setting"]),
                                int(float(v["snr_db"]))), (None,) * 4)
            cells = [f"**{_fmt(v['nmse_' + q])}** / {_fmt(d)}"
                     for q, d in zip(QUANTITIES, ref)]
            L.append(f"| {v['setting']} | {float(v['snr_db']):g} | "
                     + " | ".join(cells) + " |")
        L.append("")

    L += ["## Normalised bias", "",
          "| case | N_mc | CB-SFS Doppler | \\|A0\\|² | τ_p0 | Doppler |",
          "|---|---|---|---|---|---|"]
    for v in sorted(rows.values(), key=lambda v: v["case"]):
        if v["kind"] == "31jan":
            continue
        ref = (DOC_BIAS_S1_20DB if v["kind"] == "bias" else (None,) * 4)
        cells = [f"**{_fmt(v['bias_' + q])}**"
                 + (f" / {_fmt(d)}" if d is not None else "")
                 for q, d in zip(QUANTITIES, ref)]
        L.append(f"| {v['case']} | {v['n_mc']} | " + " | ".join(cells) + " |")
    L.append("")
    MD_PATH.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cases", nargs="*",
                    choices=("31jan", "22jan", "dense", "bias"),
                    default=("31jan", "22jan", "dense", "bias"))
    ap.add_argument("--settings", nargs="*", type=int,
                    default=sorted(SETTINGS_22JAN),
                    choices=sorted(SETTINGS_22JAN))
    ap.add_argument("--snr", nargs="*", type=float, default=[10.0, 20.0])
    ap.add_argument("--n-mc", type=int, default=128)
    ap.add_argument("--n-mc-bias", type=int, default=2 ** 12)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--resume", action="store_true",
                    help="skip cases already in results/mc_curvefit.csv")
    ap.add_argument("--report", action="store_true",
                    help="only rebuild results/mc_curvefit.md")
    args = ap.parse_args()

    if args.report:
        write_report()
        return
    if not args.resume and CSV_PATH.exists():
        CSV_PATH.unlink()
    done = _done_keys()

    # cheap cases first; Setting 5 (4x slower CB-SFS) and the bias run last
    s22 = [k for k in sorted(args.settings) if k != 5]
    jobs = []
    if "31jan" in args.cases:
        jobs.append(("31jan", 0, float("inf"), args.n_mc))
    if "22jan" in args.cases:
        jobs += [("22jan", k, snr, args.n_mc) for k in s22 for snr in args.snr]
    if "dense" in args.cases:
        jobs.append(("dense", 1, 20.0, args.n_mc))
    if "22jan" in args.cases and 5 in args.settings:
        jobs += [("22jan", 5, snr, args.n_mc) for snr in args.snr]
    if "bias" in args.cases:
        jobs.append(("bias", 1, 20.0, args.n_mc_bias))

    t0 = time.time()
    with Pool(args.procs) as pool:
        for kind, k, snr, n_mc in jobs:
            key = _case_key(kind, k, snr, n_mc)
            if key in done:
                print(f"skip {key} (already in CSV)", flush=True)
                continue
            ts = time.time()
            seeds = [1000 + s for s in range(n_mc)]
            if kind == "31jan":
                p = preset_noiseless_31jan()
                r = np.array(pool.map(trial_31jan, seeds))
                n_lo = int(np.sum(np.isclose(r[:, 3], p.a_min_eff, rtol=1e-3)))
                n_hi = int(np.sum(np.isclose(r[:, 3], p.a_max, rtol=1e-3)))
                row = _row(key, kind, "", "inf", n_mc, r,
                           (None, None, None, p.a_true_eff),
                           f"a_hat at a_min: {n_lo}, at a_max: {n_hi}",
                           time.time() - ts)
            else:
                trial_kind = "dense" if kind == "dense" else "22jan"
                p = params_22jan(trial_kind, k, snr)
                r = np.array(pool.map(trial_22jan, [(trial_kind, k, snr, s)
                                                    for s in seeds]))
                truth = (p.a_true_eff, abs(p.A_p0) ** 2, p.tau_p0,
                         p.a_true_eff)
                n_lo = int(np.sum(np.isclose(r[:, 3], p.a_min_eff, rtol=1e-3)))
                n_hi = int(np.sum(np.isclose(r[:, 3], p.a_max, rtol=1e-3)))
                row = _row(key, kind, k, snr, n_mc, r, truth,
                           f"a_hat at a_min: {n_lo}, at a_max: {n_hi}",
                           time.time() - ts)
            _append_row(row)
            print(f"{key:>24}  " + "  ".join(
                f"{q}: NMSE={_fmt(row['nmse_' + q])} bias={_fmt(row['bias_' + q])}"
                for q in QUANTITIES if row["nmse_" + q] != "")
                + f"  ({row['note']})  [{time.time() - t0:.0f}s]", flush=True)
            write_report()


if __name__ == "__main__":
    main()
