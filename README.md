# OCDM Doppler estimation

Clean-room implementation (from scratch, per Ron's request) of the signal
model and estimators behind the AF-magnitude Doppler estimator for OCDM,
and a reproduction of the previous results (22-Jan and 31-Jan 2026
documents), as the baseline for the proposed closed-loop detector.

Sources:
- **[OCDM]** Z. Tan, *OCDM Signal Analysis* — signal model (Sec. 2–3),
  analytical AF (Eqns. 8–12), Section-4 algorithm.
- **[CBSFS]** M. Kumar and R. Dabora, *A Novel Sampling Frequency Offset
  Estimation Algorithm for OFDM Systems Based on Cyclostationary
  Properties*, IEEE Access 7, 2019 — Eqns. (27)–(34).
- *Algorithm Steps & Results*, 22-Jan and 31-Jan 2026 (the results
  documents, "the documents" below).

## Status

| step | what | state |
|---|---|---|
| 1 | infrastructure: signal model, analytical AF, CB-SFS | done |
| 2 | Section-4 curve-fitting estimator + reproduction of the documents' figures and tables | done (see *Step 2 results*) |
| 3 | early-late discriminator S-curve (proof of concept for the loop) | next |
| 4 | present reproduction + loop proposal to Ron | |
| 5 | closed-loop detector | |

## Layout

Package `ocdm_doppler/` (NumPy only):

| module | contents |
|---|---|
| `params.py` | `SystemParams` — one frozen object fully describes an experiment; presets mirror the two results documents |
| `signal_model.py` | OCDM transmitter `X(t)`, single-path Doppler receiver `Y(t)` ([OCDM] Eqn. 3), deterministic block noise |
| `analytical_af.py` | `Pi`, `Theta`, `Gamma`, `c_Y`, `af_magnitude` ([OCDM] Eqns. 8–12) |
| `cbsfs.py` | segmented CAF (27), autocorrelated CAF (28), cost function (29), peak location, Doppler estimate (33)–(34) |
| `af_estimation.py` | documents' Step 2(a)–(b): time selection (Eqn. 1), lag grid, empirical AF, window detection |
| `curve_fit.py` | documents' Step 3: `|A0|^2` and `tau_p0` estimates, MSE curve fit over `N_a` candidate Doppler values |

Scripts (run from this directory; need `matplotlib` for the figures):

| script | output |
|---|---|
| `verify_infrastructure.py` | 9 sanity checks (9/9 passing) |
| `make_figures.py` | `figures/*.png` + `figures/figures_log.txt` — every figure of the documents, the curve-fitting cost surface, a [CBSFS] Fig. 5-style plot (~9 min; `--a-sr genie` skips CB-SFS) |
| `mc_curvefit.py` | `results/mc_curvefit.csv` / `.md` — Monte-Carlo NMSE and bias tables of the documents, ours next to theirs; incremental, `--resume` after an interruption |
| `mc_cbsfs_22jan.py` | Monte-Carlo NMSE of CB-SFS alone |
| `study_cbsfs_accuracy.py` | CB-SFS sweeps over `M`, `N_seg`, harmonic index (tables below) |

## Design decisions worth knowing

**No resampling.** [OCDM] Eqn. (3) is a closed form for `Y(t)`, so the
receiver is evaluated directly at whatever instants an estimator asks for.
This avoids interpolation error entirely and is faster than generating a
long oversampled waveform.

**The guard interval is a zero gap, not a cyclic prefix.** In [OCDM]
`g(t) = 1` only on `[0, T_is)`, so each OCDM symbol is followed by `T_gi`
of silence. This is what makes `|Y[k]|^2` periodic and gives CB-SFS its
strongest cyclostationary signature.

**Two distinct sampling grids.** AF estimation uses the fine grid `T_s`
(the CT approximation); CB-SFS uses the receiver's nominal rate
`T_samp_ns`. These are kept separate on purpose — quantising the nominal
samples onto the `T_s` grid introduces a spurious rate error of order
`1/rho ≈ 1e-3`, i.e. the same size as the Doppler being measured.

**`cost_function` takes arbitrary cycle frequencies.** Not just a uniform
grid. The planned early-late discriminator evaluates the cost at two
`alpha` values either side of the predicted harmonic, so it can call this
directly and nothing in `cbsfs.py` needs to change.

**Time origin at received symbol K.** The documents average over
`i = -(N_avg-1)/2 .. +(N_avg-1)/2` around `t' = 0`, i.e. they assume the
signal exists at negative times. `af_estimation.make_signal` places the
receiver's time origin at the start of received symbol `K > (N_avg-1)/2`
(`t_origin = K T_sym / (1+a)`), which shifts `(1+a)t - tau_p0` by exactly
`K T_sym`: the within-symbol position, `Gamma` and `|Pi|` are unchanged.
The analytical AF for a *candidate* `a` must be evaluated at the receiver
time `t_sel`, not at `t_origin + t_sel` — the origin shift is exact only
for the true `a` (`WindowedAF.analytical` does this).

**CB-SFS settings for `a_sr`** (not stated in the documents): `N_c = 24000`
samples, `N_seg = 2048`, harmonic 1, window `±0.05 alpha_1` with 128 points,
parabolic refinement. Figures and Monte-Carlo use the same settings.

## Step 2 results: the curve-fitting estimator

The estimator follows the documents exactly (Steps 1–3): `a_sr` from
CB-SFS; `t_sel` by the variance test; `N_lag` lags, window of highest power;
`|A0|^2_hat = 4 sigma_hat^2_Y(t_sel) / sigma_D^2`,
`tau_p0_hat = (t_sel - tau_win_r)(1 + a_sr)`; `N_a = 2^18` candidates in
`(a_min, a_max)`, minimum MSE between the windowed empirical AF and
`|Pi * Gamma|`.

**Monte-Carlo NMSE, 128 trials** (`results/mc_curvefit.md` has the full
tables incl. normalised bias). **Ours** / documents:

| case | SNR (dB) | CB-SFS | `|A0|^2` | `tau_p0` | **Doppler (curve fit)** |
|---|---|---|---|---|---|
| 31-Jan (noiseless, `a_sr = a`) | ∞ | – | – | – | **4.24** / 4.91 (bias −0.84 / −1.01) |
| S1 | 10 | 0.0081 / 0.065 | 0.099 / 0.095 | 0.21 / 0.99 | **0.78** / 1.17 |
| S1 | 20 | 0.0030 / 0.0044 | 0.0020 / 0.0018 | 0.082 / 0.030 | **1.00** / 1.49 |
| S2 (`rho = 2^6`) | 10 | 0.0069 / 0.065 | 0.097 / 0.099 | 0.37 / 1.69 | **0.81** / 1.33 |
| S2 | 20 | 0.0029 / 0.0044 | 0.0020 / 0.0020 | 0.079 / 0.031 | **0.99** / 1.64 |
| S3 (`a_max = 1e-2`) | 10 | 6.4e-5 / 6.5e-4 | 0.097 / 0.098 | 0.38 / 1.85 | **0.84** / 0.60 |
| S3 | 20 | 3.0e-5 / 4.4e-5 | 0.0021 / 0.0018 | 0.036 / 0.033 | **0.27** / 0.57 |
| S4 (`a_max = 1e-2`, `rho = 2^6`) | 10 | 8.6e-5 / 6.5e-4 | 0.099 / 0.098 | 0.23 / 1.51 | **0.89** / 0.58 |
| S4 | 20 | 3.3e-5 / 4.4e-5 | 0.0020 / 0.0019 | 0.034 / 0.105 | **0.29** / 0.49 |
| S5 (`N_is = 64`) | 10 | not run — see *What is missing* | | | |
| S5 | 20 | not run — see *What is missing* | | | |
| dense window (`N_lag = 2^8` in the known window) | 20 | 0.0030 / 0.0071 | 0.0020 / 0.0019 | 7.5e-10 / 9.7e-4 | **2.21** / 4.77 |

The documents' 4096-trial bias run (S1, 20 dB) was not repeated (see
*What is missing*); the 128-trial biases are in `results/mc_curvefit.md`
(S1, 20 dB: Doppler +0.63 vs the documents' +0.67).

**The documents' conclusion is reproduced: the curve-fitting Doppler
estimate is essentially uninformative** (NMSE 0.3–4, vs 1e-2 – 1e-5 for
the CB-SFS estimate it starts from), and a denser lag grid makes it worse.

### Why — three mechanisms

**1. The cost surface is flat and monotone, so the estimate goes to a grid
boundary.** `|Pi|` stretches by `(1+a)`, i.e. the lag axis of the analytical
curve changes by only 0.2 % across the whole range `(a_min, a_max)`
(`1 ± 1e-3`) — comparable to or below the residual
model mismatch of the empirical AF (0.6 % of the peak even noiseless,
finite `N_avg`). The MSE is therefore dominated by that mismatch and is
nearly flat and monotone in `a` (`figures/fig_curvefit_cost.png`: it varies
by ~1–2 % over the range in the noisy runs, ~20 % noiseless, with no
minimum near the true `a`). The sign of the mismatch decides which end
wins: in the 31-Jan run 107 of 128 estimates sit exactly at `a_min` (49)
or `a_max` (58), which is what the documents' numbers imply too (half at
each end gives NMSE = 0.5·3² + 0.5·1² = 5, bias −1). In the 22-Jan runs
most estimates sit at `a_max` (87–107 of 128 for S1–S4, except S3/S4 at
20 dB — see 2).

**2. The window-edge `tau_p0` estimate ties the fit to `a_sr`.** With
`tau_p0_hat = (t_sel - tau_win_r)(1 + a_sr)`, the analytical `Gamma` for the
candidate `a = a_sr` has its upper edge exactly on the last windowed lag:
candidates below `a_sr` drop that lag (`Gamma = 0`), candidates above keep
it. The cost therefore has a jump at `a_sr`, and the minimum is either
there or at a boundary. Where it lands at `a_sr`, the "curve-fit" estimate
is really the CB-SFS estimate — this is why S3/S4 at 20 dB look better
(NMSE ≈ 0.28; only 12–14 of 128 at `a_max`): not information from the AF.

**3. `|A0|^2_hat` carries the noise power.** `sigma_hat^2_Y` includes `P_n`,
so `|A0|^2_hat` is biased by `4 P_n / sigma_D^2` (relative `+0.31` at 10 dB,
`+0.031` at 20 dB) — NMSE 0.098 / 0.0020, matching the documents to ~10 %
in every setting. This scales the analytical curve and adds to the
mismatch in (1).

**Denser lags make it worse** (dense window: Doppler NMSE 2.2 vs 1.0):
`tau_p0` becomes essentially exact, but the additional lags are mostly in
the sidelobes, where the empirical AF has a noise floor
`≈ sigma_D^2 |A0|^2 / (4 sqrt(N_avg))` and the analytical AF has nulls —
more weight on the part of the curve that carries no Doppler information
and the most mismatch (documents p. 3 figure; `fig_22jan_dense_window.png`).

**Where our numbers differ from the documents:** CB-SFS is 1.5–10× more
accurate here (their CB-SFS parameters are not stated; `N_seg` alone moves
the error by orders of magnitude, see below), and at 10 dB our `tau_p0`
NMSE is lower (0.2–0.4 vs 1.0–1.8). The Doppler conclusions do not depend
on either.

### What is missing, and why

Everything in the documents has been reproduced except the following:

| item | status | why | how to complete |
|---|---|---|---|
| **Monte-Carlo, Setting 5** (`N_is = 64`), 10 and 20 dB | **not run** | Four background runs launched from Claude Code were killed by its idle-session low-memory safeguard (the system as a whole ran low on memory; the script uses ~300 MB per process). S5 is ~4× slower per trial than S1–S4 (CB-SFS has 64 lags instead of 16), so one 128-trial S5 case takes ~20–30 min and never finished before a kill. Results are saved per completed case, so nothing else was lost. | `python mc_curvefit.py --resume --cases 22jan --settings 5 --procs 1` in a normal terminal (~40–60 min); it appends to `results/mc_curvefit.csv` and rebuilds the `.md`. |
| **Normalised-bias run, N_mc = 2^12** (S1, 20 dB; documents p. 3) | **skipped (decision)** | ~2.5 h of compute, too long to survive the memory kills above. Our 128-trial biases are already in `results/mc_curvefit.md` and are close to the documents' 4096-trial values for S1, 20 dB (Doppler +0.63 vs +0.67, `|A0|^2` +0.032 vs +0.031; `tau_p0` +0.099 vs +0.093; CB-SFS +0.0032 vs +0.0012). | `python mc_curvefit.py --resume --cases bias` (~2.5 h). |

What *is* available for Setting 5: the single-realisation figure
`figures/fig_22jan_setting5.png` (it matches the documents' broken case:
analytical AF flat at ~0.005, empirical AF noise) and its curve-fit numbers
in `figures/figures_log.txt` (`a_hat_mag` at `a_min` at both SNRs; at
20 dB `tau_p0_hat` = 1.49e-6 vs 0.5e-6 true). A 2-trial smoke test gave
Doppler NMSE 9 and `tau_p0` NMSE 7.8 at 20 dB (documents: 8.48, 19.9) —
consistent with the documents but far too few trials to report.

Not started yet (by plan): Steps 3–5 (early-late S-curve, proposal to Ron,
closed-loop detector).

## Step 1 findings (CB-SFS)

**`Delta = 0` must be in the CB-SFS lag set.** Omitting it (using
`Delta = 1..N_is`) destroys the estimator — the cost function becomes
essentially noise. The guard interval makes the instantaneous power
periodic, and that is the dominant cyclostationary feature.

**The `alpha ≈ 0` skirt must be excluded from peak search.** The
stationary component produces a peak at `alpha = 0` roughly 27× taller
than the cycle-frequency peaks; any threshold relative to the global
maximum then rejects the real peaks. The search range starts at
`0.5 * alpha_1`.

**Harmonic-windowed search beats generic peak detection.** Assigning "the
k-th detected local maximum is harmonic k" is fragile: Dirichlet sidelobes
are picked up as separate peaks. `locate_harmonic` searches a window
centred on `k * alpha_1` instead.

**`N_seg` dominates CB-SFS accuracy** (noiseless, M = 1200, harmonic 2):

| `N_seg` | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| rel. error | −275% | −124% | +21% | +6.6% | +1.4% |

**Higher harmonics do *not* give net leverage here.** Measuring harmonic
`k` and dividing by `k` shrinks the error as `1/k`, but the harmonics decay
fast, so the peak-SNR penalty cancels the gain (noiseless, `N_seg = 2048`,
4 seeds):

| `k` | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| peak height | 3.1e−3 | 1.9e−3 | 8.7e−4 | 2.0e−4 | 6.6e−5 |
| rel. error | 0.7% | 3.4% | 1.3% | 1.3% | 2646% |

`k = 1..4` are usable; `k >= 5` breaks down. **Relevant to the loop
design: the discriminator should sit on the fundamental or the 2nd
harmonic, not a high one.**

## Open questions to raise with Ron

1. **Is the magnitude inside or outside the AF average?** The documents
   write

   ```
   c_hat_Y(t_sel, tau_k) = sum_i | Y(...) Y*(...) | / N_avg
   ```

   with the modulus *inside* the sum. That estimator converges to
   `E{|Y(t)||Y(t-tau)|}`, which is strictly positive and has **no nulls** —
   it cannot match the analytical `|Pi|`, which does. Writing
   `| sum_i Y Y* | / N_avg` (modulus outside) is what converges to `|c_Y|`.

   *Resolved by the figure reproduction:* the published figures match only
   the modulus-**outside** version. In the 31-Jan noiseless figure the
   empirical AF falls to ~0.02 at the edge of the main lobe; modulus-inside
   would stay near 0.32·π/4 ≈ 0.25 there. The noisy sidelobe floor
   (~0.01–0.02) is the finite-`N_avg` residual `≈ 0.32/sqrt(N_avg)`. So the
   document text is most likely a typo. `af_estimation` defaults to
   `"outside"`. To confirm with Ron.

2. **CB-SFS parameters are not stated** in either document — `N_seg`, the
   number of observed symbols `M`, `N_T`, and the cycle-frequency grid
   resolution. As the table above shows, `N_seg` alone moves the error by
   two orders of magnitude. We use the settings under *Design decisions*,
   which give 1.5–10× lower CB-SFS NMSE than the documents report.

3. **Peak refinement.** [CBSFS] searches a grid of `N_seg` points; that
   resolution is far coarser than the Doppler values here, so sub-grid
   parabolic refinement is enabled by default. Was it used originally?

4. **`T_s` reference differs between the documents** — the 22-Jan document
   uses `(1 + a_min)`, the 31-Jan one `(1 + a_true)`. Both are supported
   via `ts_reference`.

5. **Negative time indices.** The averaging sum runs over
   `i = -(N_avg-1)/2 .. +(N_avg-1)/2`, so `t_sel + i * T_sr` goes negative
   for the first half — where no signal exists. *Handled* by the time-origin
   shift (see *Design decisions*); presumably the original simulation had
   symbols at negative times.

6. **`tau_p0` for the 31-Jan experiment.** There the lags span only the
   main lobe, so the detected "window" is the whole lag set and
   `tau_p0_hat` from its edge is undefined. We use the true `tau_p0`
   (`Gamma = 1` on all those lags, so the fit does not depend on it). What
   did the original use?
