# Project context for Claude Code

This file exists so a fresh Claude Code session in this repo has the full
background without re-deriving it. Read this before doing anything else here.

## Who this is for

Yogev Drori, EE master's student at BGU (non-thesis track, 3-credit project),
specializing in classical signal processing / communications (modem
algorithms: channel, CFO and baud-rate estimation, synchronization, signal
generation). Works in MATLAB and Python. Strong preference for classical,
model-based approaches over deep learning / black-box ML.

Advisor: Ron Dabora (BGU ECE).

## The project

Ron proposed a Doppler-estimation project for OCDM (Orthogonal Chirp
Division Multiplexing) signals, based on two papers:

- **[OCDM]** Z. Tan, *OCDM Signal Analysis* — the signal model, the
  closed-form ambiguity function (AF), and a Section-4 curve-fitting
  algorithm that estimates the Doppler scaling factor `a` by matching the
  empirical AF magnitude against the analytical one over a grid of
  candidate `a` values.
- **[CBSFS]** M. Kumar and R. Dabora, *A Novel Sampling Frequency Offset
  Estimation Algorithm for OFDM Systems Based on Cyclostationary
  Properties*, IEEE Access 7, 2019 — a cyclostationarity-based SFO
  estimator (CB-SFS), reused here to estimate the Doppler scaling factor
  from cycle-frequency shifts.

Zikun Tan (a previous student) ran both estimators and found: the
curve-fitting approach (Section 4 of [OCDM]) performs badly (NMSE ~4.9,
essentially uninformative), while CB-SFS is far more accurate (NMSE ~0.004
in one 20 dB test case vs curve-fit's ~1.49). Ron's take, after seeing this:
the curve-fitting approach is likely resolution-limited/ill-conditioned, and
he suggested moving to a **closed-loop (tracking-loop) detector** instead —
conceptually like a PLL/DLL/Gardner timing-recovery loop: compensate,
discriminate the sign of the residual error, correct, repeat. This avoids
needing an absolute, well-conditioned cost surface (which curve-fitting
needs and doesn't have) — the loop only needs the *sign* of the error and
integrates over time for precision.

## Why curve-fitting fails (established, not hypothesis)

The Doppler scaling factor `a` is small in this regime (~1e-3 to 5e-4). The
AF's shape (a Dirichlet kernel in the lag `tau`) stretches by a factor
`(1+a)`, i.e. by ~0.05% for `a=5e-4` — far below any practical lag
resolution. The curve-fitting cost surface over candidate `a` is therefore
essentially flat: structurally ill-conditioned, not a bug or a
too-relaxed assumption. This was confirmed by reproducing the empirical AF
and comparing it to the analytical AF (see `verify_infrastructure.py`,
check 4) — they match to <5%, so the failure is in the *inversion*
(candidate-`a` fitting), not in the forward model.

CB-SFS instead works in the cycle-frequency (`alpha`) domain, where the
`k`-th harmonic shifts by `k * a * alpha_1` — an effect that *grows* with
the harmonic index `k`, giving real leverage that the `tau` domain lacks.

## Plan agreed with Ron

Before proposing the closed-loop detector, build a from-scratch simulation
(Ron explicitly prefers this over reusing Zikun's code) to reproduce the
baseline results, so the proposal to Ron is grounded in a working repro
rather than just re-reading the papers:

1. **Step 1 (done)** — simulation infrastructure: signal model, analytical
   AF, CB-SFS. See "Status" below.
2. **Step 2 (done)** — implement the Section-4 curve-fitting estimator on
   top of Step 1 and reproduce Zikun's key numbers: noiseless NMSE ~4.9,
   and the noisy case where CB-SFS strongly outperforms curve-fitting.
3. **Step 3 (done)** — build an early-late discriminator S-curve as a
   proof-of-concept for the loop approach.
4. **Step 4 (next)** — present the reproduction + loop proposal to Ron.
5. **Step 5** — full closed-loop detector development (the actual project
   deliverable).

## Status: Steps 1, 2 and 3 complete (Step 1 details below)

Package `ocdm_doppler/` (this directory), pure NumPy, no third-party deps
beyond `numpy`. 10/10 sanity checks pass (`python verify_infrastructure.py`;
check 6 is the discriminator).

| module | contents |
|---|---|
| `params.py` | `SystemParams` — one frozen dataclass fully describes an experiment; `preset_noiseless_31jan()` / `preset_noisy_22jan()` mirror the two results documents exactly |
| `signal_model.py` | `OCDMSignal`: transmitter `X(t)`, single-path Doppler receiver `Y(t)` ([OCDM] Eqn. 3), deterministic reproducible block noise |
| `analytical_af.py` | `Pi`, `Theta`, `Gamma`, `c_Y`, `af_magnitude` ([OCDM] Eqns. 8-12) |
| `cbsfs.py` | segmented CAF (27), autocorrelated CAF (28), cost function (29), harmonic-windowed peak location, Doppler estimate (33)-(34) |

Full design rationale, findings, and open questions for Ron: see
`README.md` in this directory — read it, it has details not repeated here
(e.g. why `Delta=0` must be in the CB-SFS lag set, why the `alpha≈0` skirt
must be excluded from peak search, the `N_seg` accuracy sweep, the
harmonic-leverage-vs-decay tradeoff table, and the open question about
whether the AF-magnitude average should take `|.|` inside or outside the
sum).

### Step 2 (new files; Step-1 modules untouched)

| file | contents |
|---|---|
| `af_estimation.py` | results docs' Step 2(a)–(b): time selection, lag grid, empirical AF, window detection. Places the time origin at received symbol K so the ±(N_avg-1)/2 average never runs into negative symbol indices. `WindowedAF.analytical` evaluates at `t_sel` (not `t_abs`) — required for any candidate `a` other than the true one. |
| `curve_fit.py` | results docs' Step 3: `|A0|^2_hat = 4 sigma_hat^2/sigma_D^2`, `tau_p0_hat` from the window edge, MSE fit over `N_a = 2^18` candidates (vectorised in chunks). |
| `../make_figures.py` | every results-doc figure + curve-fit cost surface → `../figures/`, numbers in `figures_log.txt` |
| `../mc_curvefit.py` | Monte-Carlo tables (ours next to the docs') → `../results/mc_curvefit.csv/.md`; incremental, `--resume` |

Outcome: the documents are reproduced — figures match, and the curve-fit
Doppler NMSE is 0.3–4 (docs 0.5–4.9) vs CB-SFS 1e-2–1e-5; noiseless 31-Jan
NMSE 4.24 (docs 4.91) with 107/128 estimates on a grid boundary. The
README's *Step 2 results* section has the table and the three mechanisms
(flat monotone cost → boundary; window-edge `tau_p0` ties the fit to
`a_sr`, making the cost jump there; `|A0|^2_hat` biased by
`4 P_n/sigma_D^2`). The figures also showed that the AF modulus is taken
**outside** the average (README open question 1).

Missing (see README *What is missing, and why*): Monte-Carlo for Setting 5
(`N_is = 64`; runs kept getting killed by the low-memory reaper before a
single ~25-min S5 case finished — resume with `--resume --cases 22jan
--settings 5`), and the docs' 4096-trial bias run (Yogev chose to skip it;
the 128-trial biases are in `results/mc_curvefit.md`). Yogev is writing an
interim summary from the committed state.

### Step 3 (new files only; nothing in Steps 1–2 modified)

| file | contents |
|---|---|
| `discriminator.py` | early-late discriminator on the CB-SFS cost (29): `predicted_alpha`, `early_late` (vectorised over `a_hat`, one `cost_function` call), `s_curve`, `error_from_costs`. **Sign: `e > 0` ⇒ `a_hat` too small; loop update `a_hat += mu e`.** Default `delta = 0.25/N_seg`, the value selected by the study (lowest σ_e/K_d while covering the prior range). |
| `../study_scurve.py` | S-curves (a = 5e-4 N_mc 64, a = 5e-3 N_mc 16 as an offset-only check; noiseless/20/10 dB; δ ∈ {0.25, 0.5, 0.75}/N_seg), metrics, τ-domain contrast, 4 figures `fig_scurve_*.png`, `results/scurve.*`, raw costs `results/scurve_raw/*.npz` (reuse with `--resume`; `--replot` redraws tables and figures from them without any computation). All alphas of one realisation lie on one grid (h = 1/(64 N_seg)), so it is one `cost_function` call per realisation. |
| `../verify_infrastructure.py` | new check 6 (checks 1–5 untouched) |

Outcome (δ = 0.25/N_seg, a = 5e-4): zero crossing at the truth (bias
≤ 2.6e-6, below its standard error), K_d = 174 at every SNR, linear over
a_min..a_max, pull-in ±9.9e-3 (= main-lobe nulls, ±N_sym/N_seg), σ_e/K_d =
2.5e-5 / 2.7e-5 / 4.6e-5 (noiseless / 20 / 10 dB) ≈ CB-SFS RMSE on the same
data — i.e. no steady-state gain over CB-SFS (by design; the loop's case is
tracking + 2 evaluations per update). False-lock points at ±1.41e-2 and
±2.41e-2 (sidelobe peaks ±1.5/N_seg, ±2.5/N_seg): harmless for
|a| < 1e-3, relevant for a_max = 1e-2 (initialise from CB-SFS). τ domain:
the curve-fit cost has a minimum near the truth in 1 of 64 realisations.
Not done by design: a loop simulation over blocks (Step 5).

Practical notes:
- Long background Monte-Carlo runs launched from Claude Code have been
  killed repeatedly by the idle-session low-memory reaper (not by the
  scripts, ~300 MB/process); PyCharm alone used ~4.7 GB of the 15.7 GB.
  Use `--resume`, run in chunks in the foreground (`study_scurve.py
  --max-new`), or run in the user's own terminal.
- `cbsfs.cost_function` throughput is memory-bandwidth-bound: ~9 s per
  realisation for 437 alphas × 24000 samples × 16 lags; 4 processes give
  little more than 2.
- Windows PowerShell 5.1 mangles double quotes inside `git commit -m`
  here-strings; commit with `-F <message file>`.

`study_cbsfs_accuracy.py` is the sweep script that produced the numeric
tables in the README (record length `M`, segment length `N_seg`, harmonic
index `k`). Not part of the package; run standalone from the parent
directory.

## Design decisions to preserve (do not silently change)

- **No resampling anywhere.** [OCDM] Eqn. (3) is a closed form for `Y(t)`,
  so it's evaluated directly at whatever instants a caller asks for.
- **Two separate sampling grids.** AF estimation uses the fine `T_s` grid
  (`OCDMSignal.rx_at_grid`); CB-SFS uses the receiver's nominal rate
  (`OCDMSignal.rx_nominal`, which does *not* quantize onto the `T_s` grid —
  doing so introduces a spurious rate error of the same order as the
  Doppler itself). Do not merge these.
- **`cbsfs.cost_function` takes an arbitrary array of cycle frequencies**,
  not necessarily a uniform grid. This is intentional: the planned
  early-late discriminator (Step 3) will call it with exactly two `alpha`
  values straddling a predicted harmonic. Nothing in `cbsfs.py` should need
  to change for that.
- Everything in `ocdm_doppler/` is meant to be a stable base for Steps 2-5
  — Yogev's explicit instruction was to implement Step 1 so that later
  steps build *on top of* it without modifying it. New estimators
  (curve-fitting, the loop) should live in new files that import from this
  package, not edit it, unless a genuine bug is found.

## Working style notes

- Yogev prefers classical/model-based methods; avoid steering the project
  toward deep learning unless he asks.
- He works in both MATLAB and Python; this codebase is Python, but he may
  ask for MATLAB translations of specific pieces for visualization or
  cross-checking.
- He tends to ask for the *reasoning* behind a result (why something fails,
  what a design choice buys you), not just the result — keep explanations
  grounded in the math from [OCDM]/[CBSFS], not hand-wavy.
- Language: conversation with him elsewhere has been in Hebrew; code,
  comments and this file are in English to match the existing codebase.
