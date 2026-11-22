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
4. **Step 4 (done)** — proposal presented; Ron's feedback implemented
   (task file `../tasks/step4_detector_and_loop.md`): SNR working point,
   open-loop characteristic vs the true a for Δ = {0} and {0..15}, loop
   convergence time, closed-loop simulation vs SNR.
5. **Step 5 (done)** — tracking a time-varying Doppler a(t) and the choice
   of loop pole (task file `../tasks/step5_time_varying_doppler.md`), built
   in new modules without changing `signal_model.py`.
6. **Step 6 (paused)** — second-order tracking loop vs first-order
   (task file `../tasks/step6_second_order_loop.md`): Parts A and B done,
   C–E and the robustness check pending.
7. **Step 7 (done)** — other multi-basis waveforms (OTFS) and multipath
   (task file `../tasks/step7_otfs_multipath.md`; advisor's first
   priority). Next priorities from the advisor: complexity reduction and
   the comparison with the literature (`../docs/literature.md`).

## Status: Steps 1–5 and 7 complete, Step 6 paused after Parts A–B (Step 1 details below)

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

### Step 4 (new files only; Steps 1–3 untouched)

| file | contents |
|---|---|
| `loop.py` | first-order loop `run_loop` on consecutive blocks of one long signal (`LongSignal`: lazily generated data symbols; per-block CSCG noise from a seeded stream; several SNRs in lockstep); `n95`, `steady_state_var`. Constant a only: a time-varying a raises `NotImplementedError` (needs an integrated Doppler in `signal_model.py`). |
| `../study_snr_definitions.py` | Part A: SNR definitions, simulated SER, working point |
| `../study_characteristic.py` | Part B: open-loop characteristic (â = 0, true a swept), both lag sets on the same blocks; the noisy block = noiseless + √P_n × the same `_BlockNoise` stream OCDMSignal uses (asserted equal) |
| `../study_loop.py` | Part C (`--part C`) and Part D (`--part D`; `--max-real` splits one case into per-realisation part files) |

Outcome:
- Working point: BPSK SER = 10 % at σ_D²/P_n = 4.0 dB (simulated 4.01,
  theory 4.04); Es/N0 = |A0|²/4 · σ_D²/P_n (−4.89 dB).
- K_d0 = 175.5 for both lag sets; constant down to ≈ 0 dB; at −5 dB 0.55
  ({0}) and 0.13 ({0..15}). **Δ = {0} is better at every SNR** (σ_e/K_d
  3.2× lower at the working point, 7× lower closed-loop NMSE) and costs
  20.6 RM/sample vs 300. Recommend Δ = {0}.
- n95 = 298 blocks = 1.79 s at p = 0.99, N_c = 24000. K_d independent of
  N_c, but N_c = 6000 gives a +3.8 % closed-loop bias (too short).
- Closed loop at the working point: RMSE 3.4–5.2e-6, |bias| ≤ 4.2e-6
  (≤ 2 % of a); variance matches (σ_e/K_d)² g/(2 − g) to within 0.84–1.35.
- Unexplained, to investigate: a +1.1e-5 detector offset at a = 0 exactly
  (synchronous sampling); a small, a- and SNR-dependent closed-loop bias.
- The a = 0 offset is reproduced by `../check_detector_offset.py` (~30 s)
  and `../study_characteristic.py --near-zero` (saved data).

Decisions Yogev took during Step 4 (all runs above use them):
- Part B: full spec (79 values of a × 64 blocks × 9 SNRs × 2 lag sets),
  rather than fewer blocks on the wide grid.
- Part D: option A, R = 8 realisations × L = 2000 steady-state blocks
  (NMSE to ±11 %), transient 1633 blocks; D.7 run as a reduced comparison
  (Δ = {0..15}, a ∈ {1e-4, 1e-3}, 20 dB / WP / 0 dB, R = 4, L = 1000).
- The last D.7 case, killed by the memory reaper, was redone in two halves
  (`--max-real 2`) rather than skipped.
- Yogev's `.docx` reports are not committed.

Runtimes (compute): Part A < 1 min; Part B ~20 min (4 processes); Part C
~6 min; Part D ~72 min (8 processes — the loop blocks are CPU-bound and
scale better than Step 3's memory-bound evaluations), against the ~45 min
estimated and approved; ~10 min more lost to the killed D.7 run. Total
~100 min.

### Step 5 (new files only; "add, do not change")

**Rule (Yogev, not negotiable): add, do not change.** New capability goes
into new files that reuse the existing classes. Do not modify `params.py`,
`signal_model.py`, `analytical_af.py`, `cbsfs.py`, `af_estimation.py`,
`curve_fit.py`, `discriminator.py`, or the existing functions of `loop.py`.
Allowed: `__init__.py` exports, docs, and (Step 5) the text of
`loop.run_loop`'s `NotImplementedError`. If something seems to need a change
to an existing module, stop and ask with the reason and the minimal change.
`verify_time_varying.py` check 4 confirms the protected modules are
unchanged (git diff) and that committed raw data recomputes bit-identically.

| file | contents |
|---|---|
| `doppler_profile.py` | `DopplerProfile` (a, closed-form A = ∫a, block_mean); `ConstantDoppler`, `RampDoppler`, `SinusoidalDoppler`, `FlyByDoppler`, `PiecewiseLinearDoppler`, `CallableDoppler`, `as_profile`. A(t) must be closed form (carrier phase 2π·1e9·A): all forms are written without cancellation (≤ 5.6e-17 s error). |
| `signal_tv.py` | `rx_profile`, `rx_nominal_profile`: u = t + A − τ_p0, phase 2π((f_Tx − f_Rx)t + f_Tx A) + Δφ, reusing `sig._decompose`, `sig._chirp_sum`, `sig._noise`; **a `ConstantDoppler` is delegated to `OCDMSignal.rx`** so constant-a results stay bit-identical (the general formula differs from `rx` by the old code's float rounding of (1+a)·f_Tx, and is ~1000× more accurate in phase). |
| `loop_tv.py` | `run_loop_profile`: same signal (`LongSignal`), noise stream and update as `run_loop`; `mu` and `a_hat0` may be arrays (lockstep loops, bit-identical to separate runs); result with `a_bar`, `eps = a_hat − a_bar`. Analytical `H`, `f_3db`, `sigma_ss`, `ramp_lag`, `ramp_rmse`, `optimal_g`, `sine_rmse`. |
| `../verify_time_varying.py` | Part A.4 checks → `../results/verify_time_varying.md` |
| `../study_tracking.py` | `--part B/C/D`; every realisation advanced in checkpointed segments (`results/track_raw/`), `--budget` stops a call cleanly before the 10-min limit, `--resume`, `--seg`, `--replot`, `--quick` |

Outcome (details in README *Step 5 results*):
- Ramp: lag = −β/g (bias-free, from up and down ramps over the same a
  interval); optimal pole simulated = analytical grid choice in all 6
  (r, SNR) cases; WP optimum ≈ 0.9973 / 0.9874 / 0.9425 for r = 0.1 / 1 / 10.
- Sinusoid: gain and phase follow H(z) = g z⁻¹/(1 − p z⁻¹) within ~1 % and 1°.
- Fly-by: RMSE matches the exact linear model within 8 %; the quasi-static
  −β/g fails when 1/g ≈ T_c (p = 0.99, r = 100); no loss of lock.

Decisions Yogev took in Step 5:
- Lockstep μ and checkpointed segments approved; regression check by
  recomputing committed raw data.
- Part B: up AND down ramps for all r (so the a-dependent detector bias
  cancels in the lag); p = 0.998 added for r = 0.1, 1 only; r = 10 uses
  L = 2500 (not 3000) so both directions stay within |a| ≤ 1e-3; optimal
  pole based on the bias-removed RMSE.
- Part C: R = 4, all points kept (no frequency dropped).
- Part D: p = 0.7512 (analytical r = 100 optimum) added; Part B's r = 10
  analytical optimum 0.9425 added (the simulated one, 0.95, already in set).
- The a-dependent detector bias is an open item — **not to be investigated
  in Step 5** (Yogev's instruction).

Runtimes (wall clock, this machine): A ~4 min; B ~120 min (estimated ~56;
contention with 8 processes and low free memory); C ~127 min, of which
~80 min were two stalls (the machine apparently slept / swapped during two
segments) — ~46 min of compute; D ~19 min. Total ~4.5 h wall, ~3.1 h of
compute.

Open items (see the Step 7 note under Step 6's open items: the
SNR-dependent part is DC-line leakage, closed):
- The constant-a detector bias varies with a (+1.3e-6 at a = 5e-4 vs
  −0.56e-6 over [4.35, 5.65]e-4; Part B WP ramp bias −1.1 to −1.6e-6, the
  opposite sign to Step 4's +1.0e-6 at a = 5e-4; fly-by excess of 3e-6 to
  1.2e-5 at a ≈ −9e-5). Probably the same phenomenon as the Step 4 bias floor
  and the a = 0 sampling-phase offset.

Practical notes:
- Throughput: with 8 processes the 10–12 lockstep loops per block ran at
  ~160 ms/block/worker (vs ~50 with 2): 8 processes give only ~25 % more
  than 2–4. Use 4. Time a `--quick` run with the real process count before
  estimating.
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
- **Always run Python with the project venv: `.venv\Scripts\python.exe`**
  (from the repo root). The system Python on PATH (3.13) has no
  matplotlib; the venv has numpy 2.5 and matplotlib 3.11.

### Step 6 (in progress; "add, do not change" as in Step 5)

Allowed edits: `__init__.py` exports, `README.md`, this file. No existing
module (including `loop.py`, `loop_tv.py`, `signal_tv.py`,
`doppler_profile.py`) may change; if something seems to need it, stop and
ask.

| file | contents |
|---|---|
| `loop_filters.py` | `LoopFilter` interface (`update`, `get_state`/`set_state`, `copy`, `gains`); `FirstOrderFilter(mu)`; `SecondOrderFilter(mu1, mu2)`: b̂ += μ2 e, â += b̂ + μ1 e; `make_filter(g1, g2, K_d0, …)` (μ = g/K_d0; g2 = 0 → first order). Design: `gains_from_poles`, `gains_critically_damped` (g1 = 1 − p², g2 = (1 − p)²), `gains_from_zeta_wn` (r = e^{−ζω_n}, θ = ω_n√(1−ζ²)), `design_equal_BL` (bisection on ω_n). Analysis: `H2`, `f_3db` (last −3 dB crossing, allows peaking), `B_L` (exact, discrete Lyapunov), `linear_response`, `step_response`, `impulse`, `n95`, `overshoot`, `ss_error_ramp` (0 for 2nd order), `ss_error_parabola` (−γ/g2), `closed_loop_poles(g1, g2, K_d/K_d0)`. `Design(name, g1, g2)` with `.BL`, `.order`, `.filter()`; `first_order_design(p)`, `equal_BL_design(p1, ζ)` (names "1st p=0.95", "2nd cd ~p1=0.95", "2nd z0.707 ~p1=0.95"). |
| `loop_tv2.py` | `run_loop_filters(params, profile, filters, n_blocks, snrs, …, start_block, states)` → `FilterLoopResult` (`a_hat[m,s,i]`, `b_hat`, `e`, `a_bar`, `eps`, final `states`). Same signal, noise stream ([seed, 0xA015E, block]) and detector call as `run_loop_profile`; filters copied per SNR, all in lockstep. |
| `../verify_second_order.py` | Part A checks → `../results/verify_second_order.md` (all pass, ~7 min) |
| `../study_second_order.py` | `--part B` (C/D/E/R to be added). All unfinished realisations of all cases advance together in rounds of `--seg` blocks (one pool task per realisation), each saved to `results/so_raw/<case>__r<k>.npz`; `--budget` stops between rounds; `--resume`, `--replot`, `--quick` (→ `results/quick/`, git-ignored), `--procs`. |
| `../check_bias_phase.py` | Yogev's check → `../results/bias_phase_check.md`: integer shift a·N_c per block freezes the sampling phase and inflates the detector bias; **use non-integer values** (5.1542e-4, −4.8458e-4, 9.8458e-4, 2.6542e-4, 1.1542e-4). |

Outcome so far (details in README *Step 6 results*):
- Part A: first-order filter == Step 5 loop bit for bit; μ2 = 0 == first
  order; lockstep / segments 0; helpers agree with direct recursion; the
  task file's design table reproduced; noiseless ramp r = 10: second-order
  bias-free lag −1.1e-7 ± 0.7e-7 (vs 0), first order −6.63e-6 (vs −6.51e-6).
- Part B (equal-B_L designs for p1 ∈ {0.9, 0.95, 0.99}, ζ = 1 and 0.707):
  steady-state std and bias identical for both orders (std/pred
  0.94–1.14; bias is a detector property, WP −1.0e-6 at +5.15e-4, −1.9e-6
  at −4.85e-4); transients follow the linear model; lock range identical
  for all designs (captured for |ε0| ≤ 1e-2, false lock at ±1.41e-2 from
  1.1e-2); the second-order loop settles 3–3.5× more slowly, and for
  p1 = 0.99 its overshoot tail needs 984–2427 blocks to stay below 5σ_ss.

Decisions Yogev took in Step 6:
- Part C: second-order loops start at â0 = a(0), b̂0 = β (steady state);
  still discard ≥ 900 blocks; drift check = bias-free lag of the up first
  half vs the down second half (same a sub-interval).
- Acquisition: true a kept in range, â0 = a − Δ, Δ ∈ {3, 6, 8, 9, 9.5,
  10, 11}·1e-3, both signs, first-order loops in the same sweep; lock
  criterion max |ε| < 5σ_ss over the last 500 blocks.
- Part D: â0 = a(0), b̂0 = true slope per block at t = 0.
- Part B run size: **option C** (steady state R = 2 × L = 2000;
  acquisition 2000 blocks, R = 1, steps from 0 R = 2) instead of the spec
  (~140 min measured). Consequence: the p1 = 0.99 second-order loops cannot
  meet the 5σ_ss criterion within 2000 blocks; Yogev chose to add a
  "captured" column (max |ε| < 1e-3 and mean ε within 5σ_ss + 25 % of the
  linear tail) rather than rerun longer.
- `docs/literature.md` (Yogev's literature notes) is in the repo; `*.docx`
  and `results/quick/` are git-ignored.

Pending: Part C (ramps r ∈ {1, 10}), Part D (fly-bys), Part E (sinusoid
for the p1 = 0.99 designs), robustness (K_d = 0.549·K_d0 poles; short
−5 dB runs), then README/CLAUDE.md update. Timed `--quick` estimate before
each part; ask above ~35 min.

Open items:
- The a-dependent detector bias (from Step 5) persists: WP bias −1.0e-6 at
  a = +5.15e-4 and −1.9e-6 at −4.85e-4, identical for both loop orders.
  Not to be investigated unless Yogev asks. **SNR-dependent part closed in
  Step 7** (`results/dc_removal.md`): it is leakage of the DC line of |y|²
  (mean power + noise power) through the rectangular N_seg window into the
  two probes. Removing the block mean of |y|² before the CAF
  (`cbsfs_dc.py`) shifts the OCDM bias by +1.1e-6 / +1.9e-6 at the WP and
  +4.3e-6 / +7.2e-6 at 0 dB (paired, se ≤ 5e-8), after which the bias no
  longer depends on N_seg (2048 vs 1920 consistent with 0 at every SNR).
  The small noiseless a-dependent part (±3e-7) is not explained by this.
- σ_e/K_d (4.59e-5 at the WP) was measured at a = 5e-4; the std ratio at
  a = −4.85e-4 (1.07–1.14) suggests it is slightly larger there.

Practical notes (Step 6):
- Throughput depends strongly on free memory: ~80 ms/block/worker (18
  lockstep loops, 3 processes, ~3 GB free) vs 350–500 ms with < 1.5 GB free
  (PyCharm ~4.4 GB). Check free RAM before a run (Yogev's rule: ≥ 3 GB →
  4 processes, else 3; stop and report if a round exceeds ~400
  ms/block/worker or free memory drops below 1 GB).
- One round stalled for 46 min (machine asleep or swapping); a call that
  outlives 10 min moves to the background but still stops cleanly at the
  budget check.

### Step 7 (done; "add, do not change")

Allowed edits: `__init__.py` exports, `README.md`, this file. The code was
tied to OCDM in three places; each is solved with new code:
- the cycle-frequency period `SystemParams.N_sym` → `wf_params.WaveformParams`
  (`N_sym` returns the waveform's mean-power period `P_samp`), passed **only
  to the detector** (identical integer 20 for OCDM);
- `OCDMSignal._chirp_sum` / `LongSignal` → the waveform layer;
- single-path `rx` → the channel layer, with **OCDM single path delegated
  to the old code** (bit-identical, as Step 5 did for constant Doppler).

| file | contents |
|---|---|
| `wf_params.py` | `WaveformParams(SystemParams)` with field `P_samp`; `detector_params(params, P_samp)` |
| `waveforms.py` | `Waveform` specs → per-realisation signals with `tx(u)` (exact at any time), `P_samp`, `mean_power(u)`, `autocorr(t1, t2)` = E[X(t1)X*(t2)]. `OCDMWaveform` wraps `loop.LongSignal`. `OTFSWaveform(M=16, N=16, N_gi=4, guard="symbol"/"frame")`: X_dd (delay l × Doppler k, BPSK, lazy seeded chunks of 256 frames, stream [seed, 0x07F5, chunk]) → `isfft` (unitary: fft over delay, ifft over Doppler, norm="ortho") → x(t) = (1/√M) Σ_m X_tf[m,n] e^{j2π(m − M/2) r/T_is}; `demod` inverts it (fft of y·(−1)^i, then `sfft`). Guard "symbol": N_gi zeros per slot (P_samp 20); "frame": N·N_gi zeros per frame (P_samp 320). Same per-active-sample power σ_D² as OCDM. |
| `channel.py` | `Path(gain, delay [samples, fractional], profile=None)`, `MultipathChannel(paths)` (gains normalised Σ|c|² = 1), `single_path`, `exponential_pdp(D_max, P=4, decay_db=10, rayleigh, rng)`; Y(t) = Σ_p h_p e^{jφ_p(t)} X(t + A_p(t) − τ_p), h_p = A_p0 c_p e^{−j2πf_Tx τ_p}/2, φ_p = 2π((f_Tx − f_Rx)t + f_Tx A_p(t)) + φ_Tx − φ_Rx; `received`, `rx_block`. |
| `channel_norm.py` | `mean_received_power` (exact, incl. cross terms; diagonal only for per-path Doppler), `power_factor` G = P_ref/P_ch, `normalised_params` (A_p0 × √G). Yogev's rule for Part D: every channel at the single-path received power (fractional delays add partly coherently). |
| `loop_wf.py` | `run_loop_wf(params, waveform, channel, profile, filters, …, ks)` (harmonic per filter), `open_loop_e`; same noise stream and lockstep as `run_loop_filters`, which it reproduces bit for bit for OCDM single path. |
| `loop_wf2.py` | `run_loop_det` / `open_loop_det`: as `loop_wf` with a cost function per filter (`dets`). |
| `cbsfs_dc.py` | `cost_function_dc`: cbsfs cost with the block mean of each lag product (|y|²) subtracted before the segmented CAF; `remove_dc=False` = `cbsfs.cost_function` bit for bit. |
| `../verify_step7.py` | Part A checks → `../results/verify_step7.md` (all pass) |
| `../study_otfs.py` | `--part B/C/D` → `../results/otfs_{B,C,D}.md/.csv`, raw `results/otfs_raw/`; pool tasks save per task / realisation; `--resume`, `--replot`, `--budget`, `--procs`, `--quick` |
| `../check_nseg_leakage.py`, `../check_dc_removal.py` | the leakage checks (paired on the same blocks) |
| `../count_ops_lag0.py` | Yogev's explicit real-multiplication count of one Δ = {0} update (495,244 RM = 20.635 RM/sample, bit-exact vs `early_late`) |

Outcome (details in README *Step 7 results*):
- Verification: OCDM through the new layers bit-identical; OTFS demod exact;
  OTFS WP 4.04 dB = theory = OCDM.
- OTFS-symbol = OCDM in K_d, pull-in and closed loop; lower detector
  self-noise (σ_e/K_d 0.41× noiseless, 0.79× WP).
- OTFS-frame: best k = 2 (predicted from the on/off Fourier coefficients);
  ~10× noisier than OCDM; with N_seg = 2048 a large SNR-growing bias.
- **Detector bias = leakage through the rectangular N_seg window**: N_seg a
  multiple of P_samp (1920) reduces it for OTFS-frame; removing the DC
  line (`cbsfs_dc.py`) removes the SNR-dependent part of the OCDM bias
  (open item since Step 4, closed) and makes the bias independent of N_seg.
- Multipath (power-normalised, μ from the single-path K_d): K_d unchanged
  by delay spread; σ_e/K_d at the WP 0.48–1.72× single path (OCDM),
  1.05–1.55× (OTFS-frame up to D_max = 64); always locked. OCDM fixed gains
  D_max = 2, 4: +3e-6 WP bias (one realisation, to be confirmed).
- Multiscale (Δa = 1e-4): the loop follows the stronger path (2:1); equal
  powers: it wanders.

Decisions Yogev took in Step 7:
- Gains normalised per realisation; then (after check 5) every Part D
  channel scaled to the exact single-path received power (G reported).
- PDP: P = 4, fractional delays uniform over D_max, last path −10 dB;
  μ from the single-path K_d (effective pole reported).
- Multiscale: physical (paths drift apart), short (3 s), which path the
  loop follows; plus a 2:1 power case.
- Part C: μ from the noiseless K_d; k = 5 open loop only (c_5 = 0).
- Part D: OTFS-frame N_seg = 1920, k = 2, DC removed; OCDM N_seg = 2048 with
  DC removal as the main detector, the current one on the same blocks;
  run size option B (fixed × 1 + Rayleigh × 2, 100 + 200 + 600 blocks).

Open items after Step 7:
- OTFS-frame at the WP: 16–22 % non-linearity over the prior range (not
  DC; neighbouring harmonics?) and a residual −3e-5 closed-loop bias.
- OCDM noiseless a-dependent bias (±3e-7) not explained by DC leakage.
- (Closed by the follow-ups, `results/step7_followups.md`): the +3e-6 OCDM
  bias in fixed-gain multipath was a single-seed fluctuation (16 batches:
  +0.1e-6 ± 0.5e-6, +0.2e-6 ± 0.7e-6); σ_e/K_d vs delay spread follows the
  k = 1 harmonic strength of the received mean power (≈ strength^−1.5…−1.9
  at the WP); the multiscale behaviour is the capture effect (the loop
  follows the stronger of two unresolved lines).
- **Statistics caveat (audited, `../audit_bias_se.py` →
  `results/bias_se_audit.md`):** the loop output is correlated over
  τ_int ≈ 1/B_L blocks, so the valid std err of a bias is ≈ (σ_e/K_d)/√N.
  Batch means need batches ≥ 10τ; the spread of R = 2 realisation means is
  useless (Step 6 B se were 5× too small in the median, up to 50×). Use
  several seeds, batches ≥ 10τ or the τ_int correction. The DC-leakage
  conclusion (open-loop paired, independent blocks) and the frozen-phase
  conclusion survive the audit; Step 6 B's WP biases drop to ~2σ each.
- Accuracy vs cost (`../study_cost_accuracy.py` → `results/cost_accuracy.md`):
  RM/update = 22 N_c − 16 N_seg + 12 (+1 DC removal); cost ≈ 77–87 M RM/s at
  any (N_c, N_seg), only the duty cycle M reduces it; N_seg is the accuracy
  lever (K_d ∝ N_seg, capture ∝ 1/N_seg). N_c = 48000, N_seg = 4096, M = 4
  matches the current WP accuracy at ~1/4 of the cost.

Practical notes (Step 7):
- **Run with `.venv\Scripts\python.exe`** (see above).
- Multipath block generation dominates the cost (one waveform evaluation
  per path): OCDM 4 paths ~0.1–0.3 s/block, OTFS-frame ~0.29 s/block per
  worker; timings on this machine vary 2–4× between runs.

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
- Everything in `ocdm_doppler/` is meant to be a stable base for Steps 2-6
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

## Working rules for Claude (set by Yogev, apply in every session)

- **Ask before acting.** Before running any command, changing any code or
  file in the repo, or committing/pushing, describe the action and get
  Yogev's approval. Reading and searching are fine without asking. A task
  file or a fully specified request counts as approval for the work it
  describes (including its commit/push step), not for anything beyond it.
- **Ask before long runs.** Estimate the runtime first (time a `--quick`
  run) and ask before any run longer than ~20 min, with options (fewer
  trials, fewer points) and their precision cost.
- **Never restart a job that Claude Code's low-memory safeguard stopped**
  without asking. Report what was stopped and what is saved.
- **Long runs:** foreground only (a call is capped at 10 min; past that it
  moves to the background, where the reaper can kill it). Every study
  script saves incrementally and has `--quick`, `--resume`, `--max-new`
  (and `--max-real` in `study_loop.py`) so a run can be split into
  < 10-min calls; `--replot` redoes tables and figures from saved data.
- **Report honestly:** say when a run took longer than estimated, when a
  result is unexpected (and investigate it), and do not tune parameters to
  make a figure look good.
- Do not commit Yogev's own documents (e.g. `.docx` reports) unless asked.
- The repo is private (`yogevdrori/OCDM-Doppler`); keep it private.
