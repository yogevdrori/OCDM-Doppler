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
2. **Step 2 (next)** — implement the Section-4 curve-fitting estimator on
   top of Step 1 and reproduce Zikun's key numbers: noiseless NMSE ~4.9,
   and the noisy case where CB-SFS strongly outperforms curve-fitting.
3. **Step 3 (optional, strengthens the proposal)** — build an early-late
   discriminator S-curve as a proof-of-concept for the loop approach.
4. **Step 4** — present the reproduction + loop proposal to Ron.
5. **Step 5** — full closed-loop detector development (the actual project
   deliverable).

## Status: Step 1 complete

Package `ocdm_doppler/` (this directory), pure NumPy, no third-party deps
beyond `numpy`. 9/9 sanity checks pass (`python verify_infrastructure.py`).

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

Added after Step 1 (new files, Step-1 modules untouched):
`af_estimation.py` (Step 2(a)–(b) of the results docs — time selection,
lag grid, empirical AF, window detection; places the time origin at received
symbol K so the ±(N_avg-1)/2 average never runs into negative symbol
indices) and `../make_figures.py`, which reproduces every results-doc figure
into `../figures/`. The reproduced figures match the documents, and they show
that the AF modulus is taken **outside** the average (see README open
question 1).

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
