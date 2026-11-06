# Step 5: tracking a time-varying Doppler factor

Task file for Claude Code. Read `ocdm_doppler/CLAUDE.md` and `README.md`
first. Steps 1–4 are done. Step 4 (`tasks/step4_detector_and_loop.md`)
established, for a **constant** Doppler factor `a`: the open-loop detector
characteristic, the working point, the 95 % convergence time and the
closed-loop bias / NMSE versus SNR. This step adds a **time-varying** `a(t)`
and studies how well the first-order loop tracks it, and which loop pole
(bandwidth) to choose.

## Two principles for this step (from me, not negotiable)

1. **Generic.** The code must accept a constant Doppler (as today) **and** any
   time-varying profile `a(t)`. The profiles used in the studies below are
   examples, not a closed list.
2. **Add, do not change.** Existing behaviour and results must not change.
   Implement the new capability in **new files** that reuse the existing
   classes. Do not modify `params.py`, `signal_model.py`, `analytical_af.py`,
   `cbsfs.py`, `af_estimation.py`, `curve_fit.py`, `discriminator.py` or the
   existing functions in `loop.py`. Allowed edits to existing files:
   - `ocdm_doppler/__init__.py`: export the new names;
   - `loop.py`: only the text of the `NotImplementedError` in `run_loop`,
     to point to the new function (no logic change);
   - `README.md`, `ocdm_doppler/CLAUDE.md`: documentation.
   If you find that something cannot be done without changing an existing
   module, **stop and ask me first**, with the reason and the minimal change.

## Physics of a time-varying Doppler factor

Today `signal_model.OCDMSignal.rx` uses a constant `a`:

    u(t)      = (1 + a) t - tau_p0                      (time warp)
    carrier   = exp(j (2 pi Delta_f(a) t + Delta_phi)),
                Delta_f(a) = (1 + a) f_Tx - f_Rx

For a time-varying `a(t)`, the time warp and the carrier phase are the
**integrals** of the instantaneous rates. With `A(t) = int_0^t a(s) ds`:

    u(t)      = t + A(t) - tau_p0
    phase(t)  = 2 pi ((f_Tx - f_Rx) t + f_Tx A(t)) + Delta_phi

(the front factor `A_p0 exp(-j 2 pi f_Tx tau_p0) / (2 sqrt(N_is))` is
unchanged). For constant `a`, `A(t) = a t` and this reduces exactly to the
current expressions. Evaluating the existing `rx` with a different constant
`a` per block is **wrong**: the symbol timing and the carrier phase would jump
at block boundaries.

Precision note: the carrier phase contains `2 pi f_Tx A(t)` with
`f_Tx = 1 GHz`, so an error `dA` in the integral gives a phase error of
`2 pi 1e9 dA`. `A(t)` must therefore be computed in **closed form** (or to
an accuracy of about `1e-13` s), not by a coarse numerical integration.

## Part A: infrastructure and verification

### A.1 Doppler profiles: new module `ocdm_doppler/doppler_profile.py`

- A base class `DopplerProfile` with:
  - `a(t)`: instantaneous Doppler factor (vectorised over `t` in seconds);
  - `A(t)`: `int_0^t a(s) ds`, in closed form;
  - `block_mean(t0, t1)`: `(A(t1) - A(t0)) / (t1 - t0)`, the mean Doppler
    over a block (the quantity the loop estimates per block).
- Implementations, each with a closed-form `A(t)`:
  - `ConstantDoppler(a)`;
  - `RampDoppler(a0, rate)`: `a(t) = a0 + rate * t` (`rate` in 1/s);
  - `SinusoidalDoppler(a0, amp, freq, phase=0)`:
    `a(t) = a0 + amp * sin(2 pi freq t + phase)`;
  - `FlyByDoppler(a_max, t0, T_c)`: a source passing the receiver,
    `a(t) = -a_max * (t - t0) / sqrt((t - t0)^2 + T_c^2)`. It goes from
    `+a_max` to `-a_max`, with the largest rate `a_max / T_c` at `t = t0`;
  - `PiecewiseLinearDoppler(times, values)`: an arbitrary profile through
    given points, with exact piecewise-quadratic `A(t)`. This is the generic
    "any profile" option.
  - Optionally `CallableDoppler(a_fn, A_fn)` for a user function, which
    **requires** the closed-form integral `A_fn`.

### A.2 Time-varying received signal: new module `ocdm_doppler/signal_tv.py`

- Functions (or a mixin) `rx_profile(sig, t, profile, noiseless=False,
  noise_index=None)` and `rx_nominal_profile(sig, n, profile, noiseless=False,
  start_sample=0)`, the time-varying counterparts of `OCDMSignal.rx` and
  `OCDMSignal.rx_nominal`, with the formulas above.
- Reuse the existing internals of `OCDMSignal` (`_decompose`,
  `_chirp_sum`, the noise generator) instead of copying them. They must work
  for both `OCDMSignal` and `loop.LongSignal`.
- A plain `float` passed as the profile is wrapped in `ConstantDoppler`.

### A.3 Loop with a profile: new module `ocdm_doppler/loop_tv.py`

- `run_loop_profile(params, profile, mu, n_blocks, snrs=(inf,), a_hat0=0.0,
  N_c=24000, N_seg=2048, k=1, delta=None, seed=0, start_block=0)`, the same
  algorithm and the same data / noise generation as `loop.run_loop`
  (same seeds, same per-block noise stream), but the received block is
  generated with `rx_nominal_profile`.
- Return a result that contains, in addition to `a_hat[s, i]` and `e[s, i]`,
  the true Doppler per block: the block mean `a_bar[i]` (and `a(t)` at the
  block start and end).
- **Error convention:** `eps[s, i] = a_hat[s, i] - a_bar[i]`, i.e. the
  estimate used in block `i` minus the true mean of block `i`. With this
  convention the linear model of the report gives, for a ramp with
  `beta = a_bar[i+1] - a_bar[i]` per block, a steady-state lag
  `eps = -beta / g` (`g = mu K_d`).

### A.4 Verification: new script `verify_time_varying.py`

All checks must pass **before** any study is run. Report the measured
numbers, not just pass/fail.
1. **Equivalence with the existing code (constant a).**
   - `rx_profile(ConstantDoppler(a))` against `OCDMSignal.rx(a=a)`,
     noiseless and noisy, for several `a` and times up to the longest run
     length used below. Report the maximum absolute difference (expected:
     floating-point level, from `(1 + a) t` versus `t + a t`).
   - `run_loop_profile(ConstantDoppler(a))` against `run_loop(a)`, same
     seeds, noiseless and at 4 dB: the `a_hat` trajectories must agree to
     floating-point level.
2. **Integrals.** For each profile, `A(t)` against a high-accuracy
   numerical integral of `a(t)`.
3. **Continuity.** One long record generated at once against the same record
   generated block by block: identical samples (no jump at block boundaries).
4. **Old results unchanged.** `git diff` on the existing modules is empty
   (except the allowed edits). Re-run a quick existing check (for example
   `study_loop.py --part C --quick`) and confirm the numbers equal the
   committed ones.
5. **Detector on a ramp (noiseless, open loop).** For a steep ramp
   (the largest `beta` below), compare the mean of `e / K_d0` over many
   blocks with `a_bar - a_hat`. This checks that the Doppler change within one
   block does not bias the detector.

## Fixed settings (as in Step 4 unless stated)

22-Jan preset (BPSK), `Delta = {0}`, `N_c = 24000`, `N_seg = 2048`, `k = 1`,
`delta = 0.25 / N_seg`, `K_d0 = 175.5`, `mu = (1 - p) / K_d0`.
SNRs: noiseless and the working point `sigma_D^2 / P_n = 4.0 dB` (SNR per data
symbol after DFnT at 10 % BPSK symbol error, confirmed by the advisor).
At the working point, `sigma_e / K_d = 4.59e-5` (Step 4, Table 4).
Seeds `1000 + index`. Keep `|a(t)| <= 1e-3` (the prior range) in all runs.

**Rate of change.** There is no physical scenario fixed yet, so the Doppler
rate is defined **relative to the loop**. With pole `p = 0.99` at the working
point, the steady-state noise RMSE is
`sigma_ss = (sigma_e / K_d) sqrt(g / (2 - g)) = 3.25e-6`. Define the ramp
slope per block

    beta = r * g * sigma_ss,   g = 0.01,   r in {0.1, 1, 10},

i.e. `beta = 3.25e-9, 3.25e-8, 3.25e-7` per block, for which the predicted
lag of the `p = 0.99` loop is 0.1, 1 and 10 times its noise RMSE. Also give
each rate in 1/s (block duration 6 ms).

## Part B: ramp (constant rate) and the choice of pole

1. `RampDoppler` with the three `beta` values. Poles
   `p in {0.9, 0.95, 0.98, 0.99, 0.995}`. Start with `a_hat0 = a(0)`, choose
   `a(0)` so that `a(t)` stays within the prior range for the whole run.
   Discard at least `3 n95` blocks, then measure (several realisations and
   enough blocks for a reliable mean and variance, as in Step 4 Part D).
2. Report per (`beta`, `p`, SNR): mean error (lag) against `-beta / g`, error
   standard deviation against `(sigma_e / K_d) sqrt(g / (2 - g))`, total RMSE
   against the prediction `sqrt((beta / g)^2 + (sigma_e / K_d)^2 g / (2 - g))`.
3. **Optimal pole.** For each `beta`, the pole that minimises the total RMSE:
   analytically (minimise the expression above over `g`) and from the
   simulation. Figure: total RMSE versus pole, one curve per `beta`, simulated
   points and the analytical curve, optimum marked. Table: optimal pole and
   its RMSE per `beta`.
4. Noiseless: confirm that the lag equals `beta / g` (this also covers the
   in-block effect of A.4.5 in closed loop).

## Part C: sinusoidal Doppler (frequency response of the loop)

1. `SinusoidalDoppler(a0 = 0, amp = 5e-4, freq)`. For a first-order loop the
   transfer from `a_bar` to `a_hat` is `H(z) = g z^-1 / (1 - (1 - g) z^-1)`
   (one update per block). Compute its exact 3 dB frequency `f_3dB` (in Hz)
   per pole, and use `freq / f_3dB in {0.1, 0.3, 1, 3}`.
2. Poles `p in {0.9, 0.99}`, noiseless and the working point.
3. After the transient, fit a sinusoid at the known frequency to `a_hat` and
   to `a_bar`: report the measured gain and phase against `|H|` and `arg H`,
   and the tracking RMSE against
   `sqrt(amp^2 |1 - H|^2 / 2 + (sigma_e / K_d)^2 g / (2 - g))`.
4. Figure: measured and predicted gain (and phase) versus `freq / f_3dB`,
   per pole; an example trajectory (`a_bar`, `a_hat`) per pole.
5. Low frequencies with `p = 0.99` need long runs (the period at
   `0.1 f_3dB` is tens of seconds). Use at least 3 periods after the
   transient. If this is too long, estimate the runtime and ask me before
   dropping or shortening a point.

## Part D: fly-by (a realistic profile)

1. `FlyByDoppler(a_max = 5e-4, t0, T_c)` with `T_c` chosen so that the peak
   slope per block, `a_max N_c T_samp / T_c`, equals `beta` for `r = 10` and
   `r = 100` (about `T_c = 9.2 s` and `0.92 s`). Run from `t0 - 4 T_c` to
   `t0 + 4 T_c`, `a_hat0 = a(0)` (start in steady state).
2. Poles `{0.9, 0.95, 0.99}` plus the optimal pole from Part B for the closest
   `beta`. Noiseless and the working point, several realisations.
3. Report the error versus time against the quasi-static prediction
   `eps(t) ~ -beta(t) / g` (`beta(t)` the slope per block at time `t`), the
   peak error and the RMSE over the run.
4. Figures: `a_bar(t)` and `a_hat(t)` for each pole; the error versus time
   with the prediction.

## Constraints

- Earlier long runs were killed by a low-memory safeguard. Run in the
  foreground, save results incrementally, support `--quick` and `--resume`
  (and `--replot` where useful).
- **Before any run longer than ~20 min, estimate its runtime and ask me.**
  Give the estimate for Parts B, C and D separately, before starting them.
- Report honestly. If something does not behave as predicted (the lag is not
  `beta / g`, the detector is biased on a ramp, the loop loses lock, the
  sinusoid response does not follow `H`), report and investigate. Do not tune
  things to make figures look good.

## Deliverables

- New code: `ocdm_doppler/doppler_profile.py`, `ocdm_doppler/signal_tv.py`,
  `ocdm_doppler/loop_tv.py`, `verify_time_varying.py`, and study scripts
  (for example `study_tracking.py --part B/C/D`).
- `results/*.md` / `*.csv` per part, figures in `figures/`.
- README: a "Step 5 results" section with the tables, figures and a short,
  honest interpretation; update the status table and `ocdm_doppler/CLAUDE.md`
  (Step 5 status, the new modules, the "add, do not change" rule).
- Commit and push. Then give me a short summary: the verification numbers
  (A.4), the lag against `beta / g`, the optimal pole per `beta`, the
  frequency response against `H`, the fly-by peak errors, anything unexpected,
  and total runtime.
