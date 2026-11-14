# Step 6: second-order tracking loop

Task file for Claude Code. Read `ocdm_doppler/CLAUDE.md`, `README.md` and
`tasks/step5_time_varying_doppler.md` first. Steps 1–5 are done. Step 5 showed
that the first-order loop tracks a time-varying Doppler factor exactly as the
linear model predicts, including its weakness: on a ramp it lags by
`beta / g`, so the best pole is a compromise between lag and noise. This step
adds a **second-order loop**, which also estimates the rate of change of the
Doppler factor and therefore follows a ramp **without** a steady-state lag,
and compares it with the first-order loop.

## Principles (unchanged from Step 5)

1. **Generic.** The new loop must work with every Doppler profile of
   `doppler_profile.py` and with both lag sets, noiseless and noisy.
2. **Add, do not change.** Implement everything in **new files**. Do not
   modify any existing module, including `loop.py`, `loop_tv.py`,
   `signal_tv.py` and `doppler_profile.py`. Allowed edits:
   `ocdm_doppler/__init__.py` (exports), `README.md`, `ocdm_doppler/CLAUDE.md`.
   If something cannot be done without changing an existing module, **stop
   and ask me first**.

## The loop

Per block `i` the detector gives `e_i` (unchanged, `disc.error_from_costs`
on the two probes around `alpha_p = (1 + a_hat_i) / N_sym`). The loop keeps
the estimate `a_hat` and a rate estimate `b_hat` (change of `a` per block):

    b_hat[i+1] = b_hat[i] + mu2 * e[i]
    a_hat[i+1] = a_hat[i] + b_hat[i+1] + mu1 * e[i]

(a proportional-plus-integral loop filter followed by an integrator). With
`mu2 = 0` and `b_hat[0] = 0` it is exactly the first-order loop with
`mu = mu1`.

### Linear model

With `e_i = K_d (a_bar_i - a_hat_i) + w_i`, define `g1 = mu1 K_d`,
`g2 = mu2 K_d`. The transfer function from `a_bar` to `a_hat` is

    H2(z) = ((g1 + g2) z - g1) / (z^2 - (2 - g1 - g2) z + (1 - g1)).

- **Stability:** `0 < g1 < 2`, `g2 > 0`, `2 g1 + g2 < 4`.
- **Design by poles:** for closed-loop poles `r exp(+/- j theta)`,
  `g1 = 1 - r^2`, `g2 = 1 - 2 r cos(theta) + r^2`. Critically damped
  (double real pole `p`): `g1 = 1 - p^2`, `g2 = (1 - p)^2`. For a damping
  `zeta` and natural frequency `w_n` (rad per block), use
  `r = exp(-zeta w_n)`, `theta = w_n sqrt(1 - zeta^2)`.
- **Steady-state errors** (`eps = a_hat - a_bar`): constant `a`: 0; ramp
  (slope `beta` per block): **0**; constant second difference `gamma` per
  block^2 (constant rate of change of the slope): `-gamma / g2`.
- **Noise:** `Var(eps) = (sigma_e / K_d)^2 * B_L`, with the noise bandwidth
  `B_L = sum_n h_n^2` (`h_n` the impulse response of `H2`). For the
  first-order loop `B_L = g / (2 - g)`.

I checked these numerically (linear recursion): a ramp gives zero error, a
parabola gives `-gamma / g2`, and the critically damped design has
`B_L = 0.0126` for `p = 0.99`. **Equal-noise designs** (critically damped,
same `B_L` as the first-order loop with pole `p1`):

| first-order `p1` | `B_L` | second-order double pole `p` | `g1` | `g2` | step `n95` (2nd / 1st) | step overshoot |
|---|---|---|---|---|---|---|
| 0.9 | 0.0526 | 0.95942 | 0.0795 | 1.65e-3 | 101 / 28 | 14 % |
| 0.95 | 0.0256 | 0.97986 | 0.0399 | 4.06e-4 | 205 / 58 | 14 % |
| 0.99 | 0.00503 | 0.99599 | 0.00800 | 1.60e-5 | 1033 / 298 | 14 % |

So at equal noise the second-order loop removes the ramp lag but converges
about 3.5 times more slowly from a step, with overshoot. The experiments must
quantify this trade-off. Recompute these numbers in your helpers and report
any disagreement.

## Fixed settings (as in Step 5 unless stated)

22-Jan preset (BPSK), `Delta = {0}`, `N_c = 24000`, `N_seg = 2048`, `k = 1`,
`delta = 0.25 / N_seg`, `K_d0 = 175.5`, `mu1 = g1 / K_d0`, `mu2 = g2 / K_d0`.
SNRs: noiseless and the working point 4.0 dB (`sigma_e / K_d = 4.59e-5`).
Seeds `1000 + index`. Rates `r` and `beta = r * 3.25e-8` per block as in
Step 5.

**Constant-Doppler tests: use Doppler values with a non-integer shift per
block** (`a * N_c` not an integer, e.g. `a = 5.154e-4`,
`results/bias_phase_check.md`), otherwise the frozen-sampling-phase bias of
Step 4 mixes into the results.

## Part A: implementation and verification

1. **New module `ocdm_doppler/loop_filters.py`.** Loop filters as small
   classes with a common interface (state, `update(e) -> a_hat_next`, gains),
   at least `FirstOrderFilter(mu)` and `SecondOrderFilter(mu1, mu2,
   b_hat0=0)`, plus design helpers: gains from poles, from (`zeta`, `w_n`),
   and from a target `B_L` (critically damped); `H2`, `B_L`, step response,
   `n95`, overshoot, and the steady-state errors above.
2. **New runner** (e.g. `ocdm_doppler/loop_tv2.py`,
   `run_loop_filters(params, profile, filters, n_blocks, snrs, ...)`): same
   signal, noise stream, detector and lockstep design as
   `loop_tv.run_loop_profile`, but each loop is a filter object. Return
   `a_hat`, `b_hat` (where defined), `e`, `a_bar`, `eps` as in Step 5.
   Support checkpointed segments as in `study_tracking.py`.
3. **Verification script `verify_second_order.py`** (numbers, not just
   pass/fail), all before Parts B–E:
   - `FirstOrderFilter` in the new runner vs `run_loop_profile`: identical
     trajectories (bit for bit), noiseless and at 4 dB.
   - `SecondOrderFilter(mu1, mu2 = 0)` vs the first-order loop with `mu1`:
     identical.
   - Lockstep vs separate runs, and segmented vs single run: identical.
   - Analytical helpers vs a direct linear recursion (step, ramp, parabola,
     `B_L`), and the table above.
   - Noiseless closed loop on a ramp: steady-state error vs 0 (it will be
     limited by the `a`-dependent detector offset of Step 5; use the up/down
     ramp method of Step 5 to separate it).

## Part B: constant Doppler and acquisition

1. Designs: the first-order poles `p1 in {0.9, 0.95, 0.99}` and, for each,
   the equal-`B_L` second-order loop with `zeta = 1` (critically damped) and
   with `zeta = 0.707`.
2. **Steady state** at a constant non-integer `a` (two values in the prior
   range): error standard deviation vs `(sigma_e / K_d) sqrt(B_L)`, and the
   bias. Noiseless and 4 dB.
3. **Acquisition** from `a_hat0 = 0`, `b_hat0 = 0` to `a in {5e-4, 1e-3}` and
   to larger initial errors `{3e-3, 6e-3}` (inside the pull-in range
   `+/-1e-2` of Step 4): `n95` and overshoot vs the linear prediction, and
   whether the loop locks. The overshoot of the second-order loop may push
   the error beyond the pull-in range for large initial errors: report the
   largest initial error that still locks, for each design.

## Part C: ramp

1. Ramps with `r in {1, 10}` (as in Step 5, up and down over the same
   interval), same designs as Part B.
2. Report the bias-free lag `(eps_up - eps_down) / 2` against 0 for the
   second-order loop and against `-beta / g` for the first-order loop, the
   error standard deviation, and the total RMSE. Main comparison: the total
   RMSE of each second-order design against the **best** first-order pole for
   that `r` (Step 5: 0.99 for `r = 1`, 0.95 for `r = 10`).

## Part D: fly-by

1. The two fly-bys of Step 5 (`a_max = 5e-4`, `T_c = 9.22 s` and `0.92 s`),
   same seeds and length.
2. Second-order designs: critically damped, double pole `p` on a grid wide
   enough to contain the best one (for example `p in {0.95, 0.97, 0.98, 0.99,
   0.995}`), and the `zeta = 0.707` design at the best `B_L`. Include the
   best first-order loop of Step 5 (0.95 for `T_c = 9.22 s`, 0.9 for
   `T_c = 0.92 s`) in the same lockstep run for a direct comparison.
3. Report RMSE over the run, peak error, error vs time against the
   quasi-static prediction `-gamma(t) / g2` and against the full linear model
   (`H2` applied to `a_bar`, plus noise). Find the best second-order design
   per fly-by (simulation and linear model) and compare its RMSE with the best
   first-order loop.

## Part E: frequency response (reduced)

Sinusoid `amp = 5e-4` as in Step 5, for the critically damped and the
`zeta = 0.707` designs that match the first-order `p1 = 0.99`, at
`f / f_3dB in {0.1, 0.3, 1, 3}` (`f_3dB` of `H2`). Measured gain and phase vs
`H2`. Expect a gain above 1 near the natural frequency (peaking), stronger for
`zeta = 0.707`. Noiseless and 4 dB. If the low frequencies need runs longer
than in Step 5, estimate and ask.

## Robustness check

The detector gain drops at low SNR (Step 4: `K_d / K_d0 = 0.994` at 0 dB,
`0.549` at -5 dB), which moves the poles of the loop. For the main designs,
compute where the poles go with `K_d = 0.549 K_d0` and confirm with a short
run at -5 dB (constant `a` and the fast fly-by) that the loop stays stable.

## Constraints

- Run in the foreground, save incrementally, support `--quick`, `--resume`,
  `--replot`.
- **Estimate the runtime of each part before running it and ask me if it
  exceeds ~35 min.** Step 5 took longer than estimated; base the estimate on
  a timed `--quick` run.
- Report honestly; do not tune designs to make figures look good. If the
  second-order loop is not better somewhere, say so.

## Deliverables

- New code: `ocdm_doppler/loop_filters.py`, the new runner,
  `verify_second_order.py`, study script(s) (e.g. `study_second_order.py
  --part B/C/D/E`).
- `results/*.md` / `*.csv` per part, figures in `figures/`, at least:
  step responses (1st vs 2nd order, equal `B_L`); RMSE vs design for the ramp
  and for the fly-bys, with the best first-order loop marked; fly-by error vs
  time (best 1st vs best 2nd order); measured vs predicted frequency response.
- README "Step 6 results" section and `ocdm_doppler/CLAUDE.md` update.
- Commit and push. Then a short summary: verification numbers, ramp lag of
  the second-order loop, the acquisition trade-off (convergence time,
  overshoot, largest initial error that locks), the fly-by comparison (best
  1st vs best 2nd order), the frequency response, the low-SNR stability
  check, anything unexpected, and total runtime.
