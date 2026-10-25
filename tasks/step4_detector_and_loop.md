# Step 4: detector characteristic and first-order loop

Task file for Claude Code. Read `ocdm_doppler/CLAUDE.md` and `README.md`
first (Steps 1–3 are done: infrastructure, curve-fit reproduction, early-late
S-curve). This file records what the advisor (Ron Dabora) asked for after
reviewing the Step-3 report, and how to do it.

## Advisor feedback (summary)

1. The Step-3 S-curve figures (`fig_scurve_delta.png`, `fig_scurve_snr.png`)
   fix the true `a` and sweep `a_hat`. He finds that uninformative. He wants
   the **detector characteristic** with the **true `a` on the x-axis** and the
   **mean error `a - a_hat`** on the y-axis (like Fig. 7 of Rotem & Dabora
   2020: mean estimate vs. true value, against the unbiased line).
2. The key performance figures are **bias and NMSE vs. SNR for a set of
   Doppler values** (like Figs. 6–7 of the CB-SFS paper: MSE vs. the parameter
   value at a fixed low SNR, and MSE vs. SNR).
3. Reducing the lag set to `Delta = 0`: both options (`{0}` and
   `{0..N_is-1}`) must be plotted to see whether the extra lags help. It is
   fine to proceed first with `Delta = 0` only.
4. Next: compute the 95 % convergence time of a first-order loop, then
   simulate the loop vs. SNR. Start with a narrow loop bandwidth (pole 0.99).
   The loop sees the noise at the detector output, not the additive channel
   noise. Use a **low SNR**, around the point where the symbol error
   probability of the modulation is about 10 %, because the loop runs at the
   start of reception. Loop bandwidth trades steady-state error against
   tracking.

Two definitions are still being confirmed with the advisor, so **keep both
flexible in the code**:
- (Q1) Whether "detector characteristic" means the open-loop detector output
  (fixed `a_hat`, sweep the true `a`), as implemented in Part B, or the
  closed-loop final estimate. Part B is the open-loop version. The closed-loop
  version comes out of Part D anyway.
- (Q2) Which SNR the 10 % symbol-error point refers to. Run a wide SNR range
  and report the mapping between definitions (Part A), so the working point can
  be marked afterwards.

## Constraints

- Do not modify the Step 1–3 modules (`params.py`, `signal_model.py`,
  `analytical_af.py`, `cbsfs.py`, `af_estimation.py`, `curve_fit.py`,
  `discriminator.py`). Add new files. The lag set is already configurable:
  `params.with_(cbsfs_deltas=(0,))` makes `cbsfs.cost_function` (and hence
  `discriminator.early_late`) use `Delta = 0` only.
- Same settings as Step 3 unless stated: 22-Jan preset (BPSK), `N_c = 24000`,
  `N_seg = 2048`, `k = 1`, `delta = 0.25 / N_seg`, seeds `1000 + index`.
- Earlier long runs were killed by a low-memory safeguard. Run in the
  foreground, save results incrementally, support `--quick` and `--resume`.
  **Before any run longer than ~20 min, estimate its runtime and ask me.**
- Report honestly. If something does not behave as expected (`K_d` changes
  with SNR, `Delta = 0` breaks down at low SNR, false lock, bias), report and
  investigate. Do not tune things to make figures look good.

## Part A: SNR definitions and the 10 % symbol-error working point

1. Derive the SNR definitions in terms of `sigma_D^2 / P_n` (the one used so
   far):
   - per received sample (average over symbol + GI),
   - per data symbol after DFnT demodulation (single path, perfect
     synchronisation). Check the receiver front-end factor `A0 / 2` in
     `signal_model.rx` and the noise power definition in `_BlockNoise` /
     `params.noise_power`.
   My hand calculation, to verify (not to trust): per-symbol SNR =
   `|A0|^2/4 * sigma_D^2/P_n` (`|A0|^2/4 = -4.9 dB`). For BPSK, SER = 10 % at
   `Es/N0 = -0.86 dB`, i.e. `sigma_D^2/P_n ~ 4.0 dB`. For QPSK, `~ 9.2 dB`.
2. **Verify by simulation**: with `a = 0` and perfect timing, demodulate
   (DFnT of the active part of each received symbol, divide by the known
   channel factor), make hard decisions, and measure SER vs. `sigma_D^2/P_n`.
   Report the measured 10 % point next to the analytical one.
3. Write the mapping table to `results/snr_definitions.md`.

## Part B: detector characteristic (open loop), Delta = 0 vs. all lags

For each lag set `{0}` and `{0..15}`:
1. Fix `a_hat = 0`. Sweep the true `a` over a dense grid covering the prior
   range (`|a| <= 1e-3`, about 41 points) and a wider grid out to about
   `+/-2e-2`, to show the pull-in edges.
2. For each `a` and each SNR in a wide range
   (`sigma_D^2/P_n` in {-5, 0, 2.5, 5, 7.5, 10, 20, inf} dB, plus the Part A
   working point), compute `e` on `N_mc = 64` independent blocks.
3. **Gain calibration.** `K_d` is a design constant of the loop. Measure it
   once, from the noiseless characteristic (slope of mean `e` vs. `a` at
   `a = a_hat`). Also report `K_d` measured at each SNR: at low SNR the noise
   floor of `C(alpha)` may reduce the slope. If it does, that matters for the
   loop gain and must be stated.
4. Plot the characteristic: x = true `a` (with `a_hat = 0`, so x is also
   `a - a_hat`), y = mean detector output in units of `a`, i.e.
   `mean(e) / K_d`, with a +/-1 std band and the ideal line y = x. One panel
   per lag set, curves for several SNRs. Also plot the bias
   `mean(e)/K_d - (a - a_hat)` vs. `a`.
5. Summary table per lag set and SNR: `K_d`, `sigma_e`, `sigma_e / K_d`,
   linearity over the prior range, pull-in range, max |bias| over the prior
   range. Also give the complexity per sample of each lag set (Step-3 count:
   `{0..15}` ~ 306 RM/sample, `{0}` at most 28 RM/sample). Write the exact
   number for `{0}`, noting that `|Y|^2` is real.

## Part C: 95 % convergence time of the first-order loop

Loop (Algorithm 1 of the report): `a_hat[i+1] = a_hat[i] + mu * e[i]`, one
update per block of `N_c` samples, `g = mu * K_d`, pole `p = 1 - g`.
1. Analytical: `n95 = ln(0.05) / ln(p)` blocks. Tabulate for
   `p in {0.9, 0.95, 0.99}` and `N_c in {24000, 12000, 6000}` (check that
   `N_c` stays well above `2 N_seg`), in blocks and in seconds at 4 MHz.
   For reference: `p = 0.99`, `N_c = 24000` gives about 298 blocks = 1.8 s.
2. Verify with a noiseless loop simulation (step response from
   `a_hat_0 = 0` to a constant `a`).

## Part D: closed-loop simulation vs. SNR (start with Delta = 0)

New module `ocdm_doppler/loop.py`, with a function that runs the loop on
consecutive non-overlapping blocks of one long received signal. It returns
the `a_hat` trajectory and the `e` sequence. Allow a time-varying `a` (needed
later for tracking). The signal model takes a constant `a`; if a time-varying
one needs a change in `signal_model.py`, stop and ask first.

1. Pole `p = 0.99` (`mu = 0.01 / K_d` with the noiseless `K_d` of the lag set),
   `a_hat_0 = 0`.
2. Doppler values: a set covering the prior range, e.g.
   `a in {-1e-3, -5e-4, -1e-4, 1e-4, 5e-4, 1e-3}`. Avoid `a = 0`, where NMSE
   is undefined; report RMSE as well.
3. SNR grid as in Part B, including the working point.
4. Measure the **steady state** after discarding the transient (at least
   `3 * n95` blocks). The loop output is correlated over about `1/g` blocks,
   so use several realisations and enough post-transient blocks for a
   reliable estimate. State the effective number of independent samples.
   Estimate the total runtime first and ask me if it exceeds ~20 min.
   Shortening the post-transient length or the number of realisations is fine
   if justified.
5. Report per (`a`, SNR): bias `E{a_hat - a}`, normalised bias, NMSE, RMSE.
   Compare the steady-state variance with the linear prediction
   `(sigma_e / K_d)^2 * g / (2 - g)` using `sigma_e` from Part B.
6. Figures:
   - NMSE vs. SNR, one curve per `a` (the CB-SFS Fig. 7 analogue), with the
     prediction.
   - Normalised bias vs. SNR, one curve per `a`.
   - NMSE (and mean `a_hat`) vs. `a` at the working-point SNR (the CB-SFS
     Fig. 6 / Rotem–Dabora Fig. 7 analogue).
   - An example of `a_hat` trajectories (transient + steady state) at the
     working point.
7. If runtime allows, repeat a reduced version (e.g. 2 values of `a`, 3 SNRs)
   with the full lag set `{0..15}`, for the comparison requested in feedback
   point 3.

## Deliverables

- New code: `ocdm_doppler/loop.py` and scripts (e.g. `study_snr_definitions.py`,
  `study_characteristic.py`, `study_loop.py`), each with `--quick`,
  `--resume` and (where useful) `--replot`.
- `results/*.md` / `*.csv` for Parts A–D, figures in `figures/`.
- README: a "Step 4 results" section with the tables, figures and a short,
  honest interpretation. Update the status table and `ocdm_doppler/CLAUDE.md`.
- Commit and push. Then give me a short summary: the measured working point,
  `K_d` vs. SNR for both lag sets, the characteristic (is it linear and
  unbiased over the prior range?), the convergence times, the loop bias and
  NMSE at the working point vs. the prediction, anything unexpected, and
  total runtime.
