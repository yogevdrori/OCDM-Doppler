# Step 7: other multi-basis waveforms (OTFS) and multipath channels

Task file for Claude Code. Read `ocdm_doppler/CLAUDE.md`, `README.md` and the
Step 4–6 task files first. Step 6 is paused after Parts A–B (its Parts C–E are
**not** part of this step).

## Why this step (advisor feedback, Nov 2026)

The advisor's first priority: the detector is based on a cyclostationary
feature, so it should work for **any multi-basis modulation** (OCDM, OTFS,
...), not only OCDM; test it on **OTFS**. He also expects it to work in
**multipath channels** and wants that checked, because it would be an
advantage over methods in the literature. (His next priorities, complexity
reduction and the comparison with the literature, come in later steps.)

What the detector really uses: with `Delta = {0}` it tracks the periodic
**mean power** of the received signal, created by the **zero guard
interval**. The data basis inside the symbol does not matter for the mean
power; the guard structure does. A quick check (`|s[n]|^2` averaged over
4000 realisations, M = 16, guard 4, 8 symbols per frame) confirmed:
- OCDM and OTFS with a zero guard after **every symbol** have identical mean
  power (16 on, 4 off, period 20 samples);
- OTFS with **one zero guard per frame** (same total overhead, 32 zeros after
  128 samples) has a pulse only once per frame (period 160 samples): 8 times
  fewer pulses in the same block;
- OFDM with a cyclic prefix has a flat mean power: the zero-lag feature
  disappears (the feature is then at the lag of the symbol length).

## Principles (unchanged)

1. **Generic.** Waveforms and channels are pluggable; the detector, loops
   and Doppler profiles of Steps 4–6 work with all of them.
2. **Add, do not change.** New files only. Do not modify existing modules.
   Allowed edits: `ocdm_doppler/__init__.py` (exports), `README.md`,
   `ocdm_doppler/CLAUDE.md`. The current code is tied to OCDM in a few places
   (`OCDMSignal._chirp_sum`, `SystemParams.N_sym` used for the cycle
   frequency, `LongSignal` inside the loop runners). Solve this with new
   classes (subclasses, wrappers, a new runner), not by editing those
   modules. If that is impossible somewhere, **stop and ask me first**.

## Part A: waveforms, channel, runner, verification

### A.1 Waveform layer (new module, e.g. `ocdm_doppler/waveforms.py`)

A waveform defines the continuous-time transmitted signal `X(t)` (evaluated
exactly at any time, like `OCDMSignal`), its data symbols, its demodulator
(for SER checks), and the **period of its mean power** in samples
(`P_samp`), which sets the cycle frequencies `k (1 + a) / P_samp`.
- `OCDMWaveform`: wraps the existing OCDM signal; must reproduce it bit for
  bit.
- `OTFSWaveform`, frame of `N` sub-symbols of `M` samples, data on an `M x N`
  delay-Doppler grid, ISFFT to the time-frequency grid, then an OFDM-type
  (Heisenberg) modulator with a rectangular pulse and **centred**
  subcarriers, with the same sample rate, bandwidth and average power per
  active sample as OCDM. Two guard structures:
  - `guard="symbol"`: a zero guard of `N_gi` samples after every sub-symbol
    (period `N_sym = M + N_gi`, as OCDM);
  - `guard="frame"`: one zero guard of `N * N_gi` samples after the frame
    (same overhead; period `N * M + N * N_gi`).
  Defaults: `M = N_is = 16`, `N_gi = 4`, `N = 16`.
- Lazy, seeded generation of the data for unbounded records, as
  `loop.LazySymbols` does for OCDM.

### A.2 Channel layer (new module, e.g. `ocdm_doppler/channel.py`)

Multipath with `P` paths, gains `A_p`, delays `tau_p`, and the Doppler profile
of Step 5 (`A(t) = int a`):

    Y(t) = sum_p A_p exp(-j 2 pi f_Tx tau_p) * carrier(t) * X(t + A(t) - tau_p) + n(t)

with the same front factor and carrier as the existing single-path model. A
single path with `tau_0 = 2 T_samp`, `A_0` as today must reproduce the current
received signal bit for bit. Options: common Doppler for all paths (default),
and optionally a per-path Doppler factor (multiscale) for one robustness
case. Path gains normalised to unit total power, so the SNR definition is
unchanged; fixed or Rayleigh gains per realisation.

### A.3 Runner (new module)

A loop runner that takes a waveform and a channel (and the Step 6 filter
objects), and uses `P_samp` for the predicted cycle frequency. With the OCDM
waveform and the single-path channel it must reproduce
`loop_tv2.run_loop_filters` bit for bit. The detector itself
(`cbsfs.cost_function`, `discriminator.error_from_costs`) is reused unchanged;
for the cycle frequency, pass parameters whose period is `P_samp` (for
example a small `SystemParams` subclass in a new file that overrides
`N_sym`), and say how you did it.

### A.4 Verification script (numbers, before Parts B–E)

1. OCDM through the new layers vs the existing code: bit for bit (signal,
   open-loop `e`, closed loop), noiseless and 4 dB.
2. OTFS implementation: at `a = 0`, single path, perfect timing, noiseless
   demodulation recovers the data exactly (both guard structures).
3. Mean power of OTFS (both guard structures) vs the expected on/off pattern.
4. **Working point for OTFS:** SNR per data symbol after demodulation at
   10 % BPSK symbol error, in the simulated channel, as in Step 4 Part A
   (theory and simulated). Expect the same `sigma_D^2 / P_n = 4.0 dB` if the
   normalisation matches OCDM; report if not.
5. Multipath channel: single path reproduces the existing signal; with `P`
   paths and `a = 0`, the received mean power equals the sum of the shifted
   power patterns.

## Fixed settings

As in Steps 4–6: BPSK, `Delta = {0}`, `N_c = 24000`, `N_seg = 2048`,
`delta = 0.25 / N_seg`, first-order loop with pole 0.99 unless stated,
noiseless and the working point, seeds `1000 + index`. Constant Doppler
values with a **non-integer shift per block** (`a * N_c` not an integer).
`K_d` must be re-measured per waveform (do not assume `K_d0 = 175.5`).

## Part B: OTFS with a guard after every symbol

Open-loop characteristic (as Step 4 Part B, reduced: prior range, SNRs
infinity, 20, 4 (WP), 0, -5 dB) for OCDM and OTFS-symbol: `K_d`,
`sigma_e / K_d`, linearity, pull-in. Closed loop at the WP for two Doppler
values: RMSE vs the linear prediction. Expectation: OTFS-symbol equals OCDM
within the Monte-Carlo error. Report any difference and its cause.

## Part C: OTFS with one guard per frame

1. Same measurements, with the harmonic `k` of the detector as a parameter.
   With one guard per frame the mean power is a long rectangular pulse, so
   higher harmonics are strong; the sensitivity to `a` grows with `k` while
   the harmonic amplitude falls. Predict the best `k` from the Fourier
   coefficients of the on/off pattern (roughly where `k * N * N_gi / P_samp`
   is about 1/2, i.e. `k ~ 2-3` for the defaults), then measure `k in {1, 2,
   3, 4, 5}`.
2. Report `K_d`, `sigma_e / K_d`, pull-in range (it scales with
   `P_samp / (k N_seg)`), and closed-loop RMSE at the WP, against
   OTFS-symbol and OCDM.
3. Comment on the trade-off: fewer pulses per block vs a longer guard.

## Part D: multipath

1. Exponential power-delay profile with `P` paths (e.g. 4), common Doppler,
   RMS / maximum delay spread `{0, N_gi / 2, N_gi, 2 N_gi}` samples (for
   OTFS-frame also `{N * N_gi / 2, N * N_gi}`), fixed and Rayleigh gains.
2. For OCDM (= OTFS-symbol) and OTFS-frame: open-loop `K_d` and
   `sigma_e / K_d`, and closed-loop RMSE and bias at the WP, vs delay spread.
3. Expectation to test: delay spread fills the zero gap of the mean power,
   so the feature weakens once the spread approaches the guard; a longer guard
   (OTFS-frame) should tolerate a longer spread. Does the loop still lock? Is
   there a bias (for example from the power-weighted mean delay)?
4. One multiscale case (two paths with Doppler factors differing by `1e-4`):
   what does the loop converge to (power-weighted mean?).

## Constraints

- Run in the foreground in checkpointed rounds; `--quick`, `--resume`,
  `--replot`. Memory rule from Step 6 (>= 3 GB free: 4 processes, else 3;
  stop above ~400 ms/block/worker or below 1 GB free).
- Before each part, a runtime estimate from a timed `--quick` run; ask me if
  it exceeds ~35 min. Keep run sizes modest: this step is a feasibility study
  for the advisor meeting (26.11), not a final characterisation.
- Report honestly. If OTFS-frame or multipath do not work well, say so and
  show why.

## Deliverables

- New modules (waveforms, channel, runner, parameter subclass), the
  verification script, study script(s) (`--part B/C/D`).
- `results/*.md` / `*.csv`, figures: mean-power patterns of the three
  waveforms; `sigma_e / K_d` vs SNR per waveform; OTFS-frame vs harmonic `k`;
  RMSE / `K_d` vs delay spread.
- README "Step 7 results" and `ocdm_doppler/CLAUDE.md` update.
- Commit and push. Then a short summary: verification numbers, OCDM vs
  OTFS-symbol, the best harmonic and performance of OTFS-frame, the multipath
  results, anything unexpected, total runtime.
