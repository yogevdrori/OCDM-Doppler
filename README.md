# OCDM Doppler estimation — simulation infrastructure

Clean-room implementation (from scratch, per Ron's request) of the signal
model and estimators behind the AF-magnitude Doppler estimator for OCDM.

Sources:
- **[OCDM]** Z. Tan, *OCDM Signal Analysis* — signal model (Sec. 2–3),
  analytical AF (Eqns. 8–12), Section-4 algorithm.
- **[CBSFS]** M. Kumar and R. Dabora, *A Novel Sampling Frequency Offset
  Estimation Algorithm for OFDM Systems Based on Cyclostationary
  Properties*, IEEE Access 7, 2019 — Eqns. (27)–(34).

## Layout

| module | contents |
|---|---|
| `params.py` | `SystemParams` — one frozen object fully describes an experiment; presets mirror the two results documents |
| `signal_model.py` | OCDM transmitter `X(t)`, single-path Doppler receiver `Y(t)` ([OCDM] Eqn. 3), deterministic block noise |
| `analytical_af.py` | `Pi`, `Theta`, `Gamma`, `c_Y`, `af_magnitude` ([OCDM] Eqns. 8–12) |
| `cbsfs.py` | segmented CAF (27), autocorrelated CAF (28), cost function (29), peak location, Doppler estimate (33)–(34) |

| `af_estimation.py` | results-doc Step 2(a)–(b): time selection (Eqn. 1), lag grid, empirical AF, window detection |

Run `python verify_infrastructure.py` for the sanity checks (9/9 passing).

`python make_figures.py` reproduces the figures of the results documents
into `figures/` (31-Jan noiseless AF, 22-Jan Settings 1–5, 22-Jan dense
window, and a [CBSFS] Fig. 5-style cost-function plot). `--a-sr genie`
skips CB-SFS; `--a-chosen <a>` adds the curve-fit curve to the 31-Jan figure
once Step 2 exists.

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

## Findings so far

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

1. **Is the magnitude inside or outside the AF average?** The steps
   documents write

   ```
   c_hat_Y(t_sel, tau_k) = sum_i | Y(...) Y*(...) | / N_avg
   ```

   with the modulus *inside* the sum. That estimator converges to
   `E{|Y(t)||Y(t-tau)|}`, which is strictly positive and has **no nulls** —
   it cannot match the analytical `|Pi|`, which does. Writing
   `| sum_i Y Y* | / N_avg` (modulus outside) is what converges to `|c_Y|`.
   The noisy AF plots do show a positive floor in the sidelobes, so this
   may be deliberate, a typo, or the source of the bias. Both variants
   need testing.

   *Update (figure reproduction):* the published figures match only the
   modulus-**outside** version. In the 31-Jan noiseless figure the empirical
   AF falls to ~0.02 at the edge of the main lobe; modulus-inside would stay
   near 0.32·π/4 ≈ 0.25 there. The noisy sidelobe floor (~0.01–0.02) is
   the finite-`N_avg` residual `≈ 0.32/sqrt(N_avg)`. So the document text is
   most likely a typo. `af_estimation` defaults to `"outside"`.

2. **CB-SFS parameters are not stated** in either results document —
   `N_seg`, the number of observed symbols `M`, `N_T`, and the
   cycle-frequency grid resolution. As the table above shows, `N_seg`
   alone moves the error by two orders of magnitude.

3. **Peak refinement.** [CBSFS] searches a grid of `N_seg` points; that
   resolution is far coarser than the Doppler values here, so sub-grid
   parabolic refinement is enabled by default. Was it used originally?

4. **`T_s` reference differs between the documents** — the 22-Jan document
   uses `(1 + a_min)`, the 31-Jan one `(1 + a_true)`. Both are supported
   via `ts_reference`.

5. **Negative time indices.** The averaging sum runs over
   `i = -(N_avg-1)/2 .. +(N_avg-1)/2`, so `t_sel + i * T_sr` goes negative
   for the first half — where no signal exists. The record must be offset
   so all `N_avg` symbols are present; otherwise roughly half the average
   is zeros.
