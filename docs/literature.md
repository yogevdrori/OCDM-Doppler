# Literature notes: Doppler-scale estimation (working notes)

Working notes behind Section 2 ("State of the Art") of the report. Status per
entry: **read** (full text read), **abstract** (abstract / secondary
description only), **to read**. Do not cite details marked "to verify".

Key observation: in our signal model the Doppler factor `a` acts on the
baseband waveform as a sampling frequency offset (SFO) of relative size `a`,
and on the carrier as a CFO `a * f_Tx`. So both the Doppler-scale literature
(mostly underwater acoustics, UWA) and the SFO literature are relevant.

## A. Preamble- and pilot-based Doppler-scale estimation (UWA)

- **Sharif, Neasham, Hinton, Adams, IEEE JOE 25(1):52-61, Jan. 2000**, "A
  computationally efficient Doppler compensation system for underwater
  acoustic communications". *read*.
  - Two LFM chirps (10 ms), one at the start and one at the end of each data
    block; a single matched filter measures the received block duration; the
    Doppler is the relative change of the duration ("block Doppler
    estimation"). Compensation by linear interpolation (sample-rate
    conversion), on the passband signal.
  - Alternative discussed and rejected: a bank of correlators with
    Doppler-shifted replicas (cross-ambiguity function); +/-1 % range at
    0.02 % resolution would need 100 correlators.
  - Required accuracy: a DFE with embedded PLL tolerated a fractional Doppler
    of about 1e-4 (10 kHz bandwidth, earlier trials), so the estimator must be
    accurate to about 1e-4. Our loop: RMSE about 4e-6 at the WP.
  - Acceleration: the estimate is the mean over the block (mid-point
    velocity). Remedies: shorter blocks, or fit the estimates of several
    successive blocks to get the acceleration and use a time-varying
    interpolation factor. (Our loop does this tracking continuously.)
  - Also states that a PLL loop bandwidth able to track Doppler of ~1 % is very
    hard to achieve (carrier PLL, not a scale tracker).
  - Sea trial, North Sea: 16 kbit/s, platform up to +/-2.6 m/s.
- **Li, Zhou, Stojanovic, Freitag, Willett, IEEE JOE 33(2):198-209, 2008**,
  "Multicarrier communication over UWA channels with nonuniform Doppler
  shifts". Two steps: coarse scale from the packet duration between a known
  preamble and postamble, resampling; then residual CFO from null subcarriers
  (1-D search). Per packet / per block.
- **Mason, Berger, Zhou, Willett, IEEE JSAC 26(9):1638-1649, Dec. 2008**,
  "Detection, synchronization, and Doppler scale estimation with multicarrier
  waveforms in UWA communication". *read*.
  - Preamble: two identical OFDM symbols (K0 subcarriers) with a CP; data in
    ZP-OFDM after it.
  - Receiver: bank of L self-correlators; branch l uses window N_l (candidate
    period). Metric M(N_l, d) = sum y*[i] y[i+N_l] / sqrt(energy * energy),
    normalised to [0, 1] (CFAR threshold independent of noise). Detection if
    max over branches > threshold. Estimate a = (lambda K0 - N_hat)/N_hat
    from the best branch (lambda = oversampling factor).
  - Resolution: 1/(lambda K0) (N_l integer). Refinement: second, finer grid
    around the coarse estimate, or 3-point interpolation (Eq. 54) between the
    best branch and its neighbours.
  - Complexity: recursive sliding sums; per delay and per branch at most 7
    complex multiplications, 6 complex additions, 1 square root, 1 division.
  - Simulation: K0 = 512, lambda = 8, coarse grid 1.46 m/s (about 1e-3 in
    scale), fine grid 0.366 m/s; RMSE of velocity vs speed at 0 and 10 dB
    (Fig. 7: given only as a figure). Data demodulation needs a mismatch below
    0.3-0.5 m/s (K = 1024).
  - Experiment (Buzzards Bay 2006, transmitter passing the receiver): within
    0.5 m/s of the LFM pre/postamble method, similar BER, no need to buffer
    the packet.
  - Limitation stated by the authors: point estimate at the preamble only;
    for a fast-changing Doppler, shorten packets or insert several sync blocks
    and use a tracking algorithm.
  - **Candidate preamble-based baseline.** For OCDM it needs a dedicated
    preamble (two identical OCDM symbols + CP); our data frame has none.
- **Wan, Wang, Zhou, Yang, Shi, J. Electr. Comput. Eng. 2012, Art. 703243**,
  "Performance comparison of Doppler scale estimation methods for UWA OFDM".
  *read (open access)*. Compares five grid-search estimators:
  1. CP-OFDM preamble, self-correlation (blind w.r.t. the waveform);
  2. CP-OFDM preamble, cross-correlation with prescaled replicas;
  3. ZP-OFDM, null-subcarrier energy (blind), resampling over tentative scales;
  4. ZP-OFDM, pilot-aided cross-correlation;
  5. ZP-OFDM, decision-aided (re-encoded block as template).
  Findings: correlation-based methods best at low SNR; blind ones equal or
  better at medium/high SNR and no error floor in multipath; all reach RMSE
  below 0.1 m/s in common-Doppler multipath (c = 1500 m/s, so 6.7e-5 in
  scale). Includes the CRLB for a known waveform, single path. Parameters:
  fc = 13 kHz, B = 4.88 kHz, 512 / 1024 subcarriers, speeds within +/-4.5 m/s
  (|a| <= 3e-3). **Useful as the template for a fair comparison and for the
  CRLB reference.**

## B. Cyclostationarity-based (blind)

- **Zhang, Wang, Wang, Zheng, Zhuang, Xu, Proc. Meet. Acoust. 30, 055006,
  2017**, "Cyclic-feature based Doppler scale estimation for OFDM signals
  over doubly selective UWA channels". *read (open access)*.
  - CFDE: CAF of the received signal; the cycle frequencies are multiples of
    (1 + gamma)/T; the estimate is the peak of the CAF magnitude over the first
    cycle frequency, searched on a grid within [gamma_min, gamma_max].
  - CPDE: peak over the delay of the zero-cycle-frequency CAF (CP
    correlation); not usable for ZP-OFDM.
  - Blind, no preamble. Resolution delta_gamma = 1/(rho N) (rho oversampling,
    N subcarriers); finer resolution costs more.
  - Setup: B = 5 kHz, fc = 15 kHz, N = 128, rho = 8, 64-QAM, CP = N/4, 5-tap
    channel, gamma = 1e-3 (also 1.5e-3, 2e-3), SNR -14 .. 8 dB, 2 or 4 blocks.
  - Metric: probability of a correct estimate (within +/- 1/(rho N)); no MSE.
    Simulation only. Asynchronous sampling (almost-cyclostationarity) is not
    treated.
  - **Closest method of our family; proposed blind baseline.** It is
    essentially an open-loop grid search of a CAF peak, i.e. what our loop
    replaces by two evaluations per block.
- **Socheleau, IEEE JOE 50(2):425-447, Apr. 2025 (early access Dec. 2022),
  doi 10.1109/JOE.2022.3218106**, "Cyclostationarity of communication signals
  in UWA channels". *read (Sections I-III, Application 1)*.
  - Model: motion gives a time warp psi(t) = (1 - v/c) t - (a/2c) t^2 per path
    (velocity and acceleration). The received signal is a time-warped
    cyclostationary process (Napolitano 2017); the autocorrelation becomes a
    sum of chirps in t; the spectral correlation function does not exist when
    the acceleration is non-zero. (Matches our formulation u(t) = t + A(t).)
  - Estimator "CSB" (single scale): uses the zero-lag autocorrelation
    estimate |r(t)|^2 (same feature as our Delta = {0}). Cost
    J(mu1, mu2) = |(1/Tobs) int |r(t)|^2 exp(-i 2pi/Ts (mu1 t + mu2 t^2)) dt|^2,
    maximised by a coarse grid (steps 1/6 m/s, 1/2 m/s^2) and Barzilai-Borwein
    gradient ascent (about 20 iterations). Non-data-aided. Complexity
    O(Nobs log Nobs) with FFTs. Multiscale extension by iterative peeling
    (Algorithm 1).
  - Setup: RRC QPSK, Ts = 0.2 ms, fc = 20 kHz, Tobs = 1 s, |v| <= 10 m/s,
    |a| <= 1 m/s^2, Bellhop shallow-water channel; metric = normalised RMSE of
    psi(t) over the window, vs Eb/N0 (Fig. 6, figure only).
  - Result: slightly better than the multibranch autocorrelation (MBA) method
    of Li, Zakharov, Henson 2018, which needs repeated pilot blocks; both have
    an error floor from multipath.
  - **MATLAB code published**: github.com/fxsocheleau/cyclostationarity-uac
    (also CodeOcean capsule 2187158).
  - **Closest work to ours; proposed blind baseline.** Differences: feed-
    forward per window (1 s) with a 2-D search, symbol-rate cycle frequency of
    PSK/QAM (1/Ts); ours tracks block by block (6 ms) with two cost
    evaluations, OCDM with zero GI (cycle frequency 1/T_sym).
- **Napolitano, IEEE TSP 65(20):5526-5541, Oct. 2017**, "Time-warped
  almost-cyclostationary signals: characterization and statistical function
  measurements". *cited by Socheleau; to read*. Theory for an almost-
  cyclostationary signal under a time warp: our case (asynchronous sampling
  plus time-varying a).
- **Li, Zakharov, Henson, IEEE JOE 43(4):1099-1113, Oct. 2018**, "Multibranch
  autocorrelation method for Doppler estimation in UWA channels". *description
  in Socheleau only*. Velocity and acceleration from repeated (unknown)
  pilot blocks; halves the data rate in the configuration compared there.
- **Kumar, Dabora, IEEE Access 7:100692-100705, 2019** (CB-SFS). Blind SFO
  estimator, segmented + autocorrelated CAF, handles almost-cyclostationarity.
  Source of our cost function.
- **Huang, Qiu, Zhu, VTC 2004**: Doppler frequency offset in CDMA from a
  cycle frequency of the CAF; narrowband.
- **Bolcskei, IEEE Trans. Commun. 49(6):988-999, Jun. 2001**: blind symbol
  timing and CFO for OFDM from cyclostationarity (second-order statistics).
  *abstract*.

## C. SFO / CFO estimation in OFDM (from Rotem & Dabora 2020, read)

- **Rotem, Dabora, IEEE Access 8:194978-194991, 2020**. Pilot-assisted,
  sequential SFO then CFO; a cost function evaluated at a few values of an
  auxiliary parameter gives an explicit estimate (no search). Section IV has
  a complexity analysis in real multiplications for the proposed estimator
  and for the baselines: **the template for our complexity comparison**.
  Baselines there: LSE (Jung et al., IEEE Access 2018) and the blind
  estimator of Li et al. (WCNC 2017); CRB from Murin & Dabora.
- **Morelli, Moretti, IEEE TWC 9(4):1514-1524, 2010**: pilot-aided,
  ML-oriented, 1-D grid search.
- **Murin, Dabora, WCMC 16(9):1018-1034, 2016**: data-aided, Taylor expansion
  of the ML cost, polynomial roots; CRB.
- **Jeon, Kim, Serpedin, IEEE Trans. Commun. 59(4):1133-1141, 2011**: blind
  CFO from three values of a cost function. Conceptually close to our
  two-point discriminator.

## D. OCDM

- **Ouyang, Zhao, IEEE Trans. Commun. 64(9), 2016**: OCDM.
- **Bai, Bouvet, Sensors 18(11):3815, 2018**: OCDM for UWA. How Doppler is
  handled there: to verify (full text not read).
- **Zegrar, Arslan, IEEE WCL 13(3), 2024; Haif, Zegrar, Arslan, IEEE TVT
  73(8), 2024**: per-path delay and Doppler *shift* in doubly-dispersive OCDM
  with embedded pilots; narrowband Doppler model.
- Not found so far: a blind Doppler-scale estimator for OCDM, or a closed-loop
  Doppler-scale tracker driven by a cyclostationary feature.

## Proposed baselines (to be confirmed with the advisor)

1. **Blind, same feature, time-varying Doppler:** Socheleau's CSB estimator
   (velocity + acceleration per window), applied to our OCDM signal with the
   same observation length; the published MATLAB code can be the reference
   implementation. Zhang et al. 2017 is its constant-Doppler special case
   (grid search of the CAF peak).
   Open points: window length vs our loop memory; cycle frequency 1/T_sym
   instead of 1/Ts; complexity in real multiplications per sample (FFT size,
   grid size, iterations).
2. **With training overhead:** Mason et al. 2008 (preamble of two identical
   symbols + bank of self-correlators + interpolation). Needs a preamble added
   to our frame; complexity per sample from the paper (7 CM per branch per
   delay). Gives one estimate per packet, so on a time-varying Doppler it
   shows the cost of not tracking.
3. **Bound:** CRLB of the Doppler scale (Wan et al. 2012 give one for a known
   waveform) as a reference line.

Full texts of Sharif 2000, Mason 2008 and Socheleau read Nov 2026.
Search log (Nov 2026): web searches on Doppler-scale estimation for OCDM /
OFDM / UWA, cyclostationary Doppler estimation, closed-loop Doppler-scale
tracking.
