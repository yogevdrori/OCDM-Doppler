"""
Sanity checks for the OCDM Doppler simulation infrastructure.

Each check is independent and prints PASS/FAIL with the numbers, so a
regression anywhere in the signal model, the analytical AF or CB-SFS shows
up immediately.
"""

import numpy as np

from ocdm_doppler import (SystemParams, OCDMSignal, chirp_matrix,
                          analytical_af, cbsfs, preset_noiseless_31jan)

OK, BAD = "PASS", "FAIL"
results = []


def check(name, passed, detail=""):
    results.append((name, passed))
    print(f"[{OK if passed else BAD}] {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------
print("=" * 72)
print("1. Chirp basis")
print("=" * 72)

N = 64
Phi = chirp_matrix(N)
err_unit = np.max(np.abs(Phi.conj().T @ Phi - np.eye(N)))
check("IDFnT is unitary", err_unit < 1e-12, f"max|Phi^H Phi - I| = {err_unit:.2e}")

# chirp -> FFT -> chirp factorisation
rng = np.random.default_rng(0)
D = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
k = np.arange(N); n = np.arange(N)
pre = np.exp(-1j * np.pi / N * k ** 2) * D
inner = np.fft.ifft(pre) * N
x_fast = np.exp(1j * np.pi / 4) * np.exp(-1j * np.pi / N * n ** 2) * inner / np.sqrt(N)
err_fact = np.max(np.abs(Phi @ D - x_fast))
check("chirp-FFT-chirp factorisation", err_fact < 1e-10,
      f"max abs diff = {err_fact:.2e}")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("2. Transmitted signal structure")
print("=" * 72)

p = SystemParams(N_is=16, modulation="qpsk", snr_db=float("inf"))
print(p.summary())
print()

sig = OCDMSignal(p, n_symbols=8, seed=1)
Tsamp = p.T_samp_syn

# guard interval must be zero (g(t)=1 only on [0, T_is))
t_active = (np.arange(p.N_is) + 0.5) * Tsamp            # inside symbol 0
t_guard = p.T_is + (np.arange(p.N_gi_eff) + 0.5) * Tsamp  # inside its GI
x_act = sig.tx(t_active)
x_gi = sig.tx(t_guard)
check("guard interval is zero (not a cyclic prefix)",
      np.allclose(x_gi, 0) and np.all(np.abs(x_act) > 1e-9),
      f"max|X| in GI = {np.abs(x_gi).max():.2e}, "
      f"min|X| in symbol = {np.abs(x_act).min():.3f}")

# sampling X at the transmitter rate inside one symbol must equal Phi @ D
x_samp = sig.tx(np.arange(p.N_is) * Tsamp)
x_ref = chirp_matrix(p.N_is) @ sig.D[0]
err_tx = np.max(np.abs(x_samp - x_ref))
check("X(n T_samp) equals the IDFnT of the symbols", err_tx < 1e-9,
      f"max abs diff = {err_tx:.2e}")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("3. Received signal reduces to the transmitted one")
print("=" * 72)

p_id = SystemParams(N_is=16, A_p0=1.0 + 0j, tau_p0_in_samples=0.0,
                    cfo=0.0, a_max=1e-3, a_true=0.0, snr_db=float("inf"))
sig_id = OCDMSignal(p_id, n_symbols=8, seed=2)
t = np.linspace(0, 3 * p_id.T_sym, 501)
y = sig_id.rx(t, a=0.0, noiseless=True)
x = sig_id.tx(t)
# with A0=1, tau=0, a=0, CFO=0 the receiver front-end contributes exactly 1/2
err_rx = np.max(np.abs(y - x / 2.0))
check("Y(t) = X(t)/2 for a=0, A0=1, tau=0, CFO=0", err_rx < 1e-9,
      f"max abs diff = {err_rx:.2e}")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("4. Empirical AF matches the analytical expression  [OCDM] Eqn. (11)")
print("=" * 72)

p4 = SystemParams(N_is=16, modulation="qpsk", a_max=1e-3, a_true=5e-4,
                  snr_db=float("inf"), rho=32, ts_reference="a_true")
a = p4.a_true_eff
T_sr = p4.T_sym / (1.0 + a)        # received symbol period (a_hat_sr = a)
N_avg = 4096

sigA = OCDMSignal(p4, n_symbols=N_avg + 64, seed=3)

# centre the averaging window so all N_avg symbols exist
t_sel = p4.T_sym / (2.0 * (1.0 + a)) + (N_avg // 2) * T_sr
i_vec = np.arange(-(N_avg // 2), N_avg // 2)

lag_max = analytical_af.first_null_lag(a, p4)
lags = np.linspace(-1.6 * lag_max, 1.6 * lag_max, 65)

emp = np.empty(lags.size)
for j, lag in enumerate(lags):
    i1 = sigA.grid_index(t_sel + i_vec * T_sr)
    i2 = sigA.grid_index(t_sel - lag + i_vec * T_sr)
    y1 = sigA.rx_at_grid(i1, a=a, noiseless=True)
    y2 = sigA.rx_at_grid(i2, a=a, noiseless=True)
    emp[j] = np.abs(np.mean(y1 * np.conj(y2)))

ana = analytical_af.af_magnitude(t_sel, lags, a, p4)

peak_emp, peak_ana = emp.max(), ana.max()
expected_peak = p4.sigma_D2 * abs(p4.A_p0) ** 2 / 4.0
rel = np.abs(emp - ana).max() / peak_ana

check("AF peak equals sigma_D^2 |A0|^2 / 4",
      abs(peak_ana - expected_peak) < 1e-9,
      f"analytical peak = {peak_ana:.4f}, expected = {expected_peak:.4f}")
check("empirical AF matches analytical AF", rel < 0.05,
      f"peak emp/ana = {peak_emp:.4f}/{peak_ana:.4f}, "
      f"max rel. deviation = {rel:.3%}")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("5. CB-SFS: cost-function peaks sit at k (1+a) / N_sym")
print("=" * 72)

for a_test in (0.0, 5e-4):
    p5 = SystemParams(N_is=16, modulation="qpsk", a_max=1e-3, a_true=a_test,
                      snr_db=float("inf"), rho=32, ts_reference="a_min",
                      N_seg=2 ** 8, N_T=2, n_alpha_grid=4096)
    M = 600
    sig5 = OCDMSignal(p5, n_symbols=M + 16, seed=4)
    # sample at the receiver's nominal rate T_samp^(ns) (exact, unquantised)
    y5 = sig5.rx_nominal(M * p5.N_sym, a=a_test, noiseless=True)

    got = np.array([cbsfs.locate_harmonic(y5, p5, k)[0] for k in (1, 2)])
    expect = cbsfs.expected_cycle_frequencies(a_test, p5)
    detail = (f"a={a_test:g}  peaks={np.array2string(got, precision=6)}  "
              f"expected={np.array2string(expect, precision=6)}")
    ok = np.all(np.abs(got - expect) < 0.01 * p5.alpha_1_syn)
    check(f"CB-SFS peaks at expected cycle frequencies (a={a_test:g})", ok, detail)
    res = cbsfs.estimate_doppler_harmonic(y5, p5, N_last=2)
    print(f"        -> a_hat = {res.a_hat:+.6e}   (true a = {a_test:+.6e})")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
n_pass = sum(1 for _, ok in results if ok)
print(f"SUMMARY: {n_pass}/{len(results)} checks passed")
print("=" * 72)
