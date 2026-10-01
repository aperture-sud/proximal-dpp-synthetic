"""Sweep runners for the time-varying experiments.

Default operating point, adjusted for the much-less-convex f/h in
time_varying/problem.py (rho_f=3.27, rho_h=0.12 -- vs. 0.086/0.075
static) and the psi_t drift: beta=679, gamma=0.065, eta=0.2, V=20, T=4000,
psi_max=0.08 (xi_eff=0.32).

Every number below was picked by actually running the instance first, not
guessed by hand:
  - rho_f=3.2736 needs eta < 1/rho_f=0.3055; eta=0.2 keeps a clear margin.
  - gamma=0.065 sits inside (rho_h/2, xi_eff/D^2) = (0.0602, 0.08) at
    psi_max=0.08.
  - Theorem 3's threshold beta* (where delta_g crosses 0) is 568.8 at V=20
    and rises as V shrinks, via the H_bound/V term in q_bar, to 590.4 at
    V=0.1, the worst case over V_GRID; BETA=679 (~1.15x) keeps delta_g>0
    across the whole V_GRID below.
  - the REAL exact-penalty threshold (where g(x) actually stops going
    positive) is between beta=0.4 and beta=0.5 -- over 1000x smaller than
    beta*. Theorem 3's threshold is a sufficient, very conservative
    condition; the beta sweep below is chosen to show both numbers on the
    same figure.
"""
import numpy as np

from time_varying.drift import make_psi_drift, make_psi_burst, functional_variation
from time_varying.algorithm import run_dpp_tv
from time_varying.theory import bounds_vs_t_tv, predicted_constants_tv
from time_varying.problem import build_problem_tv

BETA = 679.0
GAMMA = 0.065
ETA = 0.2
V = 20.0
T = 4000

PSI_MAX_VALID = 0.08      # xi_eff = 0.4 - 0.08 = 0.32 > 0: Assumption 8 holds
PSI_MAX_SLATER_VIOLATE = 0.6   # xi_eff = 0.4 - 0.6 = -0.2 < 0: Assumption 8 fails
PSI_SEED = 11

V_GRID = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]
BETA_GRID_BASE = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 20, 50, 100, 300, 679, 800]


def beta_star(problem, gamma=GAMMA, eta=ETA, V=V, psi_max=PSI_MAX_VALID):
    """Closed-form Theorem-3 threshold: delta_g(beta) = 2*beta*r_g - L_f -
    q_bar*L_h is linear in beta here (q_bar does not depend on beta, since
    the Slater point 0 has g(0) < 0 so P(x_slater, beta) = f(0) regardless
    of beta). Solving delta_g = 0 gives beta* in closed form.
    """
    xi_eff = problem.xi - psi_max
    H_bound = problem.H + psi_max
    c = predicted_constants_tv(problem, beta=1.0, gamma=gamma, eta=eta, V=V,
                                H_bound=H_bound, xi_eff=xi_eff)
    return (problem.L_f + c["q_bar"] * problem.L_h) / (2 * problem.r_g)


def _one_run(problem, psi, beta, gamma, eta, V, T, psi_max, catch_breakdown=False):
    hist = run_dpp_tv(problem, psi, beta, gamma, eta, V, T, catch_breakdown=catch_breakdown)
    _, VT_cum = functional_variation(psi)
    xi_eff = problem.xi - psi_max
    H_bound = problem.H + psi_max
    bounds = bounds_vs_t_tv(problem, beta, gamma, eta, V, np.arange(1, T + 1),
                             VT_cum[:T], H_bound, xi_eff)
    return hist, bounds


def sweep_V_tv(problem, V_grid=V_GRID, T=T, beta=BETA, gamma=GAMMA, eta=ETA,
               psi_max=PSI_MAX_VALID):
    """Same psi_t realization (same seed) reused for every V, so differences
    across the sweep come only from V, not from a different random draw.
    """
    runs = []
    for v in V_grid:
        rng = np.random.default_rng(PSI_SEED)
        psi = make_psi_drift(T, psi_max=psi_max, rng=rng)
        hist, bounds = _one_run(problem, psi, beta, gamma, eta, v, T, psi_max)
        runs.append(dict(V=v, history=hist, bounds=bounds, psi=psi))
    return runs


SIGMA_GRID = [0.0005, 0.002, 0.006, 0.012, 0.025, 0.05, 0.1]  # realized V_T(T=4000) ~ 0.3 to 114


def sweep_VT_tv(problem, sigma_grid=SIGMA_GRID, T=T, beta=BETA, gamma=GAMMA, eta=ETA,
                 V=V, psi_max=PSI_MAX_VALID):
    """V and T held fixed; only the drift's noise level sigma is swept, which
    changes how much h_t actually varies (V_T) without touching psi_max (so
    Assumption 8's Slater margin xi_eff = xi - psi_max stays the same valid
    value across every run here). This isolates Theorem 2/3's V_T term: the
    only thing changing run to run is how jumpy psi_t is, not the queue
    weight V or the horizon T.
    """
    runs = []
    for sigma in sigma_grid:
        rng = np.random.default_rng(PSI_SEED)
        psi = make_psi_drift(T, psi_max=psi_max, rng=rng, sigma=sigma)
        hist, bounds = _one_run(problem, psi, beta, gamma, eta, V, T, psi_max)
        _, VT_cum = functional_variation(psi)
        runs.append(dict(sigma=sigma, V_T=VT_cum[T - 1], history=hist, bounds=bounds, psi=psi))
    return runs


def sweep_beta_tv(problem, beta_grid=None, T=T, V=V, gamma=GAMMA, eta=ETA,
                   psi_max=PSI_MAX_VALID):
    """beta_grid defaults to BETA_GRID_BASE with the exact Theorem-3
    threshold beta_star inserted, so the sweep always straddles it exactly.
    """
    if beta_grid is None:
        star = beta_star(problem, gamma=gamma, eta=eta, V=V, psi_max=psi_max)
        beta_grid = sorted(BETA_GRID_BASE + [star])
    runs = []
    for beta in beta_grid:
        rng = np.random.default_rng(PSI_SEED)
        psi = make_psi_drift(T, psi_max=psi_max, rng=rng)
        hist, bounds = _one_run(problem, psi, beta, gamma, eta, V, T, psi_max)
        runs.append(dict(beta=beta, history=hist, bounds=bounds))
    return runs


def headline_trajectory(problem, T=T, beta=BETA, gamma=GAMMA, eta=ETA, V=V,
                         psi_max=0.09, seed=2, n_bursts=5):
    """One run meant to show the mechanism plainly: f finds a deep basin
    while psi_t is near 0, then a burst pushes psi_t (hence h_t) up, the
    queue spikes, and the next rounds are pulled back toward feasibility,
    raising f again -- the 'aggressive descent, then the bill arrives' story.
    """
    rng = np.random.default_rng(seed)
    psi = make_psi_burst(T, psi_max=psi_max, rng=rng, n_bursts=n_bursts)
    hist, bounds = _one_run(problem, psi, beta, gamma, eta, V, T, psi_max)
    f_vals = np.array([problem.f(x) for x in hist["x"]])
    return dict(history=hist, bounds=bounds, psi=psi, f=f_vals)


def assumption_violations(problem, T=6000, beta=BETA, eta=ETA, V=V):
    """Three deliberately-invalid configurations, each isolating one
    assumption, plus the valid baseline for reference. All share the same
    psi_max=PSI_MAX_VALID drift realization except 'slater_violated', which
    needs psi_t to actually exceed xi to break Assumption 8.
    """
    runs = {}

    rng = np.random.default_rng(PSI_SEED)
    psi_valid = make_psi_drift(T, psi_max=PSI_MAX_VALID, rng=rng)
    hist, bounds = _one_run(problem, psi_valid, beta, GAMMA, eta, V, T, PSI_MAX_VALID)
    runs["valid"] = dict(history=hist, bounds=bounds, psi=psi_valid,
                          label=f"valid (gamma={GAMMA}, eta={eta}, psi_max={PSI_MAX_VALID})")

    # Assumption 5/8 (Slater margin): psi_t pushed past xi=0.4, so no common
    # xi>0 exists with h_t(0) <= -xi for every t.
    rng = np.random.default_rng(PSI_SEED)
    psi_bad = make_psi_drift(T, psi_max=PSI_MAX_SLATER_VIOLATE, rng=rng, bias=0.0015, sigma=0.02)
    hist_bad = run_dpp_tv(problem, psi_bad, beta, GAMMA, eta, V, T)
    runs["slater_violated"] = dict(history=hist_bad, psi=psi_bad,
                                    label=f"Slater violated (psi_max={PSI_MAX_SLATER_VIOLATE} > xi={problem.xi})")

    # Assumption 7 (eta window): eta chosen above 1/rho_f, so mu_0 <= 0 at
    # the very first step (q_1=0), and the subproblem's strong-convexity
    # check fails immediately.
    eta_bad = 1.0 / problem.rho_f + 0.13  # ~0.435, past 1/rho_f=0.305
    hist_eta = run_dpp_tv(problem, psi_valid, beta, GAMMA, eta_bad, V, T, catch_breakdown=True)
    runs["eta_too_large"] = dict(history=hist_eta, eta=eta_bad,
                                  label=f"eta={eta_bad:.3f} > 1/rho_f={1/problem.rho_f:.3f}")

    # Assumption 7 (gamma lower bound), forced to actually bite: combine a
    # too-small gamma with the Slater-violating psi (which drives q_t far
    # higher than it would ever reach under a valid Slater margin) and a
    # small V, so q_t crosses the point where mu_t = mu_0 + q_t*(2*gamma-rho_h)
    # turns negative.
    gamma_bad = 0.001
    V_small = 2.0
    hist_gamma = run_dpp_tv(problem, psi_bad, beta, gamma_bad, eta, V_small, T, catch_breakdown=True)
    runs["gamma_too_low"] = dict(history=hist_gamma, gamma=gamma_bad, eta=eta, V=V_small,
                                  label=f"gamma={gamma_bad} < rho_h/2={problem.rho_h/2:.4f} (+ Slater-violating psi, V={V_small})")

    return runs


# ----------------------------------------------------------------------
# Theorem 3 "dangerous" demo: a separate, tighter instance (smaller r_g,
# stronger f pull -- see build_problem_tv's f_pull/r_g) run at a
# beta BELOW its own Theorem-3 threshold, so g is actually violated instead
# of sitting at floating-point noise. This does not touch the shared
# default problem/beta used by the Theorem 1/2 figures above -- those
# don't need g to be at risk, and beta=679 there was chosen specifically
# so Theorem 3 holds, which provably forces [g]_+ -> 0 (see delta_g in
# time_varying/theory.py): a meaningful violation plot and a valid Theorem-3 bound
# cannot coexist in the same run, so this is deliberately a different
# (beta, instance) pair, clearly below its own threshold.
DANGER_R_G = 0.55
DANGER_F_PULL = 1.2
DANGER_BETA = 0.4    # true violation edge between 1.0 and 1.2, Theorem 3's own beta* ~820


def build_danger_problem(seed=0):
    return build_problem_tv(seed=seed, r_g=DANGER_R_G, f_pull=DANGER_F_PULL)


def sweep_V_tv_danger(problem, V_grid=V_GRID, T=T, beta=DANGER_BETA, gamma=GAMMA, eta=ETA,
                       psi_max=PSI_MAX_VALID):
    return sweep_V_tv(problem, V_grid=V_grid, T=T, beta=beta, gamma=gamma, eta=eta, psi_max=psi_max)


def sweep_VT_tv_danger(problem, sigma_grid=SIGMA_GRID, T=T, beta=DANGER_BETA, gamma=GAMMA, eta=ETA,
                        V=V, psi_max=PSI_MAX_VALID):
    return sweep_VT_tv(problem, sigma_grid=sigma_grid, T=T, beta=beta, gamma=gamma, eta=eta,
                        V=V, psi_max=psi_max)
