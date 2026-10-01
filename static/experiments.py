"""Sweep runners: produce the (history, predicted bounds) pairs that
plotting.py turns into each theorem's figure.
"""
import numpy as np

from static.algorithm import run_dpp
from static.theory import bounds_vs_t

# Default operating point, used whenever a sweep holds the other parameters fixed.
# eta=1.0 (rather than a value near 0) keeps D^2/(2 eta) -- and hence q_bar -- small;
# gamma=0.04 (near the low end of its window) leaves most of the Slater margin
# unconsumed. Both matter only for how tight the plotted bounds are, not for
# correctness.
BETA = 45.0
GAMMA = 0.04   # inside the (rho_h/2, xi/D^2) = (0.0375, 0.1) window
ETA = 1.0      # inside the (0, 1/rho_f) = (0, 11.57) window
V = 20
T = 600

V_GRID = [2, 5, 10, 20, 40, 80]
BETA_GRID = [25.0, 35.0, 50.0, 70.0]
GAMMA_GRID = [0.04, 0.055, 0.07, 0.09]
ETA_GRID = [0.3, 0.6, 1.0, 1.5]
EPSILON_GRID = [0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02]


def sweep_V(problem, V_grid=V_GRID, T=T, beta=BETA, gamma=GAMMA, eta=ETA):
    """One run per V, long enough to read off both the 'vs t' and 'vs V' curves."""
    runs = []
    for v in V_grid:
        history = run_dpp(problem, beta, gamma, eta, v, T)
        bounds = bounds_vs_t(problem, beta, gamma, eta, v, np.arange(1, T + 1))
        runs.append(dict(V=v, history=history, bounds=bounds))
    return runs


def sweep_beta(problem, beta_grid=BETA_GRID, T=T, V=V, gamma=GAMMA, eta=ETA):
    """Ablation runs only feed plot_ablation, which reads final metrics from
    history, not the predicted bounds -- so no bounds_vs_t call is needed here.
    """
    runs = []
    for beta in beta_grid:
        history = run_dpp(problem, beta, gamma, eta, V, T)
        runs.append(dict(beta=beta, history=history))
    return runs


def sweep_gamma(problem, gamma_grid=GAMMA_GRID, T=T, V=V, beta=BETA, eta=ETA):
    runs = []
    for gamma in gamma_grid:
        history = run_dpp(problem, beta, gamma, eta, V, T)
        runs.append(dict(gamma=gamma, history=history))
    return runs


def sweep_eta(problem, eta_grid=ETA_GRID, T=T, V=V, beta=BETA, gamma=GAMMA):
    runs = []
    for eta in eta_grid:
        history = run_dpp(problem, beta, gamma, eta, V, T)
        runs.append(dict(eta=eta, history=history))
    return runs


def sweep_epsilon(problem, eps_grid=EPSILON_GRID, beta=BETA, gamma=GAMMA, eta=ETA):
    """Corollary 1: V = ceil(eps^-2), T = ceil(eps^-3), one run per eps."""
    runs = []
    for eps in eps_grid:
        v = int(np.ceil(eps ** -2))
        t_max = int(np.ceil(eps ** -3))
        history = run_dpp(problem, beta, gamma, eta, v, t_max)
        runs.append(dict(eps=eps, V=v, T=t_max, history=history))
    return runs
