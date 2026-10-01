"""Closed-form predicted bounds from the paper's theorems, for a given
(problem, beta, gamma, eta, V) configuration. These are plotted alongside
the empirical curves produced by algorithm.run_dpp.
"""
import numpy as np


def predicted_constants(problem, beta, gamma, eta, V):
    """The V-dependent but t-independent constants used by every bound below."""
    D2 = problem.D ** 2
    xi_bar = problem.xi - gamma * D2
    assert xi_bar > 0, "gamma is outside the valid stabilization window (Assumption 7)"

    P_min = problem.P_min_bound
    P_max = problem.P_max_bound(beta)
    P_slater = problem.P(problem.x_slater, beta)
    C_circ = P_slater - P_min + D2 / (2 * eta)
    q_bar = 2 * C_circ / xi_bar + problem.H

    rho_P = problem.rho_f  # rho_g = 0, so rho_P = rho_f + beta*rho_g = rho_f
    c0 = 1.0 / eta - rho_P / 2.0
    assert c0 > 0, "eta is outside the valid range (0, 1/rho_P) (Assumption 7)"
    C_S = P_max - P_min + 2 * q_bar * problem.H

    A = 1.0 / eta + 2 * gamma * q_bar
    kappa_g = problem.kappa_g(beta)
    delta_g = kappa_g - problem.L_f - q_bar * problem.L_h  # > 0 needed for Theorem 3

    return dict(xi_bar=xi_bar, C_circ=C_circ, q_bar=q_bar, c0=c0, C_S=C_S,
                A=A, kappa_g=kappa_g, delta_g=delta_g)


def bounds_vs_t(problem, beta, gamma, eta, V, t_array):
    """Predicted bounds as a function of the horizon t, for a fixed V.

    queue_bound: Lemma 2's ceiling on Q_t (pointwise, does not depend on t).
    time_avg_h_bound: Theorem 1, eq. (8).
    movement_bound: Theorem 1, eq. (9), on the running average of ||x_{t+1}-x_t||^2.
    residual_sq_bound: Theorem 2, eq. (12), on E[R(x_tau+1, q_tau)^2].
    g_violation_bound: Theorem 3, eq. (13), on E[g(x_tau+1)]_+ (None if Assumption 6's
        threshold kappa_g > L_f + q_bar*L_h does not hold at this configuration).
    """
    c = predicted_constants(problem, beta, gamma, eta, V)
    t = np.asarray(t_array, dtype=float)

    queue_bound = 2 * V * c["C_circ"] / c["xi_bar"] + problem.H
    time_avg_h_bound = queue_bound / t
    movement_bound = c["C_S"] / (c["c0"] * t) + problem.H ** 2 / (c["c0"] * V)
    residual_sq_bound = c["A"] ** 2 * c["C_S"] / (c["c0"] * t) + c["A"] ** 2 * problem.H ** 2 / (c["c0"] * V)

    g_violation_bound = None
    if c["delta_g"] > 0:
        g_violation_bound = problem.G_g * residual_sq_bound / c["delta_g"] ** 2

    return dict(queue_bound=queue_bound, time_avg_h_bound=time_avg_h_bound,
                movement_bound=movement_bound, residual_sq_bound=residual_sq_bound,
                g_violation_bound=g_violation_bound, constants=c)
