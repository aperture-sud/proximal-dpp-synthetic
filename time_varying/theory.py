"""Predicted bounds for the time-varying run (h_t = h + psi_t), from
Lemma 3 and Theorems 1-3 of the confirmed paper draft.

Two differences from static/theory.py's time-invariant bounds:

1. xi and H are replaced by worst-case values over the whole psi_t run --
   xi_eff = xi - max(psi_t) (Assumption 8's common Slater margin) and
   H_bound = H_static + max(|psi_t|) (Assumption 4's bound on h_t, not h).

2. Theorem 2's movement/residual bounds pick up two extra terms,
   q_bar*V_T/(c0*t) and D^2/t, where V_T is the cumulative functional
   variation of h_t (drift.functional_variation). Theorem 1's
   queue/time-average-h bounds have no V_T term, as stated in the paper.

q_bar here is the V-dependent quantity 2*C_circ/xi_bar + H_bound/V. This is
the direct consequence of Lemma 3's proof (Qt <= 2*V*C_circ/xi_bar + H_bound
holds for every V > 0; dividing by V gives qt <= 2*C_circ/xi_bar + H_bound/V
without ever assuming V >= 1). The paper states the V-independent
q_bar = 2*C_circ/xi_bar + H_bound only as the V >= 1 special case of this
(where H_bound/V <= H_bound). Using the general form lets the same formulas
cover V < 1 without extending anything unproven.

No hard asserts here (unlike static/theory.py): a caller sweeping across a
deliberately invalid gamma/eta/xi is exactly the point of the "violation"
experiments, so invalid configurations return bounds=None with the reasons
recorded, instead of crashing.
"""
import numpy as np


def check_assumptions(problem, gamma, eta, xi_eff):
    """Assumption 7 (gamma and eta windows), evaluated against the
    worst-case xi_eff rather than the static problem.xi.
    """
    D2 = problem.D ** 2
    gamma_lo_ok = gamma > problem.rho_h / 2.0
    xi_bar = xi_eff - gamma * D2
    gamma_hi_ok = xi_bar > 0
    rho_P = problem.rho_f
    mu0 = 1.0 / eta - rho_P
    eta_ok = mu0 > 0
    return dict(gamma_lo_ok=gamma_lo_ok, gamma_hi_ok=gamma_hi_ok, xi_bar=xi_bar,
                eta_ok=eta_ok, mu0=mu0, all_ok=gamma_lo_ok and gamma_hi_ok and eta_ok)


def predicted_constants_tv(problem, beta, gamma, eta, V, H_bound, xi_eff):
    checks = check_assumptions(problem, gamma, eta, xi_eff)
    xi_bar = checks["xi_bar"]

    P_min = problem.P_min_bound
    P_max = problem.P_max_bound(beta)
    P_slater = problem.P(problem.x_slater, beta)
    D2 = problem.D ** 2
    C_circ = P_slater - P_min + D2 / (2 * eta)

    rho_P = problem.rho_f
    c0 = 1.0 / eta - rho_P / 2.0

    valid = checks["all_ok"]
    if not valid:
        return dict(checks, C_circ=C_circ, c0=c0, q_bar=None, C_S=None, A=None,
                    kappa_g=None, delta_g=None, valid=False)

    q_bar = 2 * C_circ / xi_bar + H_bound / V
    C_S = P_max - P_min + 2 * q_bar * H_bound
    A = 1.0 / eta + 2 * gamma * q_bar
    kappa_g = problem.kappa_g(beta)
    delta_g = kappa_g - problem.L_f - q_bar * problem.L_h

    return dict(checks, C_circ=C_circ, c0=c0, q_bar=q_bar, C_S=C_S, A=A,
                kappa_g=kappa_g, delta_g=delta_g, valid=True)


def bounds_vs_t_tv(problem, beta, gamma, eta, V, t_array, V_T_array, H_bound, xi_eff):
    """queue_bound: Lemma 3 (pointwise, constant in t).
    time_avg_h_bound: Theorem 1.
    movement_bound / residual_sq_bound: Theorem 2, with the V_T/t drift term.
    g_violation_bound: Theorem 3 (None if delta_g <= 0).
    """
    c = predicted_constants_tv(problem, beta, gamma, eta, V, H_bound, xi_eff)
    if not c["valid"]:
        return dict(queue_bound=None, time_avg_h_bound=None, movement_bound=None,
                    residual_sq_bound=None, g_violation_bound=None, constants=c)

    t = np.asarray(t_array, dtype=float)
    VT = np.asarray(V_T_array, dtype=float)

    queue_bound = 2 * V * c["C_circ"] / c["xi_bar"] + H_bound
    time_avg_h_bound = queue_bound / t
    movement_bound = (c["C_S"] / (c["c0"] * t) + H_bound ** 2 / (c["c0"] * V)
                       + c["q_bar"] * VT / (c["c0"] * t) + problem.D ** 2 / t)
    residual_sq_bound = c["A"] ** 2 * movement_bound

    g_violation_bound = None
    if c["delta_g"] > 0:
        g_violation_bound = problem.G_g * residual_sq_bound / c["delta_g"] ** 2

    return dict(queue_bound=queue_bound, time_avg_h_bound=time_avg_h_bound,
                movement_bound=movement_bound, residual_sq_bound=residual_sq_bound,
                g_violation_bound=g_violation_bound, constants=c)
