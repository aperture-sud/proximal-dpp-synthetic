"""Algorithm 1 with a time-varying constraint h_t(x) = h(x) + psi_t.

psi_t is a scalar, constant in x at each round, so it does not change the
subproblem's argmin: solve_subproblem(...) from static/algorithm.py is reused
unchanged. psi_t only enters the virtual-queue update

    Q_{t+1} = [Q_t + h(x_{t+1}) + psi_t]_+,

which is exactly Algorithm 1 applied to h_t. This is also why time variation
still shows up in the trajectory: a jump in psi_t changes Q_{t+1}, which
changes q_{t+1} = Q_{t+1}/V, which changes the weight put on h in the *next*
round's subproblem.
"""
import numpy as np

from static.algorithm import solve_subproblem


def run_dpp_tv(problem, psi, beta, gamma, eta, V, T, n_inner_iters=400,
                catch_breakdown=False):
    """Run Algorithm 1 for T outer steps with h_t = h + psi[t-1] at round t.

    If catch_breakdown is True, an AssertionError raised inside
    solve_subproblem (the subproblem's strong-convexity check, "mu > 0")
    is caught: the run stops at that step and the remaining history is
    filled with NaN, with history['broke_at'] set to the failing t. This is
    used to demonstrate Assumption 7 failing outright, rather than crashing
    the whole sweep.
    """
    assert len(psi) >= T, f"psi sequence too short: len={len(psi)} < T={T}"

    x = np.zeros(problem.d)
    Q = 0.0

    history = {
        "x": np.zeros((T + 1, problem.d)),
        "Q": np.zeros(T + 1),
        "q": np.zeros(T + 1),
        "h": np.zeros(T + 1),      # raw h(x_t), no psi
        "psi": np.zeros(T + 1),    # psi_t used at that round
        "h_t": np.zeros(T + 1),    # h(x_t) + psi_t -- what actually drives Q
        "g": np.zeros(T + 1),
        "P": np.zeros(T + 1),
        "movement": np.zeros(T + 1),
        "residual": np.zeros(T + 1),
        "broke_at": None,
    }

    for t in range(T):
        q_t = Q / V
        psi_t = psi[t]

        try:
            x_next = solve_subproblem(problem, x, q_t, beta, gamma, eta, n_iters=n_inner_iters)
        except AssertionError:
            if not catch_breakdown:
                raise
            history["broke_at"] = t + 1
            for key in ("x", "Q", "q", "h", "psi", "h_t", "g", "P", "movement", "residual"):
                history[key][t + 1:] = np.nan
            break

        h_val = problem.h(x_next)
        h_t_val = h_val + psi_t
        Q_next = max(Q + h_t_val, 0.0)
        movement = float(np.linalg.norm(x_next - x))

        A_t = 1.0 / eta + 2 * gamma * q_t
        residual = A_t * movement

        i = t + 1
        history["x"][i] = x_next
        history["Q"][i] = Q_next
        history["q"][i] = Q_next / V
        history["h"][i] = h_val
        history["psi"][i] = psi_t
        history["h_t"][i] = h_t_val
        history["g"][i] = problem.g(x_next)
        history["P"][i] = problem.P(x_next, beta)
        history["movement"][i] = movement
        history["residual"][i] = residual

        x, Q = x_next, Q_next

    return history
