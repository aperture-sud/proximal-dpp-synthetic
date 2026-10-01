"""Algorithm 1: proximal drift-plus-penalty (paper, Section 3).

At each outer step t we solve

    minimize_{x in X}  P(x) + q_t*(h(x) + gamma*||x-x_t||^2) + (1/2eta)*||x-x_t||^2

by proximal gradient descent. f, h, and the two quadratic stabilizers are
smooth, so they are handled with a gradient step. The only non-smooth piece
is beta*[g(x)]_+ together with the ball constraint X; because g(x) =
||x||^2 - r_g^2 and X are both radially symmetric around the origin, their
combined proximal operator reduces to a 1-D problem in the radius and has a
closed form (see _prox_hinge_ball). This gives linear convergence, unlike a
plain subgradient step: a subproblem near the g(x) = 0 kink was found not
to converge reliably even after 100,000 plain subgradient iterations, while
this solver reaches ~1e-10 precision within a few hundred.
"""
import numpy as np


def project_ball(x, radius):
    norm = np.linalg.norm(x)
    if norm <= radius:
        return x
    return x * (radius / norm)


def _prox_hinge_ball(y, lam, r_g, R_X):
    """argmin_{x in ball(R_X)} lam*max(||x||^2 - r_g^2, 0) + 0.5*||x - y||^2.

    Both terms are radial, so the minimizer lies along y and reduces to a
    1-D convex problem in r = ||x||, solved by comparing the unconstrained
    minimizer of each of its two (quadratic) pieces, clipped to its piece.
    """
    a = np.linalg.norm(y)
    if a == 0.0:
        return y.copy()  # r* = 0 is feasible (r_g > 0) and optimal when y = 0

    r_inner = np.clip(a, 0.0, min(r_g, R_X))
    r_outer = np.clip(a / (1.0 + 2.0 * lam), r_g, R_X)

    def psi(r):
        return lam * max(r ** 2 - r_g ** 2, 0.0) + 0.5 * (r - a) ** 2

    r_star = r_inner if psi(r_inner) <= psi(r_outer) else r_outer
    return (r_star / a) * y


def smooth_gradient(problem, x, x_t, q_t, gamma, eta):
    """Gradient of the smooth part: f(x) + q_t*(h(x) + gamma*||x-x_t||^2) + (1/2eta)*||x-x_t||^2."""
    return (problem.grad_f(x)
            + q_t * (problem.grad_h(x) + 2 * gamma * (x - x_t))
            + (x - x_t) / eta)


def solve_subproblem(problem, x_t, q_t, beta, gamma, eta, n_iters=400, tol=1e-12):
    """Proximal gradient descent on the subproblem, starting from x_t."""
    mu = 1.0 / eta - problem.rho_f + q_t * (2 * gamma - problem.rho_h)
    assert mu > 0, "subproblem is not strongly convex -- check Assumption 7"

    L = problem.smooth_f + q_t * problem.smooth_h + 2 * q_t * gamma + 1.0 / eta
    step = 1.0 / L

    x = x_t.copy()
    for _ in range(n_iters):
        y = x - step * smooth_gradient(problem, x, x_t, q_t, gamma, eta)
        x_new = _prox_hinge_ball(y, lam=beta * step, r_g=problem.r_g, R_X=problem.R_X)
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def run_dpp(problem, beta, gamma, eta, V, T, n_inner_iters=400):
    """Run Algorithm 1 for T outer steps, starting from x_1 = 0, Q_1 = 0."""
    x = np.zeros(problem.d)
    Q = 0.0

    history = {
        "x": np.zeros((T + 1, problem.d)),
        "Q": np.zeros(T + 1),
        "q": np.zeros(T + 1),
        "h": np.zeros(T + 1),
        "g": np.zeros(T + 1),
        "P": np.zeros(T + 1),
        "movement": np.zeros(T + 1),
        "residual": np.zeros(T + 1),
    }

    for t in range(T):
        q_t = Q / V
        x_next = solve_subproblem(problem, x, q_t, beta, gamma, eta, n_iters=n_inner_iters)
        Q_next = max(Q + problem.h(x_next), 0.0)
        movement = float(np.linalg.norm(x_next - x))

        # Optimality condition of the subproblem: 0 is in
        #   dP(x_next) + q_t*dh(x_next) + N_X(x_next) + A_t*(x_next - x),
        # so -A_t*(x_next - x) certifies R(x_next, q_t) <= A_t * movement,
        # which is exactly the quantity Theorem 2 bounds.
        A_t = 1.0 / eta + 2 * gamma * q_t
        residual = A_t * movement

        i = t + 1
        history["x"][i] = x_next
        history["Q"][i] = Q_next
        history["q"][i] = Q_next / V
        history["h"][i] = problem.h(x_next)
        history["g"][i] = problem.g(x_next)
        history["P"][i] = problem.P(x_next, beta)
        history["movement"][i] = movement
        history["residual"][i] = residual

        x, Q = x_next, Q_next

    return history
