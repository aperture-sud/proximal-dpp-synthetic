"""Synthetic problem instance for validating the proximal DPP theorems.

f and h share a "quadratic + bounded cosine perturbation" template, which
gives closed-form weak-convexity moduli and Lipschitz constants directly
from the constructed coefficients -- no numerical estimation is needed.
g is a ball constraint (stay within radius r_g of the origin), chosen so
Assumption 6 (violating-point regularity) holds analytically: its gradient
2x and the ball's normal cone never cancel, since both point outward.

All derived constants (D, H, xi, rho_f, rho_g, rho_h, L_f, L_g, L_h,
kappa_g, P_min/P_max bounds) are conservative analytic bounds computed
directly from the coefficients below, not estimated by optimization. This
keeps every theoretical overlay curve a valid (if sometimes loose) bound.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class WeaklyConvexTerm:
    """phi(x) = 0.5 * x^T diag(w_quad) x - c.x + sum_j alpha_j cos(w_j . x + b_j) + offset.

    The linear term -c.x does not change the curvature (it has zero
    Hessian), only where the function's minimum sits; it is used to give f
    a reason to move away from the origin, toward the region where the
    h-constraint is active.
    """

    w_quad: np.ndarray   # (d,), diagonal of the PSD quadratic part
    alphas: np.ndarray   # (m,)
    ws: np.ndarray        # (m, d)
    bs: np.ndarray        # (m,)
    c: np.ndarray          # (d,), linear pull term
    offset: float = 0.0

    def value(self, x):
        quad = 0.5 * np.sum(self.w_quad * x ** 2) - np.dot(self.c, x)
        osc = np.sum(self.alphas * np.cos(self.ws @ x + self.bs))
        return float(quad + osc + self.offset)

    def grad(self, x):
        quad_g = self.w_quad * x - self.c
        osc_g = -(self.ws.T @ (self.alphas * np.sin(self.ws @ x + self.bs)))
        return quad_g + osc_g

    @property
    def rho(self):
        """Weak-convexity modulus: an upper bound on -min eigenvalue of the Hessian."""
        return float(np.sum(np.abs(self.alphas) * np.sum(self.ws ** 2, axis=1)))

    @property
    def smoothness(self):
        """Gradient-Lipschitz bound (upper bound on the Hessian's spectral norm)."""
        return float(np.max(self.w_quad) + self.rho)

    def lipschitz_bound(self, R_X):
        """Bound on ||grad phi(x)|| for ||x|| <= R_X."""
        return float(np.max(self.w_quad) * R_X + np.linalg.norm(self.c)
                      + np.sum(np.abs(self.alphas) * np.linalg.norm(self.ws, axis=1)))

    def value_bounds(self, R_X):
        """Conservative analytic [min, max] of value(x) over the ball ||x|| <= R_X."""
        lo = -np.linalg.norm(self.c) * R_X - np.sum(np.abs(self.alphas)) + self.offset
        hi = (0.5 * np.max(self.w_quad) * R_X ** 2 + np.linalg.norm(self.c) * R_X
              + np.sum(np.abs(self.alphas)) + self.offset)
        return float(lo), float(hi)


@dataclass
class Problem:
    d: int
    R_X: float
    f_term: WeaklyConvexTerm
    h_term: WeaklyConvexTerm
    r_g: float
    xi: float             # Slater margin: h(x_slater) <= -xi
    x_slater: np.ndarray

    def __post_init__(self):
        self.D = 2 * self.R_X
        self.rho_f = self.f_term.rho
        self.rho_h = self.h_term.rho
        self.rho_g = 0.0
        self.L_f = self.f_term.lipschitz_bound(self.R_X)
        self.L_h = self.h_term.lipschitz_bound(self.R_X)
        self.L_g = 2 * self.R_X
        self.smooth_f = self.f_term.smoothness
        self.smooth_h = self.h_term.smoothness
        self.H = (0.5 * np.max(self.h_term.w_quad) * self.R_X ** 2
                  + np.sum(np.abs(self.h_term.alphas)) + abs(self.h_term.offset))
        self.G_g = self.R_X ** 2 - self.r_g ** 2
        f_lo, f_hi = self.f_term.value_bounds(self.R_X)
        self.P_min_bound = f_lo   # P = f + beta*[g]_+ >= f  for any beta >= 0
        self._f_hi = f_hi

    def f(self, x):
        return self.f_term.value(x)

    def grad_f(self, x):
        return self.f_term.grad(x)

    def h(self, x):
        return self.h_term.value(x)

    def grad_h(self, x):
        return self.h_term.grad(x)

    def g(self, x):
        return float(np.dot(x, x) - self.r_g ** 2)

    def grad_g(self, x):
        return 2 * x

    def P(self, x, beta):
        return self.f(x) + beta * max(self.g(x), 0.0)

    def P_max_bound(self, beta):
        return self._f_hi + beta * self.G_g

    def kappa_g(self, beta):
        """Assumption 6 constant: dist(0, beta*grad_g(x) + N_X(x)) >= kappa_g whenever g(x) > 0."""
        return 2 * beta * self.r_g

    def gamma_range(self):
        """Assumption 7 window: rho_h/2 < gamma < xi/D^2."""
        return self.rho_h / 2.0, self.xi / self.D ** 2

    def eta_range(self):
        """Assumption 7 window: 0 < eta < 1/rho_P, rho_P = rho_f + beta*rho_g = rho_f here."""
        return 0.0, (np.inf if self.rho_f == 0 else 1.0 / self.rho_f)


def _unit_rows(rng, m, d):
    v = rng.normal(size=(m, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_problem(seed=0, d=10, R_X=1.0, r_g=0.75, xi=0.4):
    """Construct the synthetic (f, g, h, X) instance and check Assumptions 1-7 hold.

    f is given a linear pull term -c.x so its unconstrained optimum sits
    outside the ball, near ||x|| = R_X; h's quadratic part is scaled so
    that h(x) > 0 near the boundary of X. Without this, minimizing f never
    has a reason to approach the h-constraint boundary, and the queue in
    Algorithm 1 would sit at 0 for the whole run.
    """
    rng = np.random.default_rng(seed)

    # f: mildly weakly convex -- the oscillation is a small wrinkle on top of
    # the convex quadratic (rho_f well below ||w_quad||), not a genuinely
    # multi-modal landscape. Random (nonzero) phases bs keep x=0 from being a
    # trivial critical point.
    f_alphas = np.full(4, 0.015)
    f_norms = np.full(4, 1.2)
    f_bs = rng.uniform(0.0, 2 * np.pi, size=4)
    f_c = _unit_rows(rng, 1, d)[0] * 0.5
    f_term = WeaklyConvexTerm(
        w_quad=np.full(d, 0.2),
        alphas=f_alphas,
        ws=_unit_rows(rng, 4, d) * f_norms[:, None],
        bs=f_bs,
        c=f_c,
    )

    # h: quadratic part scaled up (isotropic, so it grows with ||x|| in every
    # direction) so it swings from negative near the origin to positive near
    # the boundary. rho_h is set only by the small oscillation terms, so the
    # gamma window (rho_h/2, xi/D^2) stays wide even though h's range is wide.
    h_alphas = np.array([0.02, 0.03])
    h_norms = np.array([1.5, 1.0])
    h_bs = rng.uniform(0.0, 2 * np.pi, size=2)
    h_offset = -xi - np.sum(h_alphas * np.cos(h_bs))  # forces h(x_slater) == -xi, x_slater = 0
    h_term = WeaklyConvexTerm(
        w_quad=np.full(d, 2.0),
        alphas=h_alphas,
        ws=_unit_rows(rng, 2, d) * h_norms[:, None],
        bs=h_bs,
        c=np.zeros(d),
        offset=h_offset,
    )

    problem = Problem(d=d, R_X=R_X, f_term=f_term, h_term=h_term,
                       r_g=r_g, xi=xi, x_slater=np.zeros(d))

    gamma_lo, gamma_hi = problem.gamma_range()
    assert r_g < R_X, "the budget region must sit strictly inside X"
    assert gamma_lo < gamma_hi, (
        f"empty gamma window: rho_h/2={gamma_lo:.4f} >= xi/D^2={gamma_hi:.4f}")
    assert abs(problem.h(problem.x_slater) + xi) < 1e-9, "Slater-point calibration failed"

    return problem


if __name__ == "__main__":
    p = build_problem()
    gamma_lo, gamma_hi = p.gamma_range()
    eta_lo, eta_hi = p.eta_range()
    print(f"d={p.d}  R_X={p.R_X}  D={p.D}  xi={p.xi}")
    print(f"rho_f={p.rho_f:.4f}  rho_g={p.rho_g}  rho_h={p.rho_h:.4f}")
    print(f"L_f={p.L_f:.4f}  L_g={p.L_g:.4f}  L_h={p.L_h:.4f}")
    print(f"H={p.H:.4f}  G_g={p.G_g:.4f}  P_min_bound={p.P_min_bound:.4f}")
    print(f"gamma window: ({gamma_lo:.4f}, {gamma_hi:.4f})")
    print(f"eta window:   ({eta_lo:.4f}, {eta_hi:.4f})")
