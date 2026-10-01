"""Synthetic problem instance for the time-varying experiments.

Reuses WeaklyConvexTerm/Problem from static/problem.py unchanged. Both f and h are
made substantially less convex than the static instance (more oscillation
terms, larger amplitudes, a spread of frequencies instead of one), so each
has several competing local wells across the ball rather than one dominant
basin. The point is to make the *trade-off* visible: a step that lowers f
by dropping into one well can leave h positive (or g active) at that same
point, and the queue then has to pull the iterate toward a different well
that is worse for f but better for h -- not just faster convergence to one
basin. g is left exactly convex (a ball), since Assumption 6's closed-form
kappa_g = 2*beta*r_g depends on that; see kappa_g's docstring in static/problem.py.

rho_h is capped by the gamma window (rho_h/2 < gamma < xi/D^2), so h's
amplitude can't be raised as freely as f's without also raising xi or
shrinking the window; the defaults below were chosen by actually computing
gamma_range() for candidate amplitudes (see the __main__ block) rather than
guessed.
"""
import numpy as np

from static.problem import Problem, WeaklyConvexTerm, _unit_rows


def build_problem_tv(seed=0, d=10, R_X=1.0, r_g=0.75, xi=0.4,
                      f_alpha=0.2, n_f=6, f_norm_hi=2.2,
                      h_alpha=0.025, n_h=3, h_norm_hi=1.5, f_pull=0.5):
    """f: n_f cosine terms (default 6, vs. 4 in static/problem.py), amplitude
    f_alpha each (default 0.2, vs. 0.015), frequencies spread linearly from
    1.0 to f_norm_hi instead of all equal -- this raises rho_f to ~3.3
    (vs. 0.086 static) and spreads
    the extra curvature across several different length scales, so the
    oscillation carves out multiple wells rather than one deeper dip.

    h: n_h cosine terms (default 3, vs. 2), amplitude h_alpha each (default
    0.025, vs. 0.02/0.03), frequencies spread from 1.0 to h_norm_hi. This
    raises rho_h to ~0.12 (vs. 0.075 static), the most this construction
    allows while keeping the gamma window (rho_h/2, xi/D^2) = (0.06, 0.1)
    non-empty at xi=0.4.
    """
    rng = np.random.default_rng(seed)

    f_alphas = np.full(n_f, f_alpha)
    f_norms = np.linspace(1.0, f_norm_hi, n_f)
    f_bs = rng.uniform(0.0, 2 * np.pi, size=n_f)
    f_c = _unit_rows(rng, 1, d)[0] * f_pull
    f_term = WeaklyConvexTerm(
        w_quad=np.full(d, 0.2),
        alphas=f_alphas,
        ws=_unit_rows(rng, n_f, d) * f_norms[:, None],
        bs=f_bs,
        c=f_c,
    )

    h_alphas = np.full(n_h, h_alpha)
    h_norms = np.linspace(1.0, h_norm_hi, n_h)
    h_bs = rng.uniform(0.0, 2 * np.pi, size=n_h)
    h_offset = -xi - np.sum(h_alphas * np.cos(h_bs))
    h_term = WeaklyConvexTerm(
        w_quad=np.full(d, 2.0),
        alphas=h_alphas,
        ws=_unit_rows(rng, n_h, d) * h_norms[:, None],
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
    p = build_problem_tv()
    gamma_lo, gamma_hi = p.gamma_range()
    eta_lo, eta_hi = p.eta_range()
    print(f"d={p.d}  R_X={p.R_X}  D={p.D}  xi={p.xi}")
    print(f"rho_f={p.rho_f:.4f}  rho_h={p.rho_h:.4f}  "
          f"(static/problem.py: rho_f=0.0864, rho_h=0.0750)")
    print(f"L_f={p.L_f:.4f}  L_h={p.L_h:.4f}")
    print(f"H={p.H:.4f}  G_g={p.G_g:.4f}  P_min_bound={p.P_min_bound:.4f}")
    print(f"gamma window: ({gamma_lo:.4f}, {gamma_hi:.4f})")
    print(f"eta window:   ({eta_lo:.4f}, {eta_hi:.4f})")
