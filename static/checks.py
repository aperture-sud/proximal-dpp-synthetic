"""Validation checkpoints: a battery of correctness checks run once, with a
clear PASS/FAIL per check, written to results/report.txt.

These are the checks that were actually used, over the course of building
this experiment, to catch real bugs (a degenerate x=0 fixed point, an inner
solver that silently failed to converge near the g=0 boundary). Kept here as
a standing certificate that the current code still passes them.
"""
import os
import numpy as np

from static.problem import build_problem
from static.algorithm import run_dpp, solve_subproblem
from static.theory import predicted_constants, bounds_vs_t
from static.experiments import BETA, GAMMA, ETA, V, T, V_GRID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "report.txt")
lines = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    line = f"[{status}] {name}" + (f"  -- {detail}" if detail else "")
    lines.append(line)
    print(line)


def main():
    p = build_problem()

    # 1. Assumptions 1-7: the construction asserts these at build time, so
    # reaching this point already proves they hold; report the numbers.
    gamma_lo, gamma_hi = p.gamma_range()
    eta_lo, eta_hi = p.eta_range()
    check("Assumptions 1-7 (gamma/eta windows non-empty)", gamma_lo < gamma_hi and eta_lo < eta_hi,
          f"gamma in ({gamma_lo:.4f}, {gamma_hi:.4f}), eta in ({eta_lo:.4f}, {eta_hi:.4f})")

    # 2. Inner solver convergence: the solution should stop changing once
    # n_iters is large enough. Checked on a subproblem near the g=0
    # boundary, where a plain subgradient solver was found not to converge
    # even after 100,000 iterations.
    x_t = np.zeros(p.d)
    x_t[0] = 0.6
    sols = {n: solve_subproblem(p, x_t, q_t=0.05, beta=BETA, gamma=GAMMA, eta=ETA, n_iters=n)
            for n in [50, 200, 800]}
    drift = np.linalg.norm(sols[800] - sols[200])
    check("Inner solver convergence (800 vs 200 iters agree)", drift < 1e-8,
          f"||x_800 - x_200|| = {drift:.2e}")

    # 3. Lemma 2's telescoping step: Q_{t+1} >= Q_t + h(x_{t+1}) implies
    # sum_{s=1}^t h(x_{s+1}) <= Q_{t+1} for every t. This is the algebraic
    # core the rest of Theorem 1 is built on.
    hist = run_dpp(p, beta=BETA, gamma=GAMMA, eta=ETA, V=V, T=300)
    cumsum_h = np.cumsum(hist["h"][1:])
    Q = hist["Q"][1:]
    violations = int(np.sum(cumsum_h > Q + 1e-9))
    check("Telescoping inequality sum(h) <= Q at every t", violations == 0,
          f"{violations} violations out of {len(Q)} steps")

    # 4. Theorem 3's validity threshold: kappa_g > L_f + q_bar*L_h must hold
    # for the beta used in the main sweeps, or Theorem 3's conclusion does
    # not apply.
    c = predicted_constants(p, beta=BETA, gamma=GAMMA, eta=ETA, V=V)
    check("Theorem 3 validity threshold (kappa_g > L_f + q_bar*L_h)", c["delta_g"] > 0,
          f"delta_g = {c['delta_g']:.4f}")

    # 5. Every theorem's bound actually holds empirically, across the V grid
    # used in the main sweep -- the thing all the figures are built to show.
    all_within_bound = True
    for v in V_GRID:
        hist = run_dpp(p, beta=BETA, gamma=GAMMA, eta=ETA, V=v, T=T)
        b = bounds_vs_t(p, BETA, GAMMA, ETA, v, np.array([T]))

        avg_movement_sq = (hist["movement"][1:] ** 2).mean()
        avg_residual_sq = (hist["residual"][1:] ** 2).mean()
        avg_g_plus = np.maximum(hist["g"][1:], 0.0).mean()
        avg_h = hist["h"][1:].mean()

        checks = [
            avg_h <= b["time_avg_h_bound"][0] + 1e-9,
            avg_movement_sq <= b["movement_bound"][0] + 1e-9,
            avg_residual_sq <= b["residual_sq_bound"][0] + 1e-9,
            b["g_violation_bound"] is None or avg_g_plus <= b["g_violation_bound"][0] + 1e-9,
        ]
        all_within_bound = all_within_bound and all(checks)
    check(f"Empirical values stay within predicted bounds (V={V_GRID[0]}..{V_GRID[-1]}, T={T})", all_within_bound)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwritten to {REPORT_PATH}")


if __name__ == "__main__":
    main()
