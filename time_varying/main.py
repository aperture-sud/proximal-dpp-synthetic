"""Run every time-varying sweep and produce every figure.

Usage (from the repository root): python3 -m time_varying.main
"""
from time_varying.problem import build_problem_tv
from time_varying import experiments as exp
from time_varying import plotting as plot


def main():
    problem = build_problem_tv()
    print(f"rho_f={problem.rho_f:.3f}  rho_h={problem.rho_h:.3f}  "
          f"gamma window={problem.gamma_range()}  eta<{1/problem.rho_f:.3f}")

    print("headline trajectory (burst drift)...")
    run = exp.headline_trajectory(problem)
    plot.plot_trajectory_tv(run)

    print("V sweep...")
    v_runs = exp.sweep_V_tv(problem)
    plot.plot_lemma2_queue_tv(v_runs)    # Lemma 2 -- distinct from Theorems 1-3 below
    plot.plot_delta_t_tv(v_runs)
    plot.plot_cumulative_VT_tv(v_runs)

    print("V_T sweep (fixed V=20, T=4000, varying drift noise level)...")
    vt_runs = exp.sweep_VT_tv(problem)

    print("Theorems 1-2, one PNG per graph...")
    plot.plot_theorem1_vs_t(v_runs)
    plot.plot_theorem1_vs_V(v_runs)
    plot.plot_theorem2_movement_vs_t(v_runs)
    plot.plot_theorem2_movement_vs_V(v_runs)
    plot.plot_theorem2_movement_vs_VT(vt_runs)
    plot.plot_theorem2_residual_vs_t(v_runs)
    plot.plot_theorem2_residual_vs_V(v_runs)
    plot.plot_theorem2_residual_vs_VT(vt_runs)

    print("Theorem 3, valid instance (beta=679, delta_g>0: confirms the theorem holds)...")
    plot.plot_theorem3_vs_t(v_runs, filename="theorem3_valid_vs_t.png")
    plot.plot_theorem3_vs_V(v_runs, filename="theorem3_valid_vs_V.png")
    plot.plot_theorem3_vs_VT(vt_runs, filename="theorem3_valid_vs_VT.png")

    print("Theorem 3 'dangerous' demo (separate instance/beta, below Theorem 3's own threshold "
          "so g is actually violated instead of sitting at noise)...")
    danger_problem = exp.build_danger_problem()
    danger_v_runs = exp.sweep_V_tv_danger(danger_problem)
    danger_vt_runs = exp.sweep_VT_tv_danger(danger_problem)
    plot.plot_theorem3_vs_t(danger_v_runs)
    plot.plot_theorem3_vs_V(danger_v_runs)
    plot.plot_theorem3_vs_VT(danger_vt_runs)

    print("beta sweep...")
    beta_star_val = exp.beta_star(problem)
    beta_runs = exp.sweep_beta_tv(problem)
    plot.plot_beta_violation_tv(beta_runs, beta_star_val)

    print("assumption violations...")
    viol_runs = exp.assumption_violations(problem)
    plot.plot_assumption_violations_tv(viol_runs, problem)

    print("done. figures written to figures/time_varying/")


if __name__ == "__main__":
    main()
