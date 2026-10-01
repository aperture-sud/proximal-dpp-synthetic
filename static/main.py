"""Run every sweep and produce every figure.

Usage (from the repository root): python3 -m static.main
"""
from static.problem import build_problem
from static import experiments, plotting, trajectory


def main():
    problem = build_problem()

    print("running V sweep...")
    v_runs = experiments.sweep_V(problem)
    plotting.plot_queue(v_runs)
    plotting.plot_time_average_h(v_runs)
    plotting.plot_movement(v_runs)
    plotting.plot_residual(v_runs)
    plotting.plot_constraint_violation(v_runs)

    print("running beta ablation...")
    beta_runs = experiments.sweep_beta(problem)
    plotting.plot_ablation(beta_runs, "beta", "ablation_beta.png")

    print("running gamma ablation...")
    gamma_runs = experiments.sweep_gamma(problem)
    plotting.plot_ablation(gamma_runs, "gamma", "ablation_gamma.png")

    print("running eta ablation...")
    eta_runs = experiments.sweep_eta(problem)
    plotting.plot_ablation(eta_runs, "eta", "ablation_eta.png")

    print("running joint epsilon sweep (Corollary 1)...")
    eps_runs = experiments.sweep_epsilon(problem)
    plotting.plot_corollary(eps_runs)

    print("running default-point trajectory...")
    trajectory.main()

    print("done. figures written to figures/static/")


if __name__ == "__main__":
    main()
