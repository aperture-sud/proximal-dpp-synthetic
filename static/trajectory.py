"""Trajectory checkpoints: snapshot f(x_t), Q_t, P(x_t), and the running
averages of h, movement^2, and [g]_+ every 20 steps, at the default operating
point. Also plots the full (dense) trajectory of f, h, Q, P over the run.

Distinct from checks.py, which is a battery of PASS/FAIL correctness
checks -- this is a log and a picture of concrete numbers along one run.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from static.problem import build_problem
from static.algorithm import project_ball, run_dpp
from static.experiments import BETA, GAMMA, ETA, V, T
from static.plotting import _save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
TRAJECTORY_PATH = os.path.join(RESULTS_DIR, "trajectory.txt")
CHECKPOINT_EVERY = 20


def write_table(hist, f_vals):
    checkpoints = list(range(CHECKPOINT_EVERY, T + 1, CHECKPOINT_EVERY))
    header = (f"{'t':>5} {'Q_t':>10} {'q_t':>10} {'f(x_t)':>12} {'P(x_t)':>12} "
               f"{'avg h':>10} {'avg move^2':>12} {'avg g_+':>10}")
    lines = [header, "-" * len(header)]

    for t in checkpoints:
        avg_h = hist["h"][1:t + 1].mean()
        avg_movement_sq = (hist["movement"][1:t + 1] ** 2).mean()
        avg_g_plus = np.maximum(hist["g"][1:t + 1], 0.0).mean()
        lines.append(
            f"{t:>5} {hist['Q'][t]:>10.4f} {hist['q'][t]:>10.4f} {f_vals[t]:>12.4f} {hist['P'][t]:>12.4f} "
            f"{avg_h:>10.4f} {avg_movement_sq:>12.2e} {avg_g_plus:>10.2e}"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TRAJECTORY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {TRAJECTORY_PATH}")


def plot_trajectory(hist, f_vals, filename="trajectory_overview.png"):
    t = np.arange(1, T + 1)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_left.plot(t, f_vals[1:], label="f(x_t)", color="tab:blue", linewidth=2)
    ax_left.plot(t, hist["P"][1:], label="P(x_t)", color="tab:orange", linestyle="--", linewidth=2)
    ax_left.plot(t, hist["h"][1:], label="h(x_t)", color="tab:green", linewidth=1.2, alpha=0.8)
    ax_left.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax_left.set_xlabel("t")
    ax_left.set_title("f, P (overlapping) and h")
    ax_left.legend()
    ax_left.grid(True, alpha=0.3)

    ax_right.plot(t, hist["Q"][1:], color="tab:red", linewidth=2)
    ax_right.set_xlabel("t")
    ax_right.set_ylabel("Q_t")
    ax_right.set_title("Q_t")
    ax_right.grid(True, alpha=0.3)

    fig.suptitle("One run: f, h, Q, P vs t (default operating point)")
    fig.tight_layout()
    _save(fig, filename)


def unconstrained_f_min(p, n_iters=5000):
    """Plain projected gradient descent minimizing f alone over the ball,
    ignoring h and g entirely -- the "price of feasibility" reference point.
    """
    x = np.zeros(p.d)
    step = 1.0 / p.smooth_f
    for _ in range(n_iters):
        x = project_ball(x - step * p.grad_f(x), p.R_X)
    return p.f(x), p.g(x), p.h(x)


def plot_price_of_feasibility(hist, f_vals, f_min, filename="price_of_feasibility.png"):
    t = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(t, f_vals[1:], color="tab:blue", linewidth=2, label="f(x_t)  (DPP algorithm)")
    ax.axhline(f_min, color="tab:red", linestyle="--", linewidth=2,
               label="min f over X alone (infeasible)")
    arrow_t = 0.92 * T
    ax.annotate("", xy=(arrow_t, f_min), xytext=(arrow_t, f_vals[T]),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(arrow_t + 0.015 * T, (f_min + f_vals[T]) / 2, "price of\nfeasibility", va="center", fontsize=10)
    ax.set_xlabel("t")
    ax.set_ylabel("f(x)")
    ax.set_title("The price of feasibility")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def main():
    p = build_problem()
    hist = run_dpp(p, beta=BETA, gamma=GAMMA, eta=ETA, V=V, T=T)
    f_vals = np.array([p.f(x) for x in hist["x"]])

    write_table(hist, f_vals)
    plot_trajectory(hist, f_vals)

    f_min, g_at_min, h_at_min = unconstrained_f_min(p)
    print(f"\nunconstrained min f over X alone: {f_min:.4f}  "
          f"(g={g_at_min:.4f}, h={h_at_min:.4f} there -- infeasible)")
    print(f"DPP algorithm settles at f={f_vals[T]:.4f}  "
          f"(price of feasibility: {f_vals[T] - f_min:.4f})")
    plot_price_of_feasibility(hist, f_vals, f_min)


if __name__ == "__main__":
    main()
