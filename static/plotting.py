"""One figure per theorem, built from experiments.py sweep outputs.

V, beta, gamma, eta are all ordered quantities, so curves indexed by them use
a light-to-dark viridis ramp (sequential coloring for a magnitude variable,
colorblind-safe by construction) rather than an arbitrary categorical cycle.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures", "static")


def running_average(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)


NOISE_FLOOR = 1e-12  # values at or below this are floating-point roundoff, not signal


def fit_loglog_slope(x, y):
    """Least-squares slope of log(y) vs log(x), i.e. y ~ x^slope.

    Returns None if fewer than three points exceed NOISE_FLOOR (e.g. a
    constraint that is satisfied almost exactly everywhere, leaving nothing
    but floating-point roundoff -- ~1e-16 to 1e-17 in our runs -- with too
    few genuine violations left to fit a decay rate through). Two points
    would always give a "perfect" fit with no statistical content, and
    fitting a slope through pure roundoff noise is not a measurement.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = y > NOISE_FLOOR
    if mask.sum() < 3:
        return None
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def sequential_colors(n):
    return plt.cm.viridis(np.linspace(0.15, 0.85, n))


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_queue(runs, filename="lemma2_queue.png"):
    """Lemma 2: Q_t stays under a V-dependent ceiling, so q_t = Q_t/V stays
    under the same ceiling q_bar regardless of V."""
    colors = sequential_colors(len(runs))
    fig, (ax_Q, ax_q) = plt.subplots(2, 1, figsize=(6.5, 8.5))

    T_max = len(runs[0]["history"]["Q"]) - 1
    for run, color in zip(runs, colors):
        t = np.arange(1, len(run["history"]["Q"]))
        ax_Q.plot(t, run["history"]["Q"][1:], color=color, label=f"V={run['V']}")
        ax_Q.axhline(run["bounds"]["queue_bound"], color=color, linestyle="--", linewidth=1, alpha=0.5)
        ax_q.plot(t, run["history"]["q"][1:], color=color, label=f"V={run['V']}")

    q_bar = runs[0]["bounds"]["constants"]["q_bar"]  # same predicted ceiling for every V
    ax_q.axhline(q_bar, color="black", linestyle="--", linewidth=1.3, label="predicted ceiling q_bar")

    for ax, ylabel, title in [(ax_Q, "Q_t", "queue (dashed = predicted ceiling, per V)"),
                               (ax_q, "q_t = Q_t / V", "q_t stays under the same ceiling regardless of V")]:
        # Log scale: the worst-case ceiling sits orders of magnitude above what
        # is actually achieved (typical of these bounds), so a linear axis
        # would flatten every empirical curve to the bottom of the plot.
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, T_max)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Lemma 2: pointwise queue bound")
    fig.tight_layout()
    _save(fig, filename)


def _plot_running_average_vs_t_and_V(runs, raw_fn, bound_key, ylabel, title, filename, log_y,
                                      linear_t_axes=False):
    """Shared body for the Theorem 1/2/3 figures: running average vs t (all
    V, dashed predicted bound), and final value vs V zoomed to the empirical
    curve's own scale, with the predicted bound overlaid on a secondary axis
    -- the bound is typically orders of magnitude larger, which would
    otherwise flatten any fine structure in the empirical values.

    linear_t_axes forces the "vs t" panel onto plain linear x and y axes
    instead of log/symlog -- off by default since the bound and empirical
    values usually span several orders of magnitude.
    """
    colors = sequential_colors(len(runs))
    fig, (ax_t, ax_V) = plt.subplots(2, 1, figsize=(6.5, 8.5))

    final_values = []
    final_bounds = []
    for run, color in zip(runs, colors):
        raw = raw_fn(run["history"])
        avg = running_average(raw)
        t = np.arange(1, len(avg) + 1)
        bound = run["bounds"][bound_key]

        ax_t.plot(t, avg, color=color, label=f"V={run['V']}")
        if bound is not None and not linear_t_axes:
            # Skipped on a plain linear axis: the bound is typically orders
            # of magnitude larger than the empirical curve, so on a linear
            # scale it would dominate the plot exactly the way the log/symlog
            # axes exist elsewhere to avoid.
            ax_t.plot(t, bound, color=color, linestyle="--", linewidth=1, alpha=0.6)
        final_values.append(avg[-1])
        final_bounds.append(bound[-1] if bound is not None else None)

    T_max = len(runs[0]["history"]["h"]) - 1
    ax_t.set_xlabel("t")
    ax_t.set_ylabel(ylabel)
    ax_t.set_title("running average vs t" if linear_t_axes
                    else "running average vs t (dashed = predicted bound)")
    if linear_t_axes:
        # Plain linear x and y; the predicted bound is not
        # drawn here (it is orders of magnitude larger than the empirical
        # curve and would dominate a linear axis).
        ax_t.set_xlim(0, T_max)
    else:
        ax_t.set_xscale("log")
        # The default log-scale locator only labels powers of 10, which can
        # look like the data stops at t=100 even though it runs to T_max;
        # label the actual endpoint explicitly.
        decade_ticks = [10 ** k for k in range(0, int(np.floor(np.log10(T_max))) + 1)]
        all_ticks = decade_ticks + [T_max]
        ax_t.set_xticks(all_ticks)
        ax_t.set_xticklabels([str(v) for v in all_ticks])
        ax_t.set_xlim(1, T_max)
        if log_y:
            ax_t.set_yscale("log")
        else:
            # symlog rather than linear: the worst-case bound is orders of
            # magnitude larger than the empirical values, and h can be
            # negative, so a plain linear axis would force a choice between
            # showing the bound and showing the empirical curve. symlog shows
            # both on one axis (linear near 0, log further out).
            ax_t.set_yscale("symlog", linthresh=0.1)
    ax_t.legend(fontsize=8)
    ax_t.grid(True, which="both", alpha=0.2)

    V_values = [run["V"] for run in runs]
    ax_V.plot(V_values, final_values, "o-", color="tab:blue", label="empirical")
    ax_V.set_xlabel("V")
    ax_V.set_ylabel(ylabel + f" (at t=T, T={T_max} fixed)")
    slope = fit_loglog_slope(V_values, final_values) if log_y else None

    # Zoom the primary axis to the empirical curve's own range -- the bound
    # is typically orders of magnitude larger, which would otherwise flatten
    # any fine structure in the empirical values (e.g. the small decrease
    # with V that a shared log axis hides).
    lo, hi = min(final_values), max(final_values)
    pad = 0.1 * (hi - lo) if hi > lo else max(abs(lo), 1e-12) * 0.1
    band = hi - lo if hi > lo else max(abs(hi), 1e-12)

    if log_y and all(b is not None for b in final_bounds):
        # The predicted bound is orders of magnitude larger than the (now
        # zoomed-in) empirical curve, so it is not drawn to scale: it is
        # rescaled to its own shape and placed in a band just above the
        # empirical data, purely to show whether it rises, falls, or flattens
        # alongside the empirical curve -- not its absolute magnitude.
        bound_arr = np.array(final_bounds, dtype=float)
        bspan = bound_arr.max() - bound_arr.min()
        bound_norm = (bound_arr - bound_arr.min()) / bspan if bspan > 0 else np.zeros_like(bound_arr)
        band_lo, band_hi = hi + pad, hi + pad + band
        ax_V.plot(V_values, band_lo + bound_norm * (band_hi - band_lo),
                  "--", color="gray", label="predicted bound (shape only)")
        ax_V.set_ylim(lo - pad, band_hi + pad)
    else:
        ax_V.set_ylim(lo - pad, hi + pad)

    if slope is not None:
        ax_V.set_title(f"final value vs V, zoomed (fitted slope = {slope:.2f})")
    elif log_y:
        ax_V.set_title("final value vs V, zoomed (too few nonzero points for a slope fit)")
    else:
        # Signed metric (time-average h): Theorem 1's bound is affine in V
        # at fixed T, and orders of magnitude larger than the empirical
        # values, so it is left off rather than forcing a shared axis that
        # would flatten the (already tightly zoomed) empirical curve.
        ax_V.set_title("final value vs V")
    ax_V.legend(fontsize=8)
    ax_V.grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, filename)


def plot_time_average_h(runs, filename="theorem1a_time_average_h.png"):
    """Theorem 1, eq. (8): (1/T) sum h(x_{t+1}) -> 0 at rate O(V/T + 1/T)."""
    _plot_running_average_vs_t_and_V(
        runs, raw_fn=lambda h: h["h"][1:], bound_key="time_avg_h_bound",
        ylabel="running avg h(x)", title="Theorem 1: time-average constraint",
        filename=filename, log_y=False)


def plot_movement(runs, filename="theorem1b_movement.png"):
    """Theorem 1, eq. (9): running average of ||x_{t+1}-x_t||^2."""
    _plot_running_average_vs_t_and_V(
        runs, raw_fn=lambda h: h["movement"][1:] ** 2, bound_key="movement_bound",
        ylabel="running avg ||x_{t+1}-x_t||^2", title="Theorem 1: movement bound",
        filename=filename, log_y=True, linear_t_axes=True)


def plot_residual(runs, filename="theorem2_residual.png"):
    """Theorem 2, eq. (12): E[R(x_tau+1, q_tau)^2], estimated as the running
    average of R(x_{t+1}, q_t)^2 over a uniformly sampled t."""
    _plot_running_average_vs_t_and_V(
        runs, raw_fn=lambda h: h["residual"][1:] ** 2, bound_key="residual_sq_bound",
        ylabel="running avg R(x_{t+1}, q_t)^2", title="Theorem 2: penalized near-stationarity",
        filename=filename, log_y=True)


def plot_constraint_violation(runs, filename="theorem3_violation.png"):
    """Theorem 3, eq. (13): E[g(x_tau+1)]_+, estimated the same way."""
    _plot_running_average_vs_t_and_V(
        runs, raw_fn=lambda h: np.maximum(h["g"][1:], 0.0), bound_key="g_violation_bound",
        ylabel="running avg [g(x)]_+", title="Theorem 3: ordinary-constraint feasibility",
        filename=filename, log_y=True)


def plot_corollary(eps_runs, filename="corollary1_joint_rate.png"):
    """Corollary 1: V=ceil(eps^-2), T=ceil(eps^-3) gives E[R]=O(eps),
    E[g]_+=O(eps^2), time-average h = O(eps)."""
    eps = np.array([run["eps"] for run in eps_runs])
    metrics = [
        (np.array([run["history"]["residual"][1:].mean() for run in eps_runs]),
         "E[R(x_tau+1, q_tau)]"),
        (np.array([np.maximum(run["history"]["g"][1:], 0.0).mean() for run in eps_runs]),
         "E[g(x_tau+1)]_+"),
        (np.array([abs(run["history"]["h"][1:].mean()) for run in eps_runs]),
         "|time-average h|"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (y, ylabel) in zip(axes, metrics):
        ax.plot(eps, y, "o-", color="tab:blue", markersize=7, linewidth=2)
        slope = fit_loglog_slope(eps, y)
        ax.set_xlabel("epsilon")
        ax.set_ylabel(ylabel)
        if slope is not None:
            ax.set_yscale("log")
            ax.set_title(f"fitted slope={slope:.2f}")
        elif np.all(y <= NOISE_FLOOR):
            ax.set_title("empirical value is at floating-point noise (<1e-12), not a real violation")
        else:
            ax.set_title("too few nonzero points for a slope fit")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Corollary 1: V = ceil(eps^-2), T = ceil(eps^-3)")
    fig.tight_layout()
    _save(fig, filename)


def plot_ablation(runs, param_name, filename):
    """Final metrics vs one swept parameter (beta, gamma, or eta), others fixed."""
    x = [run[param_name] for run in runs]
    metrics = [
        ([run["history"]["h"][1:].mean() for run in runs], "avg h(x)"),
        ([(run["history"]["movement"][1:] ** 2).mean() for run in runs], "avg ||x_{t+1}-x_t||^2"),
        ([(run["history"]["residual"][1:] ** 2).mean() for run in runs], "avg R(x_{t+1}, q_t)^2"),
        ([np.maximum(run["history"]["g"][1:], 0.0).mean() for run in runs], "avg [g(x)]_+"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (y, ylabel) in zip(axes.flat, metrics):
        ax.plot(x, y, "o-", color="tab:blue")
        ax.set_xlabel(param_name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Ablation over {param_name} (other parameters fixed at defaults)")
    fig.tight_layout()
    _save(fig, filename)
