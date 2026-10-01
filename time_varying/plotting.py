"""Figures for the time-varying experiments. All axes are plain linear (no
log/symlog/log-log anywhere in this file, unlike static/plotting.py). Since the
predicted bounds here are typically 10-1000x the empirical values (same
conservatism as the static case), a shared linear axis would flatten the
empirical curve to a line at the bottom.

The fix used throughout: the empirical curve is always drawn true to scale,
zoomed to its own range. Where a bound is included, it is rescaled into a
thin band placed just above the data and labeled "(shape only, rescaled)"
-- this shows whether the bound rises, falls, or flattens alongside the
data, without ever changing the y-axis scale. This is the same trick
static/plotting.py already uses for its "final value vs V" panels; here it is used
everywhere, so every axis in this file stays linear.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from time_varying.drift import functional_variation
from time_varying.experiments import PSI_MAX_VALID, PSI_MAX_SLATER_VIOLATE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures", "time_varying")


def running_average(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def sequential_colors(n):
    return plt.cm.viridis(np.linspace(0.15, 0.85, n))


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _data_range(curves):
    lo = min(np.min(y) for y in curves)
    hi = max(np.max(y) for y in curves)
    pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1e-12) * 0.15 + 1e-12
    return lo, hi, pad


def _band_overlay(ax, x, bound, hi, pad, label="predicted bound (shape only, rescaled)"):
    """Rescale `bound` (same length as x) into a thin band just above the
    data range [*, hi+pad] and plot it dashed. Returns the new plot top.
    """
    b = np.asarray(bound, dtype=float)
    bspan = b.max() - b.min()
    bnorm = (b - b.min()) / bspan if bspan > 0 else np.zeros_like(b)
    band_lo, band_hi = hi + pad, hi + pad + max(hi - (hi - 2 * pad), 1e-12)
    ax.plot(x, band_lo + bnorm * (band_hi - band_lo), "--", color="gray", linewidth=1.3,
            alpha=0.75, label=label)
    return band_hi


# ----------------------------------------------------------------------
# Headline trajectory
# ----------------------------------------------------------------------

def plot_trajectory_tv(run, filename="trajectory_tv.png"):
    """f(x_t), psi_t & h_t(x_t), Q_t vs t on one run -- the mechanism: f
    dips deepest while psi_t is near 0, a psi burst pushes h_t up, Q_t
    spikes in response, and the next rounds are pulled back toward
    feasibility, raising f again.
    """
    hist, f_vals, psi = run["history"], run["f"], run["psi"]
    T = len(f_vals) - 1
    t = np.arange(1, T + 1)

    fig, (ax_f, ax_psi, ax_Q) = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)

    ax_f.plot(t, f_vals[1:], color="tab:blue", linewidth=1.3)
    ax_f.set_ylabel("f(x_t)")
    ax_f.set_title(r"$f(x_t)$" + "   (the objective actually achieved, each round)")
    ax_f.grid(True, alpha=0.3)

    ax_psi.plot(t, psi[:T], color="tab:purple", linewidth=1.3, label=r"$\psi_t$  (the drift)")
    ax_psi.plot(t, hist["h_t"][1:], color="tab:green", linewidth=1.0, alpha=0.75,
                label=r"$h_t(x_t)$  (constraint value actually achieved)")
    ax_psi.axhline(0, color="black", linewidth=0.7, alpha=0.4)
    ax_psi.set_ylabel(r"$\psi_t$   /   $h_t(x_t)$")
    ax_psi.set_title(r"$h_t(x_t) = h(x_t) + \psi_t$")
    ax_psi.legend(fontsize=9)
    ax_psi.grid(True, alpha=0.3)

    ax_Q.plot(t, hist["Q"][1:], color="tab:red", linewidth=1.3)
    ax_Q.set_ylabel(r"$Q_t$")
    ax_Q.set_xlabel("t")
    ax_Q.set_title(r"$Q_{t+1} = [\,Q_t + h_t(x_{t+1})\,]_+$" + "   (the virtual queue)")
    ax_Q.grid(True, alpha=0.3)
    ax_Q.set_xlim(0, T)

    fig.suptitle("One trajectory, burst drift: f dips while the queue is empty, "
                 "then pays it back once a burst raises h_t")
    fig.tight_layout()
    _save(fig, filename)


# ----------------------------------------------------------------------
# Lemma 2: pointwise queue bound
# ----------------------------------------------------------------------

def plot_lemma2_queue_tv(runs, filename="lemma2_queue_tv.png"):
    """Lemma 2's pointwise queue bound, Q_t <= 2*V*C_circ/xi_bar + H, plotted
    in raw Q_t (not q_t = Q_t/V): q_t's own ceiling is nearly flat across V
    (only the H/V term moves), which hides the V-dependence: raw Q_t and its
    bound are both linear in V, so this is the version that actually shows
    it (empirical Q_T: 0.04 at V=0.1 up to 64 at V=100, essentially
    proportional to V, same as the bound).

    Lemma 1 (every subproblem mu_t-strongly convex), which Lemma 3 restates
    alongside this bound, has no run-to-run variation to plot: it's
    enforced by an assert in solve_subproblem, not observed.
    """
    colors = sequential_colors(len(runs))
    fig, (ax_t, ax_V) = plt.subplots(2, 1, figsize=(7, 8.5))

    T_max = len(runs[0]["history"]["Q"]) - 1
    t = np.arange(1, T_max + 1)
    Q_curves = [run["history"]["Q"][1:] for run in runs]
    lo, hi, pad = _data_range(Q_curves)

    # Zoomed to the empirical curves' own range -- the per-V ceiling is
    # 100-1000x higher here and would flatten everything to a line at 0.
    for run, color, Q in zip(runs, colors, Q_curves):
        ax_t.plot(t, Q, color=color, label=f"V={run['V']}")
    ax_t.set_ylim(lo - pad, hi + pad)
    ax_t.set_xlim(0, T_max)
    ax_t.set_xlabel("t")
    ax_t.set_ylabel("Q_t")
    ax_t.set_title(r"$Q_t \leq \dfrac{2VC_\circ}{\bar{\xi}} + H$" + "  (zoomed to the data; ceiling is 100-1000x higher)")
    ax_t.legend(fontsize=8, ncol=2)
    ax_t.grid(True, alpha=0.3)

    V_vals = np.array([r["V"] for r in runs], dtype=float)
    final_Q = [run["history"]["Q"][-1] for run in runs]
    Q_bound = [run["bounds"]["queue_bound"] for run in runs]
    lo2, hi2, pad2 = _data_range([np.array(final_Q)])
    ax_V.plot(V_vals, final_Q, "o-", color="tab:blue", label="empirical Q_T")
    _band_overlay(ax_V, V_vals, Q_bound, hi2, pad2, label="predicted ceiling (shape only) -- both linear in V")
    ax_V.set_ylim(lo2 - pad2, hi2 + 2.2 * pad2 + max(hi2 - (hi2 - 2 * pad2), 1e-12))
    ax_V.set_xlim(0, V_vals.max())
    ax_V.set_xlabel("V")
    ax_V.set_ylabel("Q_T")
    ax_V.set_title(r"$Q_T \leq \dfrac{2VC_\circ}{\bar{\xi}} + H$" + "  (both Q_T and its bound grow linearly in V)")
    ax_V.legend(fontsize=8)
    ax_V.grid(True, alpha=0.3)

    fig.suptitle("Lemma 2: pointwise queue bound, time-varying h_t")
    fig.tight_layout()
    _save(fig, filename)


DELTA_T_TEX = r"$\Delta_t := \sup_{x\in\mathcal{X}} |h_{t+1}(x) - h_t(x)|$"
V_T_TEX = r"$V_T := \sum_{t=1}^{T-1} \Delta_t$"


def plot_delta_t_tv(runs, filename="delta_t_variation.png"):
    """Delta_t, the one-step functional variation, for the psi_t realization
    used in the V-sweep (same psi_t for every V, so one curve here). In this
    construction h_t(x) = h(x) + psi_t, so the h(x) part cancels exactly and
    Delta_t = |psi_{t+1} - psi_t| for every x -- the sup is trivial.
    """
    psi = runs[0]["psi"]
    delta, _ = functional_variation(psi)
    t = np.arange(1, len(delta) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(t, delta, color="tab:purple", linewidth=0.9)
    ax.set_xlabel("t")
    ax.set_ylabel("Delta_t")
    ax.set_title(DELTA_T_TEX)
    ax.set_xlim(0, len(delta))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def plot_cumulative_VT_tv(runs, filename="cumulative_VT_tv.png"):
    """V_T(t), the running cumulative sum of Delta_t up to round t -- this is
    the quantity that appears directly in Theorem 2/3's extra q_bar*V_T/(c0*t)
    term (same psi_t sequence used in every V run, so one curve here).
    """
    psi = runs[0]["psi"]
    _, VT = functional_variation(psi)
    T = len(VT)
    t = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(t, VT, color="tab:purple", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("V_T(t)")
    ax.set_title(V_T_TEX)
    ax.set_xlim(0, T)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


# ----------------------------------------------------------------------
# Theorems 1-3, one quantity each, one PNG per graph (not combined panels):
#   Theorem 1 (no V_T term in its bound): vs t, vs V            -- 2 PNGs
#   Theorem 2 (residual R_tau^2, has a V_T term):  vs t, vs V, vs V_T  -- 3 PNGs
#   Theorem 3 ([g(x)]_+, has a V_T term):          vs t, vs V, vs V_T  -- 3 PNGs
# ----------------------------------------------------------------------

def _single_vs_t(runs, raw_fn, bound_key, ylabel, title, filename, bound_transform=None):
    """Running average of raw_fn(history) vs t, one curve per V (all on one
    axis), with the middle-V run's predicted bound overlaid as a shape-only
    band (see _band_overlay) since the bound is typically 10-1000x the data.
    """
    colors = sequential_colors(len(runs))
    fig, ax = plt.subplots(figsize=(7, 5))

    T_max = len(runs[0]["history"]["h"]) - 1
    t = np.arange(1, T_max + 1)
    avgs = [running_average(raw_fn(run["history"])) for run in runs]
    lo, hi, pad = _data_range(avgs)

    for run, color, avg in zip(runs, colors, avgs):
        ax.plot(t, avg, color=color, label=f"V={run['V']}")

    bound = runs[len(runs) // 2]["bounds"][bound_key]
    if bound is not None and bound_transform is not None:
        bound = bound_transform(np.asarray(bound, dtype=float))
    if bound is not None:
        _band_overlay(ax, t, bound, hi, pad)
        ax.set_ylim(lo - pad, hi + 2.2 * pad + max(hi - (hi - 2 * pad), 1e-12))
    else:
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, T_max)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def _single_vs_V(runs, raw_fn, bound_key, ylabel, title, filename, bound_transform=None):
    """Final (t=T) value of raw_fn(history) vs V, true linear x-axis (the
    actual V values, not evenly-spaced category ticks) -- most of the 10
    points land below V=20 since the grid spans 0.1 to 100, but that
    crowding is the real, linear picture, not an artifact of relabeling.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    V_vals = np.array([r["V"] for r in runs], dtype=float)
    final_vals = [running_average(raw_fn(run["history"]))[-1] for run in runs]
    final_bounds = [run["bounds"][bound_key] for run in runs]
    final_bounds = [b[-1] if b is not None else None for b in final_bounds]
    if bound_transform is not None:
        final_bounds = [bound_transform(b) if b is not None else None for b in final_bounds]

    lo, hi, pad = _data_range([np.array(final_vals)])
    ax.plot(V_vals, final_vals, "o-", color="tab:blue", label="empirical")
    if all(b is not None for b in final_bounds):
        _band_overlay(ax, V_vals, final_bounds, hi, pad, label="predicted bound (shape only)")
        ax.set_ylim(lo - pad, hi + 2.2 * pad + max(hi - (hi - 2 * pad), 1e-12))
    else:
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, V_vals.max())
    ax.set_xlabel("V")
    ax.set_ylabel(ylabel + " (at t=T)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


def _single_vs_VT(vt_runs, raw_fn, bound_key, ylabel, title, filename, bound_transform=None):
    """Final (t=T) value of raw_fn(history) vs the realized cumulative
    functional variation V_T, one point per drift-noise-level run (V and T
    held fixed across the whole sweep; only how jumpy psi_t is changes)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    V_T = np.array([run["V_T"] for run in vt_runs])
    order = np.argsort(V_T)
    V_T = V_T[order]
    final_vals = np.array([running_average(raw_fn(run["history"]))[-1] for run in vt_runs])[order]
    final_bounds = [vt_runs[i]["bounds"][bound_key] for i in order]
    final_bounds = [b[-1] if b is not None else None for b in final_bounds]
    if bound_transform is not None:
        final_bounds = [bound_transform(b) if b is not None else None for b in final_bounds]

    lo, hi, pad = _data_range([final_vals])
    ax.plot(V_T, final_vals, "o-", color="tab:blue", label="empirical")
    if all(b is not None for b in final_bounds):
        _band_overlay(ax, V_T, final_bounds, hi, pad, label="predicted bound (shape only)")
        ax.set_ylim(lo - pad, hi + 2.2 * pad + max(hi - (hi - 2 * pad), 1e-12))
    else:
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("V_T (cumulative functional variation of h_t over the run, T=4000 fixed)")
    ax.set_ylabel(ylabel + " (at t=T)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


# Titles are the literal theorem statements (mathtext), not descriptive
# names -- the same inequality is shown whether the x-axis below it is t,
# V, or V_T, since the inequality itself doesn't change, only which
# variable is being read off it.
THM1_TEX = (r"$\dfrac{1}{t}\sum_{s=1}^{t} h_s(x_{s+1}) \;\leq\; "
            r"\dfrac{2VC_\circ}{\bar{\xi}\,t} + \dfrac{H}{t}$")
THM2_MOVEMENT_TEX = (r"$\dfrac{1}{t}\sum_{s=1}^{t} \|x_{s+1}-x_s\|^2 \;\leq\; "
                      r"\dfrac{C_S}{c_0 t} + \dfrac{H^2}{c_0 V} + \dfrac{\bar{q}\,V_T}{c_0 t} + \dfrac{D^2}{t}$")
THM2_RESIDUAL_TEX = (r"$\mathbb{E}[R_\tau] = O\!\left(V^{-1/2}+T^{-1/2}+\sqrt{V_T/T}\right)$")
THM3_TEX = (r"$\mathbb{E}\,[g(x_{\tau+1})]_+ \;\leq\; \dfrac{G_g}{\delta_g^2}\,\mathbb{E}[R_\tau^2] "
            r"= O\!\left(\dfrac{1}{V}+\dfrac{1}{T}+\dfrac{V_T}{T}\right)$")


def plot_theorem1_vs_t(runs, filename="theorem1_vs_t.png"):
    _single_vs_t(runs, raw_fn=lambda h: h["h_t"][1:], bound_key="time_avg_h_bound",
                 ylabel="(1/t) sum h_t(x_{s+1})",
                 title=THM1_TEX,
                 filename=filename)


def plot_theorem1_vs_V(runs, filename="theorem1_vs_V.png"):
    _single_vs_V(runs, raw_fn=lambda h: h["h_t"][1:], bound_key="time_avg_h_bound",
                 ylabel="(1/T) sum h_t(x_{s+1})",
                 title=THM1_TEX,
                 filename=filename)


def plot_theorem2_movement_vs_t(runs, filename="theorem2_movement_vs_t.png"):
    _single_vs_t(runs, raw_fn=lambda h: h["movement"][1:] ** 2, bound_key="movement_bound",
                 ylabel="(1/t) sum ||x_{s+1}-x_s||^2",
                 title=THM2_MOVEMENT_TEX,
                 filename=filename)


def plot_theorem2_movement_vs_V(runs, filename="theorem2_movement_vs_V.png"):
    _single_vs_V(runs, raw_fn=lambda h: h["movement"][1:] ** 2, bound_key="movement_bound",
                 ylabel="(1/T) sum ||x_{s+1}-x_s||^2",
                 title=THM2_MOVEMENT_TEX,
                 filename=filename)


def plot_theorem2_movement_vs_VT(vt_runs, filename="theorem2_movement_vs_VT.png"):
    _single_vs_VT(vt_runs, raw_fn=lambda h: h["movement"][1:] ** 2, bound_key="movement_bound",
                  ylabel="(1/T) sum ||x_{s+1}-x_s||^2",
                  title=THM2_MOVEMENT_TEX,
                  filename=filename)


def plot_theorem2_residual_vs_t(runs, filename="theorem2_residual_vs_t.png"):
    # E[R_tau] <= sqrt(E[R_tau^2]) <= sqrt(bound) by Jensen (sqrt is concave):
    # raw R_t averaged (not squared), compared against sqrt of the squared bound.
    _single_vs_t(runs, raw_fn=lambda h: h["residual"][1:], bound_key="residual_sq_bound",
                 ylabel="(1/t) sum R(x_{s+1}, q_s)",
                 title=THM2_RESIDUAL_TEX,
                 filename=filename, bound_transform=np.sqrt)


def plot_theorem2_residual_vs_V(runs, filename="theorem2_residual_vs_V.png"):
    _single_vs_V(runs, raw_fn=lambda h: h["residual"][1:], bound_key="residual_sq_bound",
                 ylabel="(1/T) sum R(x_{s+1}, q_s)",
                 title=THM2_RESIDUAL_TEX,
                 filename=filename, bound_transform=np.sqrt)


def plot_theorem2_residual_vs_VT(vt_runs, filename="theorem2_residual_vs_VT.png"):
    _single_vs_VT(vt_runs, raw_fn=lambda h: h["residual"][1:], bound_key="residual_sq_bound",
                  ylabel="(1/T) sum R(x_{s+1}, q_s)",
                  title=THM2_RESIDUAL_TEX,
                  filename=filename, bound_transform=np.sqrt)


def plot_theorem3_vs_t(runs, filename="theorem3_vs_t.png"):
    _single_vs_t(runs, raw_fn=lambda h: np.maximum(h["g"][1:], 0.0), bound_key="g_violation_bound",
                 ylabel="(1/t) sum [g(x_{s+1})]_+",
                 title=THM3_TEX,
                 filename=filename)


def plot_theorem3_vs_V(runs, filename="theorem3_vs_V.png"):
    _single_vs_V(runs, raw_fn=lambda h: np.maximum(h["g"][1:], 0.0), bound_key="g_violation_bound",
                 ylabel="(1/T) sum [g(x_{s+1})]_+",
                 title=THM3_TEX,
                 filename=filename)


def plot_theorem3_vs_VT(vt_runs, filename="theorem3_vs_VT.png"):
    _single_vs_VT(vt_runs, raw_fn=lambda h: np.maximum(h["g"][1:], 0.0), bound_key="g_violation_bound",
                  ylabel="(1/T) sum [g(x_{s+1})]_+",
                  title=THM3_TEX,
                  filename=filename)


# ----------------------------------------------------------------------
# Theorem 3: beta sweep, both the real exact-penalty threshold and the
# (far more conservative) Theorem-3 sufficient threshold beta_star.
# ----------------------------------------------------------------------

def plot_beta_violation_tv(beta_runs, beta_star_val, filename="theorem3_beta_violation_tv.png"):
    beta = np.array([r["beta"] for r in beta_runs])
    avg_gplus = np.array([np.maximum(r["history"]["g"][1:], 0.0).mean() for r in beta_runs])
    bound = [r["bounds"]["g_violation_bound"] for r in beta_runs]
    bound_final = [b[-1] if b is not None else None for b in bound]

    fig, (ax_zoom, ax_full) = plt.subplots(1, 2, figsize=(12, 4.6))

    zoom_mask = beta <= 0.5
    ax_zoom.plot(beta[zoom_mask], avg_gplus[zoom_mask], "o-", color="tab:red")
    ax_zoom.set_xlabel("beta")
    ax_zoom.set_ylabel("avg [g(x)]_+")
    ax_zoom.set_title("Small beta: the ordinary constraint is actually violated\n(real exact-penalty threshold, not Theorem 3's)")
    ax_zoom.grid(True, alpha=0.3)

    ax_full.plot(beta, avg_gplus, "o-", color="tab:blue", label="empirical avg [g(x)]_+")
    lo, hi, pad = _data_range([avg_gplus])
    defined = [i for i, b in enumerate(bound_final) if b is not None]
    if defined:
        beta_d = beta[defined]
        bvals = np.array([bound_final[i] for i in defined])
        top = _band_overlay(ax_full, beta_d, bvals, hi, pad, label="Theorem 3 bound (shape only, where defined)")
        ax_full.set_ylim(lo - pad, top + pad)
    else:
        ax_full.set_ylim(lo - pad, hi + pad)
    ax_full.axvline(beta_star_val, color="black", linestyle=":", linewidth=1.3,
                     label=f"Theorem 3 threshold beta*={beta_star_val:.1f}")
    ax_full.set_xlabel("beta")
    ax_full.set_ylabel("avg [g(x)]_+")
    ax_full.set_title("Full beta range: Theorem 3's sufficient threshold is far more conservative")
    ax_full.legend(fontsize=8)
    ax_full.grid(True, alpha=0.3)

    fig.suptitle("Theorem 3: violating beta -- the real threshold vs. the proved one")
    fig.tight_layout()
    _save(fig, filename)


# ----------------------------------------------------------------------
# Assumption violations: xi (Slater margin), eta (stepsize window), gamma
# (stabilization window).
# ----------------------------------------------------------------------

def plot_assumption_violations_tv(viol_runs, problem, filename="assumption_violations_tv.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) Assumption 5/8 (Slater margin): Q_t, valid vs psi pushed past xi.
    ax = axes[0]
    valid = viol_runs["valid"]["history"]
    bad = viol_runs["slater_violated"]["history"]
    T = len(bad["Q"]) - 1
    t = np.arange(1, T + 1)
    ax.plot(t[:len(valid["Q"]) - 1], valid["Q"][1:], color="tab:blue",
            label=f"valid (psi_max={PSI_MAX_VALID}): saturates")
    ax.plot(t, bad["Q"][1:], color="tab:red",
            label=f"Slater violated (psi_max={PSI_MAX_SLATER_VIOLATE} > xi={problem.xi}): never saturates")
    ax.set_xlabel("t")
    ax.set_ylabel("Q_t")
    ax.set_title("Assumption 5/8 violated:\nqueue keeps growing instead of saturating")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Assumption 7 (eta window): mu_0(eta) = 1/eta - rho_f, with the
    # threshold and the chosen breaking eta marked.
    ax = axes[1]
    eta_thresh = 1.0 / problem.rho_f
    eta_bad = viol_runs["eta_too_large"]["eta"]
    eta_range = np.linspace(0.5 * eta_thresh, 1.5 * eta_bad, 200)
    mu0 = 1.0 / eta_range - problem.rho_f
    ax.plot(eta_range, mu0, color="tab:blue", label="mu_0(eta) = 1/eta - rho_f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(eta_thresh, color="gray", linestyle=":", label=f"threshold 1/rho_f={eta_thresh:.2f}")
    broke_at = viol_runs["eta_too_large"]["history"]["broke_at"]
    ax.plot([eta_bad], [1.0 / eta_bad - problem.rho_f], "rx", markersize=12, markeredgewidth=2.5,
            label=f"eta={eta_bad:.2f} used: solver breaks at t={broke_at}")
    ax.set_xlabel("eta")
    ax.set_ylabel("mu_0")
    ax.set_title("Assumption 7 violated:\nbase strong-convexity mu_0 <= 0, solver halts immediately")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Assumption 7 lower bound on gamma (gamma too small), forced to bite by
    # combining it with the Slater-violating psi and a small V: q_t vs t
    # up to the point where mu_t crosses 0.
    ax = axes[2]
    run = viol_runs["gamma_too_low"]
    hist = run["history"]
    broke_at = hist["broke_at"]
    T_show = broke_at + 20 if broke_at else len(hist["q"]) - 1
    t = np.arange(1, T_show + 1)
    ax.plot(t, hist["q"][1:T_show + 1], color="tab:purple", label="q_t")
    gamma_bad = run["gamma"]
    mu0 = 1.0 / run["eta"] - problem.rho_f
    q_break = mu0 / (problem.rho_h - 2 * gamma_bad)
    ax.axhline(q_break, color="black", linestyle=":", label=f"q_t where mu_t=0: {q_break:.2f}")
    if broke_at:
        ax.axvline(broke_at, color="tab:red", linestyle="--", label=f"solver breaks at t={broke_at}")
    ax.set_xlabel("t")
    ax.set_ylabel("q_t")
    ax.set_title(f"Assumption 7 violated (gamma={gamma_bad}):\nsubproblem loses convexity once q_t crosses the line")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Choosing the hyperparameters outside their windows: the theorems stop holding")
    fig.tight_layout()
    _save(fig, filename)
