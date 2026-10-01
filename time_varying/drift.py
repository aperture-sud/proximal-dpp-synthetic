"""psi_t sequences: the additive drift that turns h into h_t = h + psi_t.

Both generators are driven by noise, not a deterministic function of t, so
psi_t is not an obvious closed form like psi_t = t*const. Each is a clipped
random walk: psi_{t+1} = clip(psi_t + step_t, psi_min, psi_max), where
step_t differs between the two scenarios below. Clipping to [psi_min,
psi_max] is what lets us control the worst-case Slater margin
xi_eff = xi - psi_max analytically, instead of hoping a random walk happens
to stay put.

The one-step functional variation of h_t is exactly |psi_{t+1} - psi_t|,
since h_t(x) - h_s(x) = psi_t - psi_s for every x (the h(x) part cancels).
"""
import numpy as np


def make_psi_drift(T, psi_max, rng, psi_min=0.0, bias=0.0006, sigma=0.012):
    """Slow random walk with a small upward bias, clipped to [psi_min, psi_max].

    This is the 'gradually tightening, noisy' scenario: h_t's offset wanders
    upward with fluctuation, not a ramp.
    """
    psi = np.zeros(T)
    for t in range(1, T):
        step = bias + rng.normal(0.0, sigma)
        psi[t] = np.clip(psi[t - 1] + step, psi_min, psi_max)
    return psi


def make_psi_burst(T, psi_max, rng, n_bursts=5, decay=0.997, sigma=0.004,
                    burst_frac=(0.5, 1.0), warmup_frac=0.1):
    """Mostly flat near 0, with a handful of sudden upward jumps that then
    decay back down -- the 'suddenly increase h' scenario.
    """
    psi = np.zeros(T)
    burst_times = set(rng.choice(np.arange(int(T * warmup_frac), T),
                                  size=min(n_bursts, T), replace=False).tolist())
    level = 0.0
    for t in range(1, T):
        level *= decay
        if t in burst_times:
            level += psi_max * rng.uniform(*burst_frac)
        level = np.clip(level, 0.0, psi_max)
        psi[t] = np.clip(level + rng.normal(0.0, sigma), 0.0, psi_max)
    return psi


def functional_variation(psi):
    """Delta_t = |psi_{t+1} - psi_t| for t=1..T-1, and its running cumulative
    sum V_T(t) = sum_{k<=t-1} Delta_k (V_T(1) = 0), aligned to psi's own
    indexing (length T, psi[0] is the t=1 value).
    """
    delta = np.abs(np.diff(psi))          # length T-1: delta[i] = |psi[i+1]-psi[i]|
    cum = np.concatenate([[0.0], np.cumsum(delta)])  # length T, cum[t-1] = V_T up to round t
    return delta, cum
