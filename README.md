# Proximal Drift-Plus-Penalty: Synthetic Validation

Numerical experiments for a proximal drift-plus-penalty (DPP) method for weakly
convex optimization with one ordinary constraint `g(x) <= 0` and one
time-average constraint on `h_t`. Each experiment runs the algorithm on a
synthetic instance whose constants (weak-convexity moduli, Lipschitz bounds,
etc.) are known in closed form, and plots the empirical quantities against the
bounds predicted by the paper's lemmas and theorems.

There are two experiments:

- **static**: a fixed constraint `h`.
- **time_varying**: a drifting constraint `h_t(x) = h(x) + psi_t`, where `psi_t`
  is a clipped random walk.

## Setup

Tested with Python 3.10.12.

```
pip install -r requirements.txt
```

## Running

Run everything from the repository root.

| Command | What it does | Output |
|---|---|---|
| `python3 -m static.main` | All static sweeps (V, beta, gamma, eta, Corollary 1) and the default-point trajectory | `figures/static/`, `results/trajectory.txt` |
| `python3 -m static.checks` | PASS/FAIL correctness checks for the static experiment | `results/report.txt` |
| `python3 -m time_varying.main` | All time-varying sweeps (V, V_T, beta, assumption violations) | `figures/time_varying/` |

`static.main` takes a few minutes (the Corollary 1 sweep runs up to
T = 125,000); `time_varying.main` takes longer. Every run is seeded, so the
outputs are reproducible exactly.

## Layout

```
static/          problem instance, Algorithm 1, predicted bounds, sweeps, plots, checks
time_varying/    the same for h_t = h + psi_t, plus the psi_t generators (drift.py);
                 reuses static/problem.py and static/algorithm.py
figures/         generated figures, one folder per experiment
results/         generated text reports
slides/          writeup.tex (static), writeup_tv.tex (time-varying)
paper/           paper draft
topresent/       PDFs as presented (a fixed record; not regenerated)
```

## Slides

```
cd slides
pdflatex writeup.tex
pdflatex writeup_tv.tex
```

`writeup.tex` needs the LaTeX package `algpseudocode` (in TeX Live's
`algorithmicx`; on Debian/Ubuntu, `texlive-science`).
