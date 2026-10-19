"""Hegselmann–Krause bounded-confidence simulation behind the video script.

Every number shown on screen in script.md comes from this file.
Run:  python3 hk_sim.py            -> prints the numbers used in the script
      python3 hk_sim.py --json out -> also writes per-round positions for animation

Model (Hegselmann & Krause, 2002), synchronous update:
    x_i(t+1) = mean{ x_j : |x_i - x_j| <= eps_i }
"""
import json
import sys

import numpy as np

N = 100
X0 = np.linspace(0.0, 1.0, N)  # one evenly spaced row: the opening colour spectrum


def step(x, eps):
    near = np.abs(x[:, None] - x[None, :]) <= eps[:, None] + 1e-12
    return (near * x[None, :]).sum(1) / near.sum(1)


def run(x, eps, max_rounds=2000, tol=1e-9):
    """Return the list of states, round 0 first, until nothing moves."""
    eps = np.full(len(x), eps) if np.isscalar(eps) else eps
    trace = [x.copy()]
    for _ in range(max_rounds):
        nxt = step(trace[-1], eps)
        if np.max(np.abs(nxt - trace[-1])) < tol:
            break
        trace.append(nxt)
    return trace


def groups(x, gap=0.02):
    """Agents closer than `gap` count as one group. Returns [(position, size)]."""
    xs = np.sort(x)
    out = [[xs[0]]]
    for v in xs[1:]:
        if v - out[-1][-1] < gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [(round(float(np.mean(g)), 3), len(g)) for g in out]


def report(name, trace, skip=None):
    print(f"\n== {name} ==")
    for t, x in enumerate(trace):
        others = np.delete(x, skip) if skip is not None else x
        g = groups(others)
        extra = f"  bridge={x[skip]:.3f}" if skip is not None else ""
        print(f"round {t:>3}: spread={others.max() - others.min():.3f}  "
              f"groups={len(g)}  {g if len(g) <= 4 else ''}{extra}")


if __name__ == "__main__":
    wide = run(X0, 0.30)    # Experiment 1
    narrow = run(X0, 0.20)  # Experiment 2
    report("Experiment 1 · eps = 0.30", wide)
    report("Experiment 2 · eps = 0.20", narrow)

    # Bridge: one member of the left group widens its own eps to 0.40.
    BRIDGE = 10
    eps = np.full(N, 0.20)
    eps[BRIDGE] = 0.40
    bridge = run(narrow[-1], eps)
    report("Bridge · one agent eps = 0.40, everyone else 0.20", bridge, skip=BRIDGE)

    if "--random" in sys.argv:
        # Not shown on screen; backs the line "this does not always work".
        rng = np.random.default_rng(0)
        consensus = split = merged = 0
        for _ in range(300):
            x0 = rng.random(N)
            big = lambda x: [g for g in groups(x, gap=1e-3) if g[1] >= 5]
            if len(big(run(x0, 0.30)[-1])) == 1:
                consensus += 1
            end = run(x0, 0.20)[-1]
            if len(big(end)) == 2:
                split += 1
                eps = np.full(N, 0.20)
                eps[int(np.argmin(end))] = 0.40
                if len(big(run(end, eps)[-1])) == 1:
                    merged += 1
        print(f"\n== 300 random starts ==\neps 0.30 -> consensus in {consensus}/300"
              f"\neps 0.20 -> two groups in {split}/300"
              f"\nbridge eps 0.40 merged {merged}/{split} of those")

    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w") as f:
            json.dump({
                "n": N,
                "experiment1_eps030": [x.round(5).tolist() for x in wide],
                "experiment2_eps020": [x.round(5).tolist() for x in narrow],
                "bridge_index": BRIDGE,
                "bridge_eps040": [x.round(5).tolist() for x in bridge],
            }, f)
        print(f"\nwrote {path}")
