"""
CN² — one-hour-budget experiment suite for a mid-range laptop (i5-13420H,
16 GB, dense NumPy).

Pipeline (each stage is wall-timed and guarded by a hard cumulative budget):
  1. Reproduce the 5 existing experiment scripts (baseline ~4 min).
  2. T1  : per-cycle cost sweep  cost(K_ck, n)   for n x K0 grid
  3. Mseed : multi-seed reproducibility (method x problem x seed)
  4. Params : method-parameter sensitivity (L on Rosen, tau_g on Beale)
  5. CRN : cubic-regularized Newton baseline on nonconvex problems

Everything is saved to results/quick_suite.json with per-stage timings; the
runner trims seeds/parameters automatically if the elapsed time threatens the
budget, and stops completely if the budget is exhausted.

Usage:  py experiments/run_quick_suite.py [--budget SECONDS]
"""
from __future__ import annotations

import sys, os, json, time, subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core import cn2
from benchmarks.baselines import newton_cg, lbfgs
from problems.test_funcs import Rosenbrock, QuadIllCond, Beale, Himmelblau

BUDGET = 3600.0
np.seterr(all="ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Budget:
    """Cumulative wall-clock budget with ``remaining`` and a note field."""

    def __init__(self, seconds):
        self.hard = seconds
        self.notes = []
        self.t0 = time.perf_counter()

    @property
    def elapsed(self):
        return time.perf_counter() - self.t0

    @property
    def remaining(self):
        return self.hard - self.elapsed

    def ok(self, toneed=0.0):
        return self.remaining > toneed


def run_script(name):
    """Run an existing experiment script in-process-subprocess, return wall s."""
    t0 = time.perf_counter()
    sp = subprocess.run([sys.executable, os.path.join("experiments", name)],
                        cwd=ROOT)
    if sp.returncode != 0:
        print(f"  !! {name} exited {sp.returncode}")
        return None
    return round(time.perf_counter() - t0, 2)


# ----------------------------------------------------------------------
# T1: per-cycle cost sweep  cost(K_ck, n)
# ----------------------------------------------------------------------
def t1_cost_sweep(budget):
    rows, traces = [], {}
    t0 = time.perf_counter()
    print("\n--- T1: per-cycle cost sweep cost(K_ck, n) ---")
    for n in [10, 50, 100, 200, 500]:
        p, x0, tau, L = Rosenbrock(n), -1.2 * np.ones(n), 1e-8, 200.0
        for K0 in [1, 10, 20, 50, 100]:
            if time.perf_counter() - t0 > 240:      # T1 self-cap (~4 min)
                budget.notes.append("T1 trimmed: n/K0 grid truncated at 4 min")
                return rows, traces
            trace = []
            t1 = time.perf_counter()
            x, info = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=L, tau_g=1e-4,
                          K0=K0, max_cycle=60, trace=trace)
            rechecks = [e for e in trace if e.get("kind") == "recheck"]
            cycles = [e for e in trace if e.get("kind") == "cycle"]
            # per-checkpoint Hessian cost = delta in cumulative counters
            h_prev = 0
            hess_per_recheck = []
            for e in rechecks:
                hess_per_recheck.append(e["hess"] - h_prev)
                h_prev = e["hess"]
            rows.append({
                "n": n, "K0": K0, "k_ck": info["k_ck"],
                "f": float(p.f(x)), "hess": info["hess_evals"],
                "grad": info["grad_evals"], "iters": info["iters"],
                "rechecks": len(rechecks), "transits": len(cycles),
                "steps_per_recheck": round(info["iters"] / max(len(rechecks), 1), 2),
                "hessian_per_recheck": sorted(set(hess_per_recheck))
                                        if hess_per_recheck else [0],
                "wall_s": round(time.perf_counter() - t1, 3),
            })
            traces[f"n{n}_K0{K0}"] = trace
            print(f"  n={n:>4} K0={K0:>4} (k_ck={info['k_ck']:>4}): "
                  f"f={p.f(x):.1e} hess={info['hess_evals']:>4} "
                  f"rechecks={len(rechecks):>5} "
                  f"hess/recheck={rows[-1]['hessian_per_recheck']}",
                  flush=True)
    return rows, traces


def make_t1_plot(rows, path="figures/t1_cost.png"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in sorted({r["n"] for r in rows}):
        rr = sorted([r for r in rows if r["n"] == n], key=lambda r: r["k_ck"])
        ax.plot([r["k_ck"] for r in rr], [r["hess"] for r in rr],
                marker="o", label=f"n={n}")
    ax.set_xlabel(r"checkpoint cadence  $K_{ck}=\max(K_0, n/5)$")
    ax.set_ylabel("total Hessian evals (one transit)")
    ax.set_title("T1: verification cost vs checkpoint rate (Rosenbrock)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Multi-seed reproducibility
# ----------------------------------------------------------------------
def mseed_reproducibility(budget, seeds=(0, 1, 2, 3, 4)):
    rows = []
    t0 = time.perf_counter()
    print("\n--- Multi-seed reproducibility (method x problem x seed) ---")
    for si, seed in enumerate(seeds):
        if time.perf_counter() - t0 > 180 or not budget.ok(120):  # ~3 min cap
            budget.notes.append(f"multi-seed trimmed after seed {seed}")
            break
        rng = np.random.default_rng(seed)
        problems = {
            "Rosen n=50":  (Rosenbrock(50),  -1.2 * np.ones(50) + 0.1 * rng.normal(size=50), 1e-8, 200.0),
            "Rosen n=100": (Rosenbrock(100), -1.2 * np.ones(100) + 0.1 * rng.normal(size=100), 1e-8, 200.0),
            "Beale":       (Beale(),         np.array([1.0, 1.0]) + 0.1 * rng.normal(size=2), 1e-8, 50.0),
            "Himmelblau":  (Himmelblau(),    np.array([-4.0, -4.0]) + 0.1 * rng.normal(size=2), 1e-8, 50.0),
        }
        from problems.test_funcs import LogisticRegression, make_logistic_data
        X, y = make_logistic_data(300, 30, seed=seed)
        lr = LogisticRegression(X, y, reg=1e-3)
        problems["Logistic p=30"] = (lr, np.zeros(lr.n), 1e-8, None)

        for name, (p, x0, tau, L) in problems.items():
            for m in ("cn2", "newton_cg", "lbfgs"):
                t1 = time.perf_counter()
                try:
                    if m == "cn2":
                        x, info = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=L,
                                      tau_g=1e-4, newton_steps=2, max_cycle=500)
                    elif m == "newton_cg":
                        x, info = newton_cg(p.f, x0, grad=p.grad, max_iter=60,
                                            tol=tau, cg_max=15)
                    else:
                        x, info = lbfgs(p.f, x0, grad=p.grad, max_iter=1000,
                                        tol=tau)
                    fv = float(p.f(x))
                except Exception as e:
                    rows.append({"seed": seed, "problem": name, "method": m,
                                 "error": str(e)[:80], "wall_s": 0.0})
                    continue
                rows.append({
                    "seed": seed, "problem": name, "method": m,
                    "f": fv, "iters": info["iters"],
                    "hess": info.get("hess_evals", 0),
                    "hv": info.get("hv_evals", 0),
                    "grad": info["grad_evals"],
                    "wall_s": round(time.perf_counter() - t1, 3),
                })
        print(f"  seed={seed}: done", flush=True)
    return rows


# ----------------------------------------------------------------------
# Method-parameter sensitivity: L (Rosen) and tau_g (Beale)
# ----------------------------------------------------------------------
def param_sensitivity(budget):
    rows = []
    print("\n--- Parameter sensitivity: L on Rosen n=50, tau_g on Beale ---")
    p, x0, tau = Rosenbrock(50), -1.2 * np.ones(50), 1e-8
    for L in [50.0, 100.0, 200.0, 400.0]:
        t1 = time.perf_counter()
        x, info = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=L, tau_g=1e-4,
                      max_cycle=500)
        rows.append({"param": "L", "value": L, "problem": "Rosen n=50",
                     "f": float(p.f(x)), "hess": info["hess_evals"],
                     "iters": info["iters"], "wall_s": round(time.perf_counter() - t1, 2)})
    p = Beale(); x0 = np.array([1.0, 1.0]); tau = 1e-8
    for tg in [1e-4, 1e-6, 1e-8]:
        t1 = time.perf_counter()
        x, info = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=50.0, tau_g=tg,
                      max_cycle=500)
        rows.append({"param": "tau_g", "value": tg, "problem": "Beale",
                     "f": float(p.f(x)), "hess": info["hess_evals"],
                     "iters": info["iters"], "wall_s": round(time.perf_counter() - t1, 2)})
    if not budget.ok(60):
        budget.notes.append("param sweep truncated")
    return rows


# ----------------------------------------------------------------------
# CRN (cubic-regularized Newton) baseline
# ----------------------------------------------------------------------
def crn(f, grad, hess, x0, sigma=1.0, max_iter=60, tol=1e-8):
    """Minimal cubic-regularized Newton: d = argmin g d + 1/2 d H d + s/3||d||^3
    via the fixed point (H + sigma*||d|| I) d = -g, with backtracking."""
    x = np.asarray(x0, dtype=float).copy()
    hess_cnt = grad_cnt = f_cnt = 0
    for _ in range(max_iter):
        g = grad(x); grad_cnt += 1
        H = hess(x); hess_cnt += 1
        if np.linalg.norm(g) < tol:
            break
        d = np.linalg.solve(H + 1e-6 * np.eye(x.size), -g)
        for _ in range(15):                     # cubic fixed point
            lam = sigma * np.linalg.norm(d)
            dn = np.linalg.solve(H + lam * np.eye(x.size), -g)
            if np.linalg.norm(dn - d) < 1e-10 * np.linalg.norm(d):
                d = dn
                break
            d = dn
        t = 1.0
        f0 = f(x); f_cnt += 1
        while f(x + t * d) > f0 and t > 1e-14:
            t *= 0.5
            f_cnt += 1
        x = x + t * d
    return x, {"iters": hess_cnt, "hess_evals": hess_cnt,
               "grad_evals": grad_cnt, "f_evals": f_cnt}


def crn_baseline(budget):
    rows = []
    print("\n--- CRN baseline (cubic-regularized Newton) on nonconvex ---")
    cases = [(Beale(), np.array([1.0, 1.0]), "Beale"),
             (Rosenbrock(50), -1.2 * np.ones(50), "Rosen n=50")]
    for p, x0, name in cases:
        for sigma in [0.5, 1.0, 2.0]:
            t1 = time.perf_counter()
            x, info = crn(p.f, p.grad, p.hess, x0, sigma=sigma, max_iter=50)
            rows.append({"problem": name, "sigma": sigma, "f": float(p.f(x)),
                         "iters": info["iters"], "hess": info["hess_evals"],
                         "conv": float(p.f(x)) < 1e-6,
                         "wall_s": round(time.perf_counter() - t1, 2)})
            print(f"  {name:>10} sigma={sigma}: f={p.f(x):.2e} "
                  f"hess={info['hess_evals']} conv={p.f(x) < 1e-6}", flush=True)
    if not budget.ok(30):
        budget.notes.append("CRN truncated")
    return rows


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
def main():
    budget = Budget(float(sys.argv[1]) if len(sys.argv) > 1 else BUDGET)
    print("=" * 72)
    print(f"CN² quick suite  (budget: {budget.hard:.0f}s on this machine)")
    print(f"hardware: CPU={os.environ.get('PROCESSOR_IDENTIFIER', '?')} "
          f"cores={os.cpu_count()}")
    print("=" * 72)

    stages = {}

    # 1) reproduction of current scripts
    print("\n[1/5] Reproducing current experiment scripts")
    for name in ["run_extended.py", "run_realdata.py", "run_trajectory.py",
                 "run_all.py", "run_largeN.py"]:
        if not budget.ok(0):
            stages[name] = "skipped (budget)"
            print(f"  {name}: skipped (budget exhausted)")
            continue
        wall = run_script(name)
        stages[name] = wall
        if wall is not None:
            print(f"  {name}: {wall}s  (elapsed {budget.elapsed:.0f}s / "
                  f"{budget.hard:.0f}s)")

    # 2) T1
    print("\n[2/5] T1 per-cycle cost sweep")
    if budget.ok(60):
        t0 = time.perf_counter()
        t1_rows, traces = t1_cost_sweep(budget)
        stages["T1_cost_sweep_s"] = round(time.perf_counter() - t0, 2)
        print(f"  T1 done in {stages['T1_cost_sweep_s']}s")
        t1_fig = make_t1_plot(t1_rows)
        print(f"  T1 figure -> {t1_fig}")
    else:
        stages["T1_cost_sweep_s"] = "skipped (budget)"
        t1_rows, traces, t1_fig = [], {}, None

    # 3) multi-seed
    print("\n[3/5] Multi-seed reproducibility")
    if budget.ok(60):
        t0 = time.perf_counter()
        mseed = mseed_reproducibility(budget)
        stages["mseed_s"] = round(time.perf_counter() - t0, 2)
        print(f"  multi-seed done in {stages['mseed_s']}s")
    else:
        stages["mseed_s"] = "skipped (budget)"
        mseed = []

    # 4) parameter sensitivity
    print("\n[4/5] Parameter sensitivity (L / tau_g)")
    if budget.ok(60):
        t0 = time.perf_counter()
        params = param_sensitivity(budget)
        stages["param_s"] = round(time.perf_counter() - t0, 2)
    else:
        stages["param_s"] = "skipped (budget)"
        params = []

    # 5) CRN
    print("\n[5/5] CRN baseline")
    if budget.ok(60):
        t0 = time.perf_counter()
        crn_rows = crn_baseline(budget)
        stages["crn_s"] = round(time.perf_counter() - t0, 2)
    else:
        stages["crn_s"] = "skipped (budget)"
        crn_rows = []

    # aggregate
    out = {
        "machine": {
            "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "?"),
            "logical_cores": os.cpu_count(),
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "budget_s": budget.hard,
        "elapsed_s": round(budget.elapsed, 2),
        "notes": budget.notes,
        "stages": stages,
        "T1": t1_rows,
        "T1_traces": traces,
        "t1_fig": t1_fig,
        "multi_seed": mseed,
        "params": params,
        "crn": crn_rows,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/quick_suite.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"TOTAL: {budget.elapsed:.1f}s / {budget.hard:.0f}s  "
          f"(margin {budget.remaining:.0f}s)")
    if budget.notes:
        print("notes:")
        for n in budget.notes:
            print(f"  - {n}")
    print("Saved -> results/quick_suite.json")
    if len(t1_rows):
        print("\nT1 n x K0 -> (hess, rechecks, hess/recheck):")
        for r in t1_rows:
            print(f"  n={r['n']:>4} K0={r['K0']:>4} k_ck={r['k_ck']:>4}: "
                  f"hess={r['hess']:>4} rechecks={r['rechecks']:>5} "
                  f"hess/recheck={r['hessian_per_recheck']} f={r['f']:.1e}")


if __name__ == "__main__":
    main()