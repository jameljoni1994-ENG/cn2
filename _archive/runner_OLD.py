"""
Comprehensive benchmark runner.

Runs all methods (CN², Newton, Newton-CG, L-BFGS, NAG, GD) on a battery of
test problems, collects the metrics (iterations, f-evals, grad-evals,
hess-evals, and a joint walltime model n-dependent), and emits:
  - a comparison table (CSV + printed)
  - per-problem convergence plots
"""

from __future__ import annotations

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core import cn2, newton, nesterov_agd, walltime, grad_fd
from benchmarks.baselines import lbfgs, newton_cg, gd
from problems.test_funcs import (
    Rosenbrock, QuadIllCond, Beale, Himmelblau,
    LogisticRegression, make_logistic_data,
)


PROBLEMS = {}


def register_problems():
    r2 = Rosenbrock(2)

    X, y = make_logistic_data(200, 8, seed=0)
    lr = LogisticRegression(X, y, reg=1e-3)

    PROBLEMS["Rosenbrock (n=2)"] = {
        "prob": r2, "x0": np.array([-1.2, 1.0]), "tau": 1e-6, "L": 200.0,
    }
    PROBLEMS["Rosenbrock (n=10)"] = {
        "prob": Rosenbrock(10),
        "x0": -1.2 * np.ones(10), "tau": 1e-6, "L": 200.0,
    }
    PROBLEMS["Quadratic ill-cond (n=10,k=1e4)"] = {
        "prob": QuadIllCond(10, 1e4),
        "x0": 5.0 * np.ones(10), "tau": 1e-6, "L": 1e4,
    }
    PROBLEMS["Beale"] = {
        "prob": Beale(), "x0": np.array([1.0, 1.0]), "tau": 1e-6, "L": 50.0,
    }
    PROBLEMS["Himmelblau"] = {
        "prob": Himmelblau(), "x0": np.array([-4.0, -4.0]), "tau": 1e-6, "L": 50.0,
    }
    PROBLEMS["Logistic Regression"] = {
        "prob": lr, "x0": np.zeros(lr.n), "tau": 1e-8, "L": None,
    }


def run_one(prob, x0, tau, L, method, params=None):
    f = prob.f
    g = prob.grad if hasattr(prob, "grad") and prob.grad is not None else None
    h = prob.hess if hasattr(prob, "hess") and prob.hess is not None else None

    if method == "cn2":
        x, info = cn2(f, g, h, x0, tau=tau, K0=20, L=L, newton_steps=2,
                      tau_g=1e-4, max_cycle=5000)
    elif method == "newton":
        x, info = newton(f, g, h, x0, tau_tol=tau, max_iter=200)
    elif method == "newton_cg":
        x, info = newton_cg(f, x0, grad=g, max_iter=200, tol=tau)
    elif method == "lbfgs":
        x, info = lbfgs(f, x0, grad=g, max_iter=500, tol=tau)
    elif method == "nag":
        x, info = nesterov_agd(f, g, x0, L=L, max_iter=2000, tol=tau,
                               momentum="nag")
    elif method == "gd":
        x, info = gd(f, x0, grad=g, L=L if L else 1.0, max_iter=2000, tol=tau)
    else:
        raise ValueError(method)

    info["final_f"] = float(f(x))
    info["final_x"] = np.asarray(x).tolist()
    info["grad_norm"] = float(np.linalg.norm(
        g(x) if g is not None else grad_fd(f, x)))
    info["method"] = method
    return info


def main():
    register_problems()
    methods = ["cn2", "newton", "newton_cg", "lbfgs", "nag", "gd"]

    results = []
    rows = []

    for name, cfg in PROBLEMS.items():
        prob = cfg["prob"]
        n = prob.n
        x0 = cfg["x0"]
        tau = cfg["tau"]
        L = cfg["L"]
        print("=" * 70)
        print(f"PROBLEM: {name}   (n={n})")
        print("=" * 70)
        for m in methods:
            try:
                info = run_one(prob, x0, tau, L, m)
            except Exception as e:
                print(f"  {m:>10}: FAILED ({e})")
                continue
            wt = walltime(n, info)
            row = {
                "problem": name, "method": m, "n": n,
                "final_f": info["final_f"], "grad_norm": info["grad_norm"],
                "iters": info["iters"], "f_evals": info["f_evals"],
                "grad_evals": info["grad_evals"],
                "hess_evals": info["hess_evals"], "walltime": wt,
            }
            results.append(row)
            rows.append(info)
            print(f"  {m:>10}: f={info['final_f']:.3e} "
                  f"it={info['iters']:>4} hess={info['hess_evals']:>5} "
                  f"wt={wt:.1e}")

    print()
    print("=" * 70)
    print("SUMMARY TABLE (final function value; lower is better)")
    print("=" * 70)
    hdr = f"{'Method':>12} | " + " | ".join(f"{name[:22]:>22}" for name in PROBLEMS)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        cells = []
        for name in PROBLEMS:
            rr = [r for r in results if r["method"] == m and r["problem"] == name]
            cells.append(f"{rr[0]['final_f']:.1e}" if rr else "FAIL")
        print(f"{m:>12} | " + " | ".join(f"{c:>22}" for c in cells))

    # hessian-eval summary (the key metric for CN²)
    print()
    print("HESSIAN-EVALS TABLE (the key claim: CN² uses far fewer)")
    hdr = f"{'Method':>12} | " + " | ".join(f"{name[:22]:>22}" for name in PROBLEMS)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        cells = []
        for name in PROBLEMS:
            rr = [r for r in results if r["method"] == m and r["problem"] == name]
            cells.append(f"{rr[0]['hess_evals']}" if rr else "FAIL")
        print(f"{m:>12} | " + " | ".join(f"{c:>22}" for c in cells))

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved -> results/benchmark_results.json")


if __name__ == "__main__":
    main()
