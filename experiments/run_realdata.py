"""
E2. Real datasets — binomial logistic regression (L2-regularized empirical risk).

Datasets (sklearn, no downloads needed):
  - breast_cancer (569 x 30, binary)
  - digit 3-vs-8 subset (n~360 features reduced to 32 via PCA-lite: we just use
    the raw 64 dims -> d=65 params; still moderate for dense Newton)

Methods: CN2 (dense), Newton (dense), L-BFGS, Newton-CG (from baselines).
Metric: final f and Hessian evaluations (hess_evals).
"""
from __future__ import annotations

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sklearn.datasets

from src.core import cn2, newton
from benchmarks.baselines import lbfgs, newton_cg
from problems.test_funcs import LogisticRegression


def build(name):
    if name == "breast_cancer":
        X, y0 = sklearn.datasets.load_breast_cancer(return_X_y=True)
        y = np.where(y0 == 1, 1.0, -1.0)
        X -= X.mean(axis=0)
        X /= (X.std(axis=0) + 1e-12)
        return X, y
    if name == "digits_3v8":
        X, y0 = sklearn.datasets.load_digits(return_X_y=True)
        mask = np.isin(y0, [3, 8])
        X, y0 = X[mask], y0[mask]
        y = np.where(y0 == 8, 1.0, -1.0)
        X -= X.mean(axis=0)
        X /= (X.std(axis=0) + 1e-12)
        return X, y
    raise ValueError(name)


def run_case(name, reg):
    X, y = build(name)
    p = LogisticRegression(X, y, reg=reg)
    d = p.n
    x0 = np.zeros(d)
    info_all = {}

    x, info = cn2(p.f, p.grad, p.hess, x0, tau=1e-8, K0=20, L=None,
                  tau_g=1e-4, newton_steps=2, max_cycle=300)
    info_all["CN2"] = (p.f(x), info["hess_evals"], info["iters"])

    x, info = newton(p.f, p.grad, p.hess, x0, tau_tol=1e-8, max_iter=300)
    info_all["Newton"] = (p.f(x), info["hess_evals"], info["iters"])

    x, info = lbfgs(p.f, x0, grad=p.grad, max_iter=1000, tol=1e-8)
    info_all["L-BFGS"] = (p.f(x), info["hess_evals"], info["iters"])

    x, info = newton_cg(p.f, x0, grad=p.grad, max_iter=100, tol=1e-8, cg_max=20)
    info_all["Newton-CG"] = (p.f(x), info["hess_evals"], info["iters"])

    return {"d": d, "samples": X.shape[0], "methods": {
        k: {"f": v[0], "hess": v[1], "it": v[2]} for k, v in info_all.items()}}


def main():
    out = {}
    print("=" * 68)
    print("E2: real-data logistic regression")
    print("=" * 68)
    for name in ("breast_cancer", "digits_3v8"):
        print(f"\n-- {name}")
        res = run_case(name, reg=1e-3)
        out[name] = res
        print(f"   d={res['d']}  samples={res['samples']}")
        for k, v in res["methods"].items():
            print(f"   {k:>10}: f={v['f']:.6e}  hess={v['hess']:>4}  "
                  f"it={v['it']:>6}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/realdata.json", "w"), indent=2)
    print("\nSaved -> results/realdata.json")


if __name__ == "__main__":
    main()