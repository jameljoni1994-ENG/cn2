"""
Deepen the decay-rate validation.

Questions:
  Q1: How does p depend on step-size alpha?   (fixed beta)
  Q2: How does p depend on momentum beta?     (fixed alpha)
  Q3: Robustness of p across dimensions n and conditioning kappa.
  Q4: Does far-field superlinearity hold for nonconvex logistic regression?
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core import newton_decrement
from problems.test_funcs import Rosenbrock, QuadIllCond


def lambda_contraction_p(prob, x, alpha, beta, n_steps=150):
    """Return (p, c, R2) fit plus the lambda trajectory and counts."""
    x = np.asarray(x, dtype=float).copy()
    x_prev = x.copy()
    lam_b, lam_a = [], []

    for k in range(1, n_steps + 1):
        lb = newton_decrement(prob.grad(x), prob.hess(x))
        if not np.isfinite(lb) or lb <= 0:
            break
        b = beta if k > 1 else 0.0
        y = x + b * (x - x_prev)
        gy = prob.grad(y)
        x_new = y - alpha * gy
        x_prev = x.copy()
        x = x_new
        la = newton_decrement(prob.grad(x), prob.hess(x))
        if not np.isfinite(la) or la <= 0:
            break
        lam_b.append(lb)
        lam_a.append(la)

    if len(lam_b) < 5:
        return None, None, None, lam_b, lam_a

    logx = np.log(np.array(lam_b))
    logy = np.log(np.array(lam_a))
    A = np.vstack([logx, np.ones_like(logx)]).T
    coef, *_ = np.linalg.lstsq(A, logy, rcond=None)
    yhat = A @ coef
    ss_res = np.sum((logy - yhat) ** 2)
    ss_tot = np.sum((logy - logy.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef[0], float(np.exp(coef[1])), r2, lam_b, lam_a


def sweep_alpha(prob, x0, betas, alphas, label):
    print("=" * 62)
    print(f"[{label}]  sweep over alpha and beta")
    print("=" * 62)
    print(f"{'beta':>6} {'alpha':>9} {'p':>7} {'c':>9} {'R2':>6} {'n_step':>7}")
    for beta in betas:
        for al in alphas:
            p, c, r2, lb, la = lambda_contraction_p(prob, x0, al, beta)
            if p is None:
                print(f"{beta:>6.1f} {al:>9.1e} {np.nan:>7.2f} {'--':>9} {'--':>6} {'0':>7}")
            else:
                print(f"{beta:>6.1f} {al:>9.1e} {p:>7.3f} {c:>9.3e} {r2:>6.3f} {len(lb):>7d}")
    print()


def main():
    r = Rosenbrock(2)
    x0_far = np.array([-1.2, 1.0])

    print("\nQ1+Q2: sweep alpha and beta on Rosenbrock (far-field)")
    sweep_alpha(r, x0_far,
                betas=[0.0, 0.5, 0.9, 0.99],
                alphas=[1e-4, 1e-3, 1e-2],
                label="Rosenbrock 2D far")

    print("\nQ3: robustness across dimensions & conditioning")
    for n in [5, 10, 30]:
        for kappa in [1e2, 1e4, 1e6]:
            q = QuadIllCond(n=n, kappa=kappa)
            x0 = np.ones(n)
            p, c, r2, lb, la = lambda_contraction_p(q, x0, 1e-6 * min(1.0, 1/kappa*1e0), 0.9)
            # use a small alpha relative to 1/kappa_max
            alpha = 1e-3 / 1e4  # ensure below 1/kappa
            p, c, r2, lb, la = lambda_contraction_p(q, x0, 1e-8, 0.9)
            print(f"  n={n:>3} kappa={kappa:.0e} -> p={p:.3f} c={c:.3e} R2={r2:.3f}")


if __name__ == "__main__":
    main()
