"""
Experimental validation of the decay-rate hypothesis:

    lambda_{k+1}  ~  c * lambda_k^p

for single Nesterov steps, across different functions and test points.

We measure the Newton decrement BEFORE and AFTER each single Nesterov
step and fit log(lambda_{k+1}) = p*log(lambda_k) + log(c).

If the hypothesis holds, the fitted exponent p should be stable (e.g. p~1
for generic, p>1 near-solution which is the "superlinear" regime the paper
wants). We also test whether p depends on the distance-to-solution regime.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core import newton_decrement, grad_fd, hess_fd
from problems.test_funcs import Rosenbrock, QuadIllCond


def measure_lambda_pairs(prob, x, alpha, n_steps=200):
    """
    Measure the contraction of lambda under ACTUAL Nesterov steps (momentum),
    which is what CN2 uses. Records (lambda_before, lambda_after) per step.
    Uses a tiny alpha to keep the trajectory stable.
    """
    x = np.asarray(x, dtype=float).copy()
    x_prev = x.copy()
    pairs = []

    for k in range(1, n_steps + 1):
        lam_before = newton_decrement(prob.grad(x), prob.hess(x))
        if not np.isfinite(lam_before):
            break

        beta = 0.9 if k > 1 else 0.0
        y = x + beta * (x - x_prev)
        gy = prob.grad(y)
        x_new = y - alpha * gy

        x_prev = x.copy()
        x = x_new

        lam_after = newton_decrement(prob.grad(x), prob.hess(x))
        if not np.isfinite(lam_after):
            break
        pairs.append((lam_before, lam_after))

    return pairs, x


def fit_power(logx, logy):
    """Fit log(y) = p*log(x) + log(c) via least squares; return p, c, R^2."""
    A = np.vstack([logx, np.ones_like(logx)]).T
    coef, res, *_ = np.linalg.lstsq(A, logy, rcond=None)
    p, logc = coef[0], coef[1]
    yhat = A @ coef
    ss_res = np.sum((logy - yhat) ** 2)
    ss_tot = np.sum((logy - logy.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return p, float(np.exp(logc)), r2


def analyze(prob, starts, alpha, label, n_steps=300):
    print("=" * 70)
    print(f"[{label}]  alpha={alpha}")
    print("=" * 70)
    for name, x0 in starts.items():
        pairs, x = measure_lambda_pairs(prob, x0, alpha, n_steps)
        lam_b = np.array([p[0] for p in pairs])
        lam_a = np.array([p[1] for p in pairs])

        # overall fit
        p, c, r2 = fit_power(np.log(lam_b), np.log(lam_a))
        print(f"-- {name}: x0={x0}")
        print(f"   overall  p={p:.4f}  c={c:.4e}  R^2={r2:.4f}")
        print(f"   lambda: start={lam_b[0]:.3e} -> end={lam_a[-1]:.3e}")

        # regime analysis: far vs near (split by lambda)
        med = np.median(lam_b)
        far = lam_b >= med
        near = ~far
        if far.sum() > 3 and near.sum() > 3:
            p_far, c_far, r2_far = fit_power(np.log(lam_b[far]), np.log(lam_a[far]))
            p_near, c_near, r2_near = fit_power(np.log(lam_b[near]), np.log(lam_a[near]))
            print(f"   FAR  regime p={p_far:.4f} c={c_far:.4e} R^2={r2_far:.4f}")
            print(f"   NEAR regime p={p_near:.4f} c={c_near:.4e} R^2={r2_near:.4f}")
        print()


def main():
    # 1) Quadratic ill-conditioned (strongly convex, linear regime expected)
    q = QuadIllCond(n=10, kappa=1e4)
    # very small alpha (<< 1/kappa_max=1e-4) ensures stability
    analyze(q,
            {"far": 1.0 * np.ones(10), "mid": 0.1 * np.ones(10)},
            alpha=1e-5, label="Quadratic ill-cond (kappa=1e4)")

    # 2) Rosenbrock (nonconvex, curved valley) — expect p changing near solution
    r = Rosenbrock(2)
    analyze(r,
            {"start1": np.array([-1.2, 1.0]),
             "near": np.array([0.9, 0.8])},
            alpha=1e-4, label="Rosenbrock 2D")


if __name__ == "__main__":
    main()
