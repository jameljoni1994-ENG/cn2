"""
First smoke-test of CN² on 2D Rosenbrock.

Goal: verify the algorithm behaves correctly (checkpoint sequence converges
to the solution basin, then Newton takes over). This is a minimal prototype
that we will expand into a full benchmark.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core import cn2, newton, nesterov_agd, newton_decrement
from problems.test_funcs import Rosenbrock


def main():
    prob = Rosenbrock(2)
    x0 = np.array([-1.2, 1.0])  # classic Rosenbrock start

    tau = 1e-6
    K0 = 20
    L = 200.0  # rough Lipschitz estimate for Rosenbrock local region

    print("=" * 60)
    print("CN² on 2D Rosenbrock")
    print(f"x0 = {x0}, tau = {tau}, K0 = {K0}, L = {L}")
    print("=" * 60)

    x_opt, checkpoints, hist = cn2(
        prob.f, prob.grad, prob.hess, x0,
        tau=tau, K0=K0, L=L, verbose=True,
    )

    print("\n--- Checkpoint sequence (lambda should decrease) ---")
    prev = None
    for c in checkpoints:
        lam = c["lambda"]
        trend = ""
        if prev is not None:
            trend = "DEC" if lam < prev else "INC!!"
        print(f"f={c['f']:.6e}  lambda={lam:.3e}  {trend}")
        prev = lam

    print(f"\nFinal x = {x_opt}")
    print(f"Final f = {prob.f(x_opt):.6e}")
    print(f"Gradient norm = {np.linalg.norm(prob.grad(x_opt)):.3e}")

    # sanity: pure Newton comparison
    x_n, hist_n = newton(prob.f, prob.grad, prob.hess, x0, tau_tol=tau)
    print(f"\nPure Newton final f = {prob.f(x_n):.6e}, iters = {len(hist_n)}")

    # pure Nesterov comparison
    x_nag, hist_nag = nesterov_agd(prob.f, prob.grad, x0, L=L, max_iter=2000)
    print(f"Pure NAG final f = {prob.f(x_nag):.6e}, iters = {len(hist_nag)}")


if __name__ == "__main__":
    main()
