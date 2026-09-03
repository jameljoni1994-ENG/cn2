import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.core import cn2, walltime
from problems.test_funcs import (Rosenbrock, QuadIllCond, Beale, Himmelblau,
                                 LogisticRegression, make_logistic_data)

cases = [
    ("Rosen R2",  Rosenbrock(2),   np.array([-1.2, 1.0]), 1e-8, 1e-4, 200.0),
    ("Rosen R10", Rosenbrock(10),  -1.2*np.ones(10),      1e-8, 1e-4, 200.0),
    ("Quad",      QuadIllCond(10,1e4), 5.0*np.ones(10),   1e-8, 1e-4, 1e4),
    ("Beale",     Beale(),          np.array([1.0, 1.0]), 1e-8, 1e-4, 50.0),
    ("Himmel",    Himmelblau(),     np.array([-4.0,-4.0]),1e-8, 1e-2, 50.0),
]

for name, p, x0, tau, tau_g, L in cases:
    x, info = cn2(p.f, p.grad, p.hess, x0, tau=tau, K0=10, L=L,
                  tau_g=tau_g, newton_steps=2, max_cycle=300)
    print(f"{name:>9}: f={p.f(x):.3e}  it={info['iters']:>6}  "
          f"hess={info['hess_evals']:>3}  wall={walltime(p.n,info):.1e}")

# Logistic
X, y = make_logistic_data(200, 8, seed=0)
lr = LogisticRegression(X, y, reg=1e-3)
x, info = cn2(lr.f, lr.grad, lr.hess, np.zeros(lr.n), tau=1e-8, K0=10,
              tau_g=1e-4, newton_steps=2, max_cycle=300)
print(f"{'Logistic':>9}: f={lr.f(x):.3e}  it={info['iters']:>6}  "
      f"hess={info['hess_evals']:>3}")
