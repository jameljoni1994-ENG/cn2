"""Compare CN2 v3 (Newton-first + Nesterov coarse) on hard problems."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.core import cn2, walltime
from problems.test_funcs import Rosenbrock, QuadIllCond, LogisticRegression, make_logistic_data

def run(prob, x0, tau, L, K0=10, **kw):
    x, info = cn2(prob.f, prob.grad, prob.hess, x0, tau=tau, K0=K0, L=L,
                  newton_steps=2, **kw)
    return prob.f(x), info

print("="*68)
print("Rosenbrock n=2")
print("="*68)
p = Rosenbrock(2); x0=np.array([-1.2,1.0])
for K0 in [5,10,20]:
    ff, info = run(p, x0, 1e-8, 200.0, K0)
    print(f"  K0={K0:>3}: f={ff:.3e}  it={info['iters']:>5}  hess={info['hess_evals']:>3}  grad={info['grad_evals']:>5}")

print()
print("="*68)
print("Rosenbrock n=10")
print("="*68)
p = Rosenbrock(10); x0=-1.2*np.ones(10)
for K0 in [5,10,20]:
    ff, info = run(p, x0, 1e-8, 200.0, K0)
    print(f"  K0={K0:>3}: f={ff:.3e}  it={info['iters']:>5}  hess={info['hess_evals']:>3}  wall={walltime(p.n,info):.1e}")

print()
print("="*68)
print("Quadratic ill-cond n=10 k=1e4")
print("="*68)
q = QuadIllCond(10,1e4); x0=5.0*np.ones(10)
for K0 in [5,10,20]:
    ff, info = run(q, x0, 1e-8, 1e4, K0)
    print(f"  K0={K0:>3}: f={ff:.3e}  it={info['iters']:>5}  hess={info['hess_evals']:>3}  grad={info['grad_evals']:>5}")

print()
print("="*68)
print("Beale (needs care: x0=(1,1) is on flat region)")
print("="*68)
from problems.test_funcs import Beale
p = Beale(); x0=np.array([1.0,1.0])
for K0 in [10,20]:
    ff, info = run(p, x0, 1e-8, 50.0, K0)
    print(f"  K0={K0:>3}: f={ff:.3e}  it={info['iters']:>5}  hess={info['hess_evals']:>3}")
