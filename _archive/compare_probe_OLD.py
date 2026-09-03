"""Compare CN2 probe modes on a hard problem to validate convergence + cost."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.core import cn2, walltime
from problems.test_funcs import Rosenbrock, QuadIllCond, LogisticRegression, make_logistic_data

def run(prob, x0, tau, L, mode, **kw):
    x, info = cn2(prob.f, prob.grad, prob.hess, x0, tau=tau, L=L,
                  scan_mode="ratio", n0=10.0, probe_mode=mode, **kw)
    ff = prob.f(x)
    return ff, info

print("="*66)
print("Rosenbrock n=10 (hard: long valley, needs to reach solution space)")
print("="*66)
p = Rosenbrock(10)
x0 = -1.2*np.ones(10)
for mode in ["freeze","refresh","gate"]:
    kw = {"refresh_every":3} if mode=="refresh" else {"gate_tol":1e-3} if mode=="gate" else {}
    ff, info = run(p, x0, 1e-6, 200.0, mode, **kw)
    print(f"  {mode:>7}: f={ff:.3e}  it={info['iters']:>5}  hess={info['hess_evals']:>3}  "
          f"fresh={sum(1 for c in info['checkpoints'] if c['fresh'])} "
          f"wall({p.n})={walltime(p.n,info):.1e}")

print()
print("="*66)
print("Quadratic ill-cond n=10 k=1e4")
print("="*66)
q = QuadIllCond(10,1e4)
x0 = 5.0*np.ones(10)
for mode in ["freeze","refresh","gate"]:
    kw = {"refresh_every":3} if mode=="refresh" else {"gate_tol":1e-3} if mode=="gate" else {}
    ff, info = run(q, x0, 1e-6, 1e4, mode, **kw)
    print(f"  {mode:>7}: f={ff:.3e}  it={info['iters']:>5}  hess={info['hess_evals']:>3}  "
          f"fresh={sum(1 for c in info['checkpoints'] if c['fresh'])} "
          f"wall({q.n})={walltime(q.n,info):.1e}")

print()
print("="*66)
print("Logistic regression n=9")
print("="*66)
X,y = make_logistic_data(200,8,seed=0)
lr = LogisticRegression(X,y,reg=1e-3)
x0 = np.zeros(lr.n)
for mode in ["freeze","refresh","gate"]:
    kw = {"refresh_every":3} if mode=="refresh" else {"gate_tol":1e-3} if mode=="gate" else {}
    ff, info = run(lr, x0, 1e-8, None, mode, **kw)
    print(f"  {mode:>7}: f={ff:.3e}  it={info['iters']:>5}  hess={info['hess_evals']:>3}  "
          f"fresh={sum(1 for c in info['checkpoints'] if c['fresh'])} "
          f"wall({lr.n})={walltime(lr.n,info):.1e}")
