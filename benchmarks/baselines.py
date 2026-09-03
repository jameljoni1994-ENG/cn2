"""
Baseline methods (manual implementations for fair comparison).

  - lbfgs        : Limited-memory BFGS (quasi-Newton, no Hessian needed)
  - newton_cg    : Newton with conjugate-gradient inner solver (Hessian-free
                   using Hessian-vector products via finite differences)
  - gd           : plain gradient descent with backtracking (for reference)

Each returns (x, info) with f_hist, grad_evals, f_evals, hess_evals, iters.
"""

from __future__ import annotations

import numpy as np
from src.core import grad_fd


def _Hv(f, x, v, h=1e-6):
    """Hessian-vector product via finite differences of the gradient:
    H v ~ (grad(x + h v) - grad(x - h v)) / 2h."""
    xp = x + h * v
    xm = x - h * v
    return (grad_fd(f, xp) - grad_fd(f, xm)) / (2 * h)


def lbfgs(f, x0, grad=None, max_iter=500, m=10, tol=1e-8):
    """Limited-memory BFGS (two-loop recursion)."""
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    g = grad(x) if grad is not None else grad_fd(f, x)
    grad_evals = 1
    f_evals = 1
    Hdiag = 1.0
    S = []  # list of np arrays s_k
    Y = []
    hist = [float(f(x))]
    f_evals += 1

    for _ in range(max_iter):
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            break

        # two-loop recursion
        if len(S) == 0:
            d = -g
        else:
            q = g.copy()
            alphas = []
            for s, y in zip(reversed(S), reversed(Y)):
                rho = 1.0 / np.dot(y, s)
                a = rho * np.dot(s, q)
                alphas.append(a)
                q = q - a * y
            r = Hdiag * q
            for s, y, a in zip(S, Y, reversed(alphas)):
                rho = 1.0 / np.dot(y, s)
                b = rho * np.dot(y, r)
                r = r + s * (a - b)
            d = -r

        # backtracking line search
        t = 1.0
        f0 = f(x); f_evals += 1
        gd = np.dot(g, d)
        while f(x + t * d) > f0 + 1e-4 * t * gd and t > 1e-16:
            t *= 0.5
            f_evals += 1
        s = t * d
        x_new = x + s
        g_new = grad(x_new) if grad is not None else grad_fd(f, x_new)
        grad_evals += 1
        y = g_new - g
        # skip if not positive curvature
        if np.dot(s, y) > 1e-10:
            S.append(s.copy())
            Y.append(y.copy())
            if len(S) > m:
                S.pop(0); Y.pop(0)
            sy = np.dot(y, s)
            Hdiag = sy / max(np.dot(y, y), 1e-30)

        g = g_new
        x = x_new
        hist.append(float(f(x)))
        f_evals += 1
        if len(hist) > max_iter:
            break

    return x, {"f_hist": hist, "grad_evals": grad_evals, "f_evals": f_evals,
               "hess_evals": 0, "iters": len(hist)}


def newton_cg(f, x0, grad=None, max_iter=200, tol=1e-8, cg_max=20):
    """Newton-CG (truncated Newton). Hessian-free via Hessian-vector products
    approximated by finite differences of gradients. No full Hessian formed."""
    x = np.asarray(x0, dtype=float).copy()
    hist = [float(f(x))]
    grad_evals = 1
    f_evals = 1
    hv_evals = 0
    iters = 0

    for _ in range(max_iter):
        g = grad(x) if grad is not None else grad_fd(f, x)
        grad_evals += 1
        if np.linalg.norm(g) < tol:
            break

        # CG to solve H d = -g
        d = np.zeros_like(x)
        r = -g.copy()
        p = r.copy()
        rsold = np.dot(r, r)
        if rsold < 1e-30:
            break
        for _ in range(cg_max):
            Hp = _Hv(f, x, p)
            f_evals += 2  # two gradient evals in Hv
            hv_evals += 1
            Ap = np.dot(p, Hp)
            if abs(Ap) < 1e-30:
                break
            alpha_cg = rsold / Ap
            d = d + alpha_cg * p
            r = r - alpha_cg * Hp
            rsnew = np.dot(r, r)
            if np.sqrt(rsnew) < np.sqrt(np.dot(g, g)) * 1e-2:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        # line search along d
        t = 1.0
        f0 = f(x); f_evals += 1
        gd = np.dot(g, d)
        if gd >= 0:
            d = -g
            gd = -np.dot(g, g)
        while f(x + t * d) > f0 + 1e-4 * t * gd and t > 1e-16:
            t *= 0.5
            f_evals += 1
        x = x + t * d
        iters += 1
        hist.append(float(f(x)))
        f_evals += 1

    return x, {"f_hist": hist, "grad_evals": grad_evals, "f_evals": f_evals,
               "hess_evals": 0, "hv_evals": hv_evals, "iters": iters}


def gd(f, x0, grad=None, L=1.0, max_iter=2000, tol=1e-8):
    """Gradient descent with backtracking."""
    x = np.asarray(x0, dtype=float).copy()
    hist = [float(f(x))]
    grad_evals = 1
    f_evals = 1
    iters = 0
    for _ in range(max_iter):
        g = grad(x) if grad is not None else grad_fd(f, x)
        grad_evals += 1
        if np.linalg.norm(g) < tol:
            break
        a = 1.0 / L
        f0 = f(x); f_evals += 1
        gnorm2 = np.dot(g, g)
        while a > 1e-16:
            x_new = x - a * g
            f_new = f(x_new); f_evals += 1
            if f_new <= f0 - 0.5 * a * gnorm2:
                break
            a *= 0.5
        x = x_new
        iters += 1
        hist.append(float(f(x)))
    return x, {"f_hist": hist, "grad_evals": grad_evals, "f_evals": f_evals,
               "hess_evals": 0, "iters": iters}
