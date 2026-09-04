"""
Large-n experiments: CN2 with Hessian-vector products (HVP) instead of dense
Hessians.  Targets n = 500 .. 1000 with O(n) memory per iteration.

Methods compared (all Hessian-free):
  - CN2-HVP : the CN2 dual gate + Newton-CG inner solver via HVP.
  - Newton-CG (baseline): CG-Newton on HVP.
  - L-BFGS (baseline): manual two-loop recursion, gradient-only.

Reports final f, iterations, and a proxy for Hessian cost: CG inner iterations
(hv_evals) + Hessian formations (0 for all).
"""
from __future__ import annotations

import os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problems.test_funcs import Rosenbrock, QuadIllCond
from src.hvp_cg import HvpOperator, cg_solve, newton_decrement_cg, lanczos_min_eig


def sprint(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
def newton_cg_hvp(grad, x0, tau_tol=1e-8, max_iter=200, eps=1e-8, cg_max=30):
    x = np.asarray(x0, dtype=float).copy()
    hv = 0
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tau_tol:
            break
        A = HvpOperator(grad, x, eps=eps)
        d, it = cg_solve(A, -g, tol=1e-6, max_iter=cg_max)
        hv += it
        x = x + d
    return x, hv


def lbfgs_hvp(grad, f, x0, max_iter=2000, tol=1e-8, m=10):
    x = np.asarray(x0, dtype=float).copy()
    S, Y, RHO = [], [], []
    g = grad(x)
    hist = []
    for it in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        q = -g.copy()
        if len(S) > 0:
            alphas = [rho * np.dot(s, q) for s, y, rho in zip(reversed(S), reversed(Y), reversed(RHO))]
            q = q - sum(a * y for a, (s, y, rho) in zip(alphas, zip(S, Y, RHO)))
            gamma = np.dot(S[-1], Y[-1]) / (np.dot(Y[-1], Y[-1]) + 1e-12)
            q = gamma * q
            for a, (s, y, rho) in zip(reversed(alphas), zip(S, Y, RHO)):
                b = rho * np.dot(y, q)
                q = q + (a - b) * s
        d = q
        t = 1.0
        fcur = f(x)
        gd = np.dot(g, d)
        while f(x + t * d) > fcur + 1e-4 * t * gd and t > 1e-14:
            t *= 0.5
        x_new = x + t * d
        g_new = grad(x_new)
        s = x_new - x; y = g_new - g
        if np.dot(s, y) > 1e-12:
            S.append(s); Y.append(y); RHO.append(1.0 / np.dot(y, s))
            if len(S) > m:
                S.pop(0); Y.pop(0); RHO.pop(0)
        x, g = x_new, g_new
        hist.append(float(f(x)))
    return x, len(hist)


def cn2_hvp(f, grad, x0, tau, K0=10, L=None, newton_steps=2, max_cycle=40,
            eps=1e-8, tau_g=None, cg_max=30, verbose=False):
    x = np.asarray(x0, dtype=float).copy()
    x_prev = x.copy()
    cnt_f = cnt_g = hv = 0
    n = x.size
    K_ck = max(20, int(n / 5))

    def cg_dir(xx, g):
        nonlocal hv
        A = HvpOperator(grad, xx, eps=eps)
        d, it = cg_solve(A, -g, tol=1e-6, max_iter=cg_max)
        hv += it
        return d

    def entry(xx):
        nonlocal hv
        g = grad(xx); cntg = 1
        A = HvpOperator(grad, xx, eps=eps)
        # SPD probe over several directions (weak single-Rayleigh is unreliable)
        rng = np.random.default_rng(0)
        v0 = g / max(np.linalg.norm(g), 1e-12)
        probes = [v0] + [rng.standard_normal(n) for _ in range(5)]
        curv = [np.dot(u / max(np.linalg.norm(u), 1e-12), A(u)) for u in probes]
        curv_scale = max(max(abs(c) for c in curv), 1e-12)
        if all(c > 1e-6 * curv_scale for c in curv):
            # Limited Lanczos for negative-curvature detection
            lambda_min, converged, _, _ = lanczos_min_eig(A, n, k=25, tol=1e-8)
            # Use a SMALL absolute tolerance for negative curvature (not scaled by max curv)
            eta = 1e-2  # reject if min eigenvalue < -0.01
            if lambda_min > -eta:
                lam, conv = newton_decrement_cg(g, A, tol=1e-6)
                if conv:
                    return lam, "lambda", cntg
        return float(np.linalg.norm(g)), "gradnorm", cntg

    for cycle in range(max_cycle):
        for _ in range(newton_steps):
            g = grad(x); cnt_g += 1
            d = cg_dir(x, g)
            t = 1.0
            f0 = f(x); cnt_f += 1
            gd = np.dot(g, d)
            if gd >= 0:
                break
            while f(x + t * d) > f0 + 1e-4 * t * gd and t > 1e-16:
                t *= 0.5
                cnt_f += 1
            x = x + t * d
            cnt_f += 1

        val, metric, cg = entry(x)
        cnt_g += cg
        thr = tau if metric == "lambda" else (tau_g or tau)
        if verbose:
            sprint(f"  [cycle {cycle}] {metric}={val:.2e}")
        if val <= thr:
            break

        num_steps = 0
        while True:
            beta = 0.9 if num_steps > 0 else 0.0
            for _ in range(K_ck):
                y = x + beta * (x - x_prev)
                gy = grad(y); cnt_g += 1
                if L is None:
                    a = 1.0
                    f_y = f(y); cnt_f += 1
                    gnorm2 = np.dot(gy, gy)
                    while a > 1e-14:
                        x_t = y - a * gy
                        f_t = f(x_t); cnt_f += 1
                        if f_t <= f_y - 0.5 * a * gnorm2:
                            break
                        a *= 0.5
                    x_new = y - a * gy
                else:
                    a = 1.0 / L
                    x_new = y - a * gy
                x_prev = x.copy()
                x = x_new
                cnt_f += 1
                num_steps += 1

            val, metric, cg = entry(x)
            cnt_g += cg
            thr = tau if metric == "lambda" else (tau_g or tau)
            if val <= thr:
                break
            if num_steps > 50000:
                break

    return x, {"iters": num_steps + cycle * (newton_steps + 1),
               "hess_evals": 0, "hv_evals": hv, "cnt_f": cnt_f, "cnt_g": cnt_g}


def main():
    out = {}
    sprint("=" * 78)
    sprint("LARGE-n HESSIAN-FREE CN2 (n = 500 .. 1000)")
    sprint("=" * 78)

    for name, prob, x0, tau, L in [
        ("Rosen n=500", Rosenbrock(500), -1.2 * np.ones(500), 1e-5, None),
        ("Rosen n=1000", Rosenbrock(1000), -1.2 * np.ones(1000), 1e-5, None),
        ("Quad n=500", QuadIllCond(500, 1e4), 2.0 * np.ones(500), 1e-8, 1e4),
        ("Quad n=1000", QuadIllCond(1000, 1e4), 2.0 * np.ones(1000), 1e-8, 1e4),
    ]:
        sprint(f"\n--- {name}  (d={prob.n}) ---")
        row = {"problem": name}

        t0 = time.perf_counter()
        x, info = cn2_hvp(prob.f, prob.grad, x0, tau=tau, L=L, tau_g=1e-4,
                          newton_steps=2, max_cycle=40, verbose=True)
        dt = time.perf_counter() - t0
        ff = prob.f(x)
        sprint(f"  CN2-HVP : f={ff:.3e}  grad={info['cnt_g']:>8}  "
               f"cg={info['hv_evals']:>8}  wall={dt:6.2f}s")
        row["cn2_f"] = ff; row["cn2_grad"] = info["cnt_g"]
        row["cn2_hv"] = info["hv_evals"]; row["cn2_wall"] = dt

        t0 = time.perf_counter()
        x, hv = newton_cg_hvp(prob.grad, x0, tau_tol=tau, cg_max=30)
        dt = time.perf_counter() - t0
        ff = prob.f(x)
        sprint(f"  NtCG    : f={ff:.3e}  cg={hv:>8}  wall={dt:6.2f}s")
        row["ntcg_f"] = ff; row["ntcg_hv"] = hv; row["ntcg_wall"] = dt

        t0 = time.perf_counter()
        x, nit = lbfgs_hvp(prob.grad, prob.f, x0, max_iter=2000, tol=tau)
        dt = time.perf_counter() - t0
        ff = prob.f(x)
        sprint(f"  L-BFGS  : f={ff:.3e}  it={nit:>6}  wall={dt:6.2f}s")
        row["lbfgs_f"] = ff; row["lbfgs_it"] = nit; row["lbfgs_wall"] = dt

        out[name] = row

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/largeN.json", "w"), indent=2)
    sprint("\nSaved -> results/largeN.json")


if __name__ == "__main__":
    main()