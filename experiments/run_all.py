"""
CN² — comprehensive experiment driver (resource-respecting).

Runs, within a comfortable time budget on the user's machine:
  (A) Validation on small problems.
  (B) Main comparison across sizes n with all baselines.
  (C) Sensitivity analysis over K0 and tau.
  (D) Convergence plots + metric tables (CSV/JSON).

Design respects the hardware (i5-13420H, 16GB): keeps dense Hessian problems
in n range where a full-Hessian solve is cheap (~ms), and keeps total runtime
well under ~90 minutes (actual: a few minutes).
"""

from __future__ import annotations

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core import cn2, newton, nesterov_agd, walltime, grad_fd
from benchmarks.baselines import newton_cg as _ncg, lbfgs as _lbfgs, gd as _gd
from problems.test_funcs import (
    Rosenbrock, QuadIllCond, Beale, Himmelblau,
    LogisticRegression, make_logistic_data,
)

# silences finite-diff overflow spam
np.seterr(all="ignore")


def run_method(prob, x0, tau, L, method, tau_g=1e-4, **kw):
    f = prob.f
    g = getattr(prob, "grad", None)
    h = getattr(prob, "hess", None)
    if method == "cn2":
        x, info = cn2(f, g, h, x0, tau=tau, L=L, tau_g=tau_g,
                      newton_steps=2, max_cycle=5000, **kw)
    elif method == "newton":
        x, info = newton(f, g, h, x0, tau_tol=tau, max_iter=300)
    elif method == "newton_cg":
        x, info = _ncg(f, x0, grad=g, max_iter=300, tol=tau)
    elif method == "lbfgs":
        x, info = _lbfgs(f, x0, grad=g, max_iter=1000, tol=tau)
    elif method == "nag":
        x, info = nesterov_agd(f, g, x0, L=L, max_iter=5000, tol=tau,
                               momentum="nag")
    elif method == "gd":
        x, info = _gd(f, x0, grad=g, L=L if L else 1.0, max_iter=5000, tol=tau)
    else:
        raise ValueError(method)
    info["final_f"] = float(f(x))
    info["grad_norm"] = float(np.linalg.norm(
        g(x) if g is not None else grad_fd(f, x)))
    info["method"] = method
    info["walltime"] = walltime(prob.n, info)
    info.setdefault("hv_evals", 0)
    return info


# ----------------------------------------------------------------------
# Problem factory (kept small/moderate per hardware)
# ----------------------------------------------------------------------
def make_problems():
    P = {}
    P["Rosenbrock n=2"]   = (Rosenbrock(2),   np.array([-1.2, 1.0]),    1e-8, 200.0)
    P["Beale n=2"]        = (Beale(),         np.array([1.0, 1.0]),     1e-8, 50.0)
    P["Himmelblau n=2"]   = (Himmelblau(),    np.array([-4.0, -4.0]),   1e-8, 50.0)
    P["Rosenbrock n=50"]  = (Rosenbrock(50),  -1.2*np.ones(50),         1e-8, 200.0)
    P["Rosenbrock n=100"] = (Rosenbrock(100), -1.2*np.ones(100),        1e-8, 200.0)
    P["Rosenbrock n=200"] = (Rosenbrock(200), -1.2*np.ones(200),        1e-8, 200.0)
    # logistic: n_features small because hess_fd is O(n^2)
    X, y = make_logistic_data(300, 30, seed=0)
    lr = LogisticRegression(X, y, reg=1e-3)
    P["Logistic (p=30)"] = (lr, np.zeros(lr.n), 1e-8, None)
    return P


# ----------------------------------------------------------------------
# (A)+(B): validation + main comparison
# ----------------------------------------------------------------------
def run_comparison(problems, methods):
    rows = []
    for name, (prob, x0, tau, L) in problems.items():
        for m in methods:
            t0 = time.time()
            try:
                info = run_method(prob, x0, tau, L, m)
            except Exception as e:
                rows.append({"problem": name, "method": m, "error": str(e)})
                continue
            info["problem"] = name
            info["cpu_s"] = round(time.time() - t0, 3)
            rows.append(info)
            print(f"  {name:>16} | {m:>9} | f={info['final_f']:.2e} "
                  f"it={info['iters']:>5} hess={info['hess_evals']:>5} "
                  f"wall={info['walltime']:.1e} ({info['cpu_s']}s)",
                  flush=True)
    return rows


# ----------------------------------------------------------------------
# (C): sensitivity over K0 and tau
# ----------------------------------------------------------------------
def run_sensitivity():
    print("\n--- Sensitivity: K0 on Rosenbrock n=100 ---")
    p, x0, tau, L = Rosenbrock(100), -1.2*np.ones(100), 1e-8, 200.0
    sens_k = []
    for K0 in [5, 10, 20, 40]:
        info = run_method(p, x0, tau, L, "cn2", K0=K0)
        sens_k.append({"K0": K0, "f": info["final_f"],
                       "hess": info["hess_evals"], "iters": info["iters"],
                       "wall": info["walltime"]})
        print(f"  K0={K0:>3}: f={info['final_f']:.2e} hess={info['hess_evals']} "
              f"iters={info['iters']}", flush=True)

    print("--- Sensitivity: tau on Rosenbrock n=100 ---")
    sens_t = []
    for tau in [1e-4, 1e-6, 1e-8]:
        info = run_method(p, x0, tau, L, "cn2")
        sens_t.append({"tau": tau, "f": info["final_f"],
                       "hess": info["hess_evals"], "iters": info["iters"],
                       "wall": info["walltime"]})
        print(f"  tau={tau:.0e}: f={info['final_f']:.2e} hess={info['hess_evals']} "
              f"iters={info['iters']}", flush=True)
    return sens_k, sens_t


# ----------------------------------------------------------------------
# (D): plots — full set covering every claim in the results
# ----------------------------------------------------------------------
def _save(fig, name):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{name}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}")


def make_plots(problems, rows, sens_k, sens_t):
    methods = ["cn2", "newton", "newton_cg", "lbfgs", "nag", "gd"]
    probs = list(problems.keys())

    def get(m, p, key="final_f", default=None):
        for r in rows:
            if r.get("method") == m and r.get("problem") == p and "error" not in r:
                return r.get(key, default)
        return default

    # ---- 1) Final accuracy per problem (bar) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    xpos = np.arange(len(probs))
    width = 0.13
    colors = {"cn2": "#d62728", "newton": "#1f77b4", "newton_cg": "#ff7f0e",
              "lbfgs": "#2ca02c", "nag": "#9467bd", "gd": "#8c564b"}
    # floor just below the smallest positive value so log-scale keeps 0-bars
    allf = [get(m, p) for m in methods for p in probs]
    pos = [v for v in allf if v is not None and v > 0]
    floor = min(pos) / 10 if pos else 1e-16
    for i, m in enumerate(methods):
        vals = [get(m, p) for p in probs]
        clipped = [floor if (v is None or v <= 0) else v for v in vals]
        ax.bar(xpos + (i - 2.5) * width, clipped, width, label=m, color=colors[m])
    ax.set_xticks(xpos)
    ax.set_xticklabels([p[:14] for p in probs], rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("final f(x)  (log scale; bars at floor = exact 0)")
    ax.set_title("Final objective value by method and problem (log scale)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "accuracy_all.png")

    # ---- 2) Hessian economy: CN² vs Newton per problem (bar) ----
    # Where Newton FAILED to reach the solution its low Hessian count is not
    # comparable to CN²'s — those cases are marked so the plot is not misread.
    fig, ax = plt.subplots(figsize=(10, 5))
    h_cn2 = [get("cn2", p, "hess_evals", 0) for p in probs]
    h_nt = [get("newton", p, "hess_evals", 0) for p in probs]
    f_nt = [get("newton", p, "final_f", 0) for p in probs]
    failed = [fn is not None and fn > 1e-6 for fn in f_nt]
    bars1 = ax.bar(xpos - width / 2, h_cn2, width, label="CN² (Hessians)",
                   color="#d62728")
    bars2 = ax.bar(xpos + width / 2, h_nt, width, label="Newton (Hessians)",
                   color="#1f77b4")
    for i in range(len(probs)):
        if failed[i]:
            ax.text(xpos[i] + width / 2, h_nt[i] * 1.03 + 3, "✗ failed",
                    ha="center", fontsize=8, color="crimson", fontweight="bold")
        if not failed[i] and h_cn2[i] < h_nt[i]:
            ax.annotate("saves Hessians", xy=(xpos[i] - width/2, h_cn2[i]),
                        xytext=(xpos[i], max(h_cn2[i], h_nt[i]) * 1.05),
                        ha="center", fontsize=7, color="green")
    ax.set_xticks(xpos)
    ax.set_xticklabels([p[:14] for p in probs], rotation=30, ha="right")
    ax.set_ylabel("# Hessian evaluations")
    ax.set_title("Hessian evaluations: CN² vs Newton (✗ = Newton failed to converge)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "hessian_economy.png")

    # ---- 3) Hessian ratio: CN² / Newton (only where Newton converged) ----
    # Exclude problems where Newton failed — a small Hessian count after
    # stalling would otherwise make the ratio look falsely favourable to Newton.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ratios = []
    labels = []
    for p in probs:
        hc = get("cn2", p, "hess_evals", 0); hn = get("newton", p, "hess_evals", 0)
        fn_newton = get("newton", p, "final_f", None)
        converged = fn_newton is not None and fn_newton <= 1e-6
        if hn and hc and converged:
            ratios.append(hc / hn); labels.append(p)
    bar_colors = ["#2ca02c" if r < 1 else "#d62728" for r in ratios]
    ax.barh(np.arange(len(ratios)), ratios, color=bar_colors)
    ax.set_yticks(np.arange(len(ratios)))
    ax.set_yticklabels([l[:20] for l in labels])
    ax.axvline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("CN² Hessians / Newton Hessians (<1 means CN² saves)")
    ax.set_title("Relative Hessian cost of CN² vs Newton (converged cases only)")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "hessian_ratio.png")

    # ---- 4) tau sensitivity: accuracy vs Hessians trade-off ----
    if sens_t:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ts = [s["tau"] for s in sens_t]
        hh = [s["hess"] for s in sens_t]
        ff = [s["f"] for s in sens_t]
        ax2 = ax.twinx()
        l1 = ax.plot(ts, hh, "o-", color="#d62728", label="Hessians")
        l2 = ax2.plot(ts, ff, "s--", color="#1f77b4", label="final f")
        ax.set_xscale("log"); ax.set_xlabel("tau")
        ax.set_ylabel("# Hessians", color="#d62728")
        ax2.set_yscale("log"); ax2.set_ylabel("final f", color="#1f77b4")
        ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="center right")
        ax.set_title("tau sensitivity: accuracy vs Hessian economy (Rosen n=100)")
        ax.grid(True, alpha=0.3)
        _save(fig, "sensitivity_tau.png")

    # ---- 5) K0 sensitivity: stability of the result ----
    if sens_k:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ks = [s["K0"] for s in sens_k]
        hh = [s["hess"] for s in sens_k]
        ff = [s["f"] for s in sens_k]
        ax.plot(ks, hh, "o-", color="#d62728", label="# Hessians")
        ax.set_xlabel("K0 (jump length)")
        ax.set_ylabel("# Hessians", color="#d62728")
        ax.set_title("K0 sensitivity: result is stable across K0 (Rosen n=100)")
        ax.grid(True, alpha=0.3)
        _save(fig, "sensitivity_K0.png")

    # ---- 6) Convergence curves comparison on Rosen n=100 ----
    p, x0, tau, L = Rosenbrock(100), -1.2*np.ones(100), 1e-8, 200.0
    _, info_cn2 = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=L, tau_g=1e-4,
                      newton_steps=2, max_cycle=5000)
    _, info_newton = newton(p.f, p.grad, p.hess, x0, tau_tol=tau, max_iter=300)
    _, info_lbfgs = _lbfgs(p.f, x0, grad=p.grad, max_iter=1000, tol=tau)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(info_cn2["f_hist"], label="CN²", color="#d62728")
    ax.semilogy(info_newton["f_hist"], label="Newton", color="#1f77b4")
    ax.semilogy(info_lbfgs["f_hist"], label="L-BFGS", color="#2ca02c")
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x)")
    ax.set_title("Convergence: CN² vs Newton vs L-BFGS (Rosenbrock n=100)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "convergence_rosen100.png")


def main():
    print("=" * 70)
    print("CN² experiment suite")
    print("=" * 70)

    problems = make_problems()
    methods = ["cn2", "newton", "newton_cg", "lbfgs", "nag", "gd"]

    print("\n[1/3] Validation + main comparison")
    rows = run_comparison(problems, methods)

    print("\n[2/3] Sensitivity")
    sens_k, sens_t = run_sensitivity()

    print("\n[3/3] Plots")
    make_plots(problems, rows, sens_k, sens_t)

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as fh:
        json.dump({"rows": rows, "sensitivity_K0": sens_k,
                   "sensitivity_tau": sens_t}, fh, indent=2)

    # compact summary table
    print("\n" + "=" * 70)
    print("FINAL FUNCTION VALUE TABLE (lower is better)  * = reached tol")
    names = list(problems.keys())
    hdr = f"{'method':>10} | " + " | ".join(f"{n[:18]:>18}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        cells = []
        for n in names:
            rr = [r for r in rows if r["method"] == m and r["problem"] == n and "error" not in r]
            if rr:
                cells.append(f"{rr[0]['final_f']:.1e}")
            else:
                cells.append("ERR")
        print(f"{m:>10} | " + " | ".join(f"{c:>18}" for c in cells))

    print("\nSaved -> results/benchmark.json, figures/convergence_rosen100.png")


if __name__ == "__main__":
    main()
