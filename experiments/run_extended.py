"""
Extended experiments for the updated papers:

  E1. Conditioning/dimension sweep on QuadIllCond:
      kappa in {1e2, 1e4, 1e6} x n in {10, 50, 100, 500}
      CN2 vs Newton: final f, Hessian evaluations.
  E6. newton_steps ablation on Rosen n=100, Beale, Himmelblau:
      newton_steps in {1, 2, 3}: final f, Hessian evals, iters.

Outputs results/extended.json, figures/extended_heatmaps.png clean.
"""
from __future__ import annotations

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core import cn2, newton
from problems.test_funcs import Rosenbrock, QuadIllCond, Beale, Himmelblau


def run_e1():
    print("=" * 70)
    print("E1: conditioning x dimension sweep (QuadIllCond)")
    print("=" * 70)
    grid = {}
    for n in (10, 50, 100, 500):
        for kappa in (1e2, 1e4, 1e6):
            p = QuadIllCond(n, kappa)
            x0 = 5.0 * np.ones(n)
            x, info = cn2(p.f, p.grad, p.hess, x0, tau=1e-8, K0=20, L=1.0,
                          tau_g=1e-4, newton_steps=2, max_cycle=200)
            xN, infoN = newton(p.f, p.grad, p.hess, x0, tau_tol=1e-8, max_iter=200)
            row = {"f": p.f(x), "hess": info["hess_evals"],
                   "fN": p.f(xN), "hessN": infoN["hess_evals"],
                   "it": info["iters"]}
            grid[f"{n}/{kappa:.0e}"] = row
            print(f"  n={n:>4} kappa={kappa:>6.0e}: CN2 f={row['f']:.1e} "
                  f"hess={row['hess']:>4} | Nwt f={row['fN']:.1e} hess={row['hessN']:>4}")
    return grid


def run_e6():
    print("=" * 70)
    print("E6: newton_steps ablation")
    print("=" * 70)
    out = {}
    cases = [
        ("Rosen n=100", Rosenbrock(100), -1.2 * np.ones(100), 1e-8, 200.0),
        ("Beale", Beale(), np.array([1.0, 1.0]), 1e-8, 50.0),
        ("Himmelblau", Himmelblau(), np.array([-4.0, -4.0]), 1e-8, 50.0),
    ]
    for name, p, x0, tau, L in cases:
        print(f"-- {name}")
        out[name] = {}
        for ns in (1, 2, 3):
            x, info = cn2(p.f, p.grad, p.hess, x0, tau=tau, K0=20, L=L,
                          tau_g=1e-4, newton_steps=ns, max_cycle=300)
            row = {"f": p.f(x), "hess": info["hess_evals"], "it": info["iters"]}
            out[name][ns] = row
            print(f"    newton_steps={ns}: f={row['f']:.2e} "
                  f"hess={row['hess']:>4} it={row['it']:>6}")
    return out


def make_plots(grid, ablat):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = sorted(set(int(k.split('/')[0]) for k in grid))
    ks = sorted(set(float(k.split('/')[1]) for k in grid),
                 key=lambda x: -float(x))
    hess = np.array([[grid[f"{n}/{kappa:.0e}"]["hess"] for n in ns]
                     for kappa in ks])
    hessN = np.array([[grid[f"{n}/{kappa:.0e}"]["hessN"] for n in ns]
                      for kappa in ks])
    im = ax[0].imshow(hess, cmap="viridis", aspect="auto")
    ax[0].set_xticks(range(len(ns))); ax[0].set_xticklabels(ns, fontsize=8)
    ax[0].set_yticks(range(len(ks))); ax[0].set_yticklabels([f"{k:.0e}" for k in ks], fontsize=8)
    ax[0].set_title("CN2 #Hessians (QuadIllCond)")
    ax[0].set_xlabel("n"); ax[0].set_ylabel("kappa")
    ax[0].set_xticks(range(len(ns)))
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    im2 = ax[1].imshow(hessN, cmap="plasma", aspect="auto")
    ax[1].set_xticks(range(len(ns))); ax[1].set_xticklabels(ns, fontsize=8)
    ax[1].set_yticks(range(len(ks))); ax[1].set_yticklabels([f"{k:.0e}" for k in ks], fontsize=8)
    ax[1].set_title("Newton #Hessians (QuadIllCond)")
    ax[1].set_xlabel("n"); ax[1].set_ylabel("kappa")
    ax[1].set_xticks(range(len(ns)))
    fig.colorbar(im2, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/extended_heatmaps.png", dpi=140)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    problems = list(ablat.keys())
    xs = np.arange(len(problems))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for j, ns in enumerate((1, 2, 3)):
        h = [ablat[p][ns]["hess"] for p in problems]
        ax2.bar(xs + (j - 1) * 0.25, h, width=0.22, color=colors[j],
                label=f"newton_steps={ns}")
    ax2.set_xticks(xs); ax2.set_xticklabels(problems, fontsize=9)
    ax2.set_ylabel("# Hessian evaluations")
    ax2.set_title("E6: newton_steps ablation (Hessian economy)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig("figures/ablation_newton_steps.png", dpi=140)
    plt.close(fig2)
    print("saved figures/extended_heatmaps.png, ablation_newton_steps.png")


def main():
    g = run_e1()
    a = run_e6()
    make_plots(g, a)
    os.makedirs("results", exist_ok=True)
    json.dump({"E1": g, "E6": a}, open("results/extended.json", "w"), indent=2)
    print("Saved -> results/extended.json")


if __name__ == "__main__":
    main()