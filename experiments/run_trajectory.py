"""
E7. Trajectory figure: Newton decrement / gradient norm vs cycle index,
colored by phase (L = Nesterov-dominated linear, S = Newton-dominated
superlinear), on Rosenbrock n=100.

Marker color: red = lambda-gate (SPD, convex near-field => linear phase),
blue = gradnorm-gate (nonconvex far-field => superlinear transit).
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core import entry_measure, _solve
from src.core import Counter
from problems.test_funcs import Rosenbrock, Beale


def trace(f, grad, hess, x0, tau, K0=20, L=None, alpha=None, newton_steps=2,
          max_cycle=200, shift=1e-8, tau_g=None):
    """Replay of the CN2 loop recording (denominator, metric, value, psd)."""
    x = np.asarray(x0, dtype=float).copy()
    x_prev = x.copy()
    n = x.size
    K_ck = max(int(K0), int(n / 5))
    rec = []

    def _hess(xx):
        return hess(xx)

    for cycle in range(max_cycle):
        for _ in range(newton_steps):
            g = grad(x)
            H = _hess(x)
            d = _solve(H, g, shift=shift)
            t = 1.0
            f0 = f(x); gd = np.dot(g, d)
            if gd >= 0:
                break
            while f(x + t * d) > f0 + 1e-4 * t * gd and t > 1e-16:
                t *= 0.5
            x = x + t * d

        H = _hess(x)
        val, metric, psd = entry_measure(grad(x), H, tau_g=tau_g, shift=shift)
        thr = tau if metric == "lambda" else (tau_g or tau)
        rec.append((cycle, metric, val, psd))
        if val <= thr:
            break

        num_steps = 0
        while True:
            beta = 0.9 if num_steps > 0 else 0.0
            for _ in range(K_ck):
                y = x + beta * (x - x_prev)
                gy = grad(y)
                if alpha is None:
                    a = 1.0 / L if L is not None else 1.0
                    f_y = f(y); gnorm2 = np.dot(gy, gy)
                    while a > 1e-14:
                        x_new = y - a * gy
                        if f(x_new) <= f_y - 0.5 * a * gnorm2:
                            break
                        a *= 0.5
                else:
                    a = alpha
                    x_new = y - a * gy
                x_prev = x.copy()
                x = x_new
                num_steps += 1
            val, metric, psd = entry_measure(grad(x), _hess(x), tau_g=tau_g,
                                             shift=shift)
            thr = tau if metric == "lambda" else (tau_g or tau)
            if val <= thr:
                rec.append((cycle + 0.5, metric, val, psd))
                break
            if num_steps > 200000:
                break
    return rec


def main():
    cases = [
        ("Beale (nonconvex start)", "figures/trajectory_two_phase.png",
         Beale(), np.array([1.0, 1.0]), 1e-8, 50.0, 1e-4),
        ("Rosenbrock n=100", "figures/trajectory_two_phase_rosen100.png",
         Rosenbrock(100), -1.2 * np.ones(100), 1e-8, 200.0, 1e-4),
    ]
    for label, fname, p, x0, tau, L, tau_g in cases:
        rec = trace(p.f, p.grad, p.hess, x0, tau=tau, K0=10, L=L,
                    tau_g=tau_g, newton_steps=2, max_cycle=200)
        xs = [r[0] for r in rec]
        vals = [r[2] for r in rec]
        colors = ["#d62728" if r[1] == "lambda" else "#1f77b4" for r in rec]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.scatter(xs, vals, c=colors, s=44, zorder=3, edgecolors="none")
        ax.plot(xs, vals, color="#555555", lw=0.8, zorder=2, alpha=0.6)
        ax.set_yscale("log")
        # threshold lines: tau (lambda gate) and tau_g (gradnorm gate)
        ax.axhline(tau, color="#2ca02c", ls="--", lw=1,
                   label=r"$\tau=10^{-8}$ ($\lambda$-gate)")
        ax.axhline(tau_g, color="#ff7f0e", ls=":", lw=1,
                   label=r"$\tau_g=10^{-4}$ ($\|g\|$-gate)")
        ax.set_xlabel("cycle index  $j$")
        ax.set_ylabel(r"entry measure  $\delta=\lambda(x)\ \mathrm{or}\ \|g\|$")
        ax.set_title(f"E7: two-phase decay along checkpoints — {label}")
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   markersize=8, label=r"$\lambda$-gate (convex / linear $p\approx1$)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
                   markersize=8, label=r"$\|g\|$-gate (nonconvex / transit)"),
            Line2D([0], [0], color="#2ca02c", ls="--", lw=1, label=r"$\tau$"),
            Line2D([0], [0], color="#ff7f0e", ls=":", lw=1, label=r"$\tau_g$"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        fig.savefig(fname, dpi=140)
        plt.close(fig)
        lam_vals = [r[2] for r in rec if r[1] != "gradnorm"]
        gnorm_vals = [r[2] for r in rec if r[1] == "gradnorm"]
        print(f"{label}: checkpoints={len(rec)}  lambda-gate={len(lam_vals)}  "
              f"gradnorm-gate={len(gnorm_vals)}")
        print(f"saved {fname}")


if __name__ == "__main__":
    main()