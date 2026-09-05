"""
CN² — Checkpointed Newton-Nesterov.

Core building blocks for the stage-wise hybrid optimization method.

Manual (from-scratch) implementation using only NumPy for dense linear
algebra. Tracks precise evaluation counts (f, grad, Hessian) so that fair
benchmarking against baselines is possible.

Provides:
  - nesterov_agd    : Nesterov Accelerated Gradient (momentum variants)
  - newton          : damped Newton with eigenvalue-log shift
  - newton_decrement: lambda = sqrt(g^T (H+shift I)^{-1} g)
  - cn2             : Checkpointed Newton-Nesterov (stage-wise switching)

Each method returns (x, info) where info is a dict with:
  f_hist, grad_evals, f_evals, hess_evals, iters
"""

from __future__ import annotations

import numpy as np


def _fdf(f, grad, x):
    """Guard against None gradient (falls back to finite differences)."""
    if grad is not None:
        return grad(x)
    return grad_fd(f, x)


def grad_fd(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros(x.size)
    for i in range(x.size):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def hess_fd(f, x, h=1e-4):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
    return H


# ----------------------------------------------------------------------
# Newton decrement
# ----------------------------------------------------------------------
def newton_decrement(g, H, shift=1e-8):
    g = np.asarray(g, dtype=float)
    A = H + shift * np.eye(g.size)
    p = np.linalg.solve(A, g)
    return float(np.sqrt(max(np.dot(g, p), 0.0)))


def hessian_psd(H, tol=1e-10, shift=1e-8):
    """Return True if H + shift*I is numerically positive definite.

    Cheap SPD test via a Cholesky factorization (O(n^3/3)) instead of a full
    eigendecomposition (O(n^3) with a ~10x larger constant). Cholesky succeeds
    iff all leading principal minors are positive, i.e. the matrix is SPD.
    """
    try:
        np.linalg.cholesky((H + H.T) / 2.0 + shift * np.eye(H.shape[0]))
        return True
    except (np.linalg.LinAlgError, ValueError):
        return False


def entry_measure(g, H, tau_g=None, shift=1e-8):
    """
    Dual-gate entry measure (option c):
      - If H is SPD:   lambda = sqrt(g^T H^{-1} g)  (Newton decrement).
      - If H is not SPD (nonconvex region): use ||g|| as the gate instead.

    Returns (value, metric, psd):
      value  : the scalar entry measure (labelled by metric).
      metric : "lambda" or "gradnorm".
      psd    : whether H was SPD.
    """
    g = np.asarray(g, dtype=float)
    psd = hessian_psd(H)
    if psd:
        try:
            lam = newton_decrement(g, H, shift=shift)
        except np.linalg.LinAlgError:
            lam = None
        if lam is not None and np.isfinite(lam) and lam >= 0:
            return lam, "lambda", True
    # fallback: gradient norm gate
    return float(np.linalg.norm(g)), "gradnorm", False


def _entry_threshold(metric, tau, tau_g):
    """Map the entry criterion to the right scalar threshold."""
    if metric == "lambda":
        return tau
    return tau_g if tau_g is not None else tau


# ----------------------------------------------------------------------
# Nesterov Accelerated Gradient
# ----------------------------------------------------------------------
def nesterov_agd(f, grad, x0, L=None, alpha=None, max_iter=1000, tol=1e-10,
                 momentum="nag"):
    """
    Nesterov-type accelerated gradient.

    momentum:
      "nag"   : Nesterov AGD, constant beta=0.9 after first step
      "fista" : FISTA schedule beta_k=(t_k-1)/t_{k+1}
      "const" : constant beta=0.9 (heavy-ball style)

    Step size: uses alpha if given; else backtracking line search (safe).
    """
    x = np.asarray(x0, dtype=float).copy()
    x_prev = x.copy()
    hist = []
    counter = Counter(0, 0, 0)

    for k in range(max_iter):
        # momentum coefficient
        if momentum == "fista":
            if k == 0:
                t_prev = 1.0; beta_k = 0.0
            else:
                t_new = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2
                beta_k = (t_prev - 1) / t_new
                t_prev = t_new
        else:  # nag / const
            beta_k = 0.9 if k > 0 else 0.0

        y = x + beta_k * (x - x_prev)
        gy = grad(y); counter.g += 1

        if alpha is None:
            a = 1.0 / L if L is not None else 1.0
            f_y = f(y); counter.f += 1
            gnorm2 = np.dot(gy, gy)
            while a > 1e-14:
                x_new = y - a * gy
                f_new = f(x_new); counter.f += 1
                if f_new <= f_y - 0.5 * a * gnorm2:
                    break
                a *= 0.5
            alpha_now = a
        else:
            alpha_now = alpha
            x_new = y - alpha * gy

        x_prev = x.copy()
        x = x_new
        counter.f += 1
        hist.append(float(f(x)))

        gx = grad(x); counter.g += 1
        if np.linalg.norm(gx) < tol:
            break

    return x, {"f_hist": hist, "grad_evals": counter.g,
               "f_evals": counter.f, "hess_evals": 0, "iters": len(hist)}


class Counter:
    __slots__ = ("f", "g", "h")
    def __init__(self, f=0, g=0, h=0):
        self.f, self.g, self.h = f, g, h


# ----------------------------------------------------------------------
# Newton's method
# ----------------------------------------------------------------------
def _solve(H, g, shift=1e-8):
    A = H + shift * np.eye(H.shape[0])
    return np.linalg.solve(A, -g)


def newton(f, grad, hess, x0, tau_tol=1e-8, max_iter=200, shift=1e-8,
           damping=True):
    x = np.asarray(x0, dtype=float).copy()
    hist = []
    counter = Counter(0, 0, 0)

    for _ in range(max_iter):
        g = grad(x); counter.g += 1
        H = hess(x) if hess is not None else hess_fd(f, x)
        counter.h += 1
        lam = newton_decrement(g, H, shift=shift)
        if lam <= tau_tol:
            break
        d = _solve(H, g, shift=shift)
        if damping:
            t = 1.0
            f0 = f(x); counter.f += 1
            gd = np.dot(g, d)
            while f(x + t * d) > f0 + 1e-4 * t * gd and t > 1e-14:
                t *= 0.5
                counter.f += 1
            x = x + t * d
        else:
            x = x + d
        counter.f += 1
        hist.append(float(f(x)))

    return x, {"f_hist": hist, "grad_evals": counter.g,
               "f_evals": counter.f, "hess_evals": counter.h,
               "iters": len(hist)}




# ----------------------------------------------------------------------
# CN² v4 : Newton(2) -> evaluate -> Nesterov-until-entry -> loop
# ----------------------------------------------------------------------
def cn2(f, grad, hess, x0, tau, K0=20, L=None, alpha=None,
        newton_steps=2, max_cycle=200, shift=1e-8, tau_g=None, momentum="nag",
        verbose=False):
    """
    CN² final design (v4).

    Loop:
      1. Take `newton_steps` Newton steps (fresh Hessian each).
      2. Evaluate the Newton decrement lambda at the current point.
      3. If lambda <= tau -> we are in the solution space -> STOP.
      4. Else (far) -> run Nesterov until lambda drops to <= tau,
         checking lambda only rarely (every K_ck steps, K_ck grows with
         dimension and is floored/tuned by K0), using a fresh Hessian at
         each re-check.
      5. Loop back to Newton from the new position.

    Parameters
    ----------
    K0 : low-dimension probe cadence; the effective check cadence is
        K_ck = max(K0, n/5). Increasing K0 re-checks the entry measure less
        often (fewer Hessians, longer Nesterov bursts); decreasing it below
        the n/5 floor has no effect.
    momentum : "nag" (constant beta=0.9 lookahead momentum, the implemented
        default and the variant measured in the paper) or "fista" (the
        theoretically-optimal beta_k=(t_{k-1}-1)/t_k schedule, giving the
        NAG rate rho = 1 - sqrt(mu/L) asserted in Lemma 3).

    Exactly matches the agreed spec:
      Q1: Nesterov stops when the Newton decrement itself hits tau.
      Q2: after Nesterov reaches the solution space we STOP (no extra Newton).
      Q3: newton_steps = 2 per cycle.
    """
    x = np.asarray(x0, dtype=float).copy()
    x_prev = x.copy()
    f_hist = []
    counter = Counter(0, 0, 0)

    def _hess(xx):
        nonlocal counter
        counter.h += 1
        return hess(xx) if hess is not None else hess_fd(f, xx)

    def _lam(xx, H):
        g = grad(xx); counter.g += 1
        return entry_measure(g, H, tau_g=tau_g, shift=shift)

    # check cadence: grows with dimension (larger n -> compute Hessian less
    # often, per the agreed "inversely proportional" policy); K0 is the
    # user-provided low-dimension floor for this cadence
    n = x.size
    K_ck = max(int(K0), int(n / 5))

    for cycle in range(max_cycle):
        # ---- 1) Newton: `newton_steps` steps ----
        for _ in range(newton_steps):
            g = grad(x); counter.g += 1
            H = _hess(x)
            d = _solve(H, g, shift=shift)
            t = 1.0
            f0 = f(x); counter.f += 1
            gd = np.dot(g, d)
            if gd >= 0:
                break
            while f(x + t * d) > f0 + 1e-4 * t * gd and t > 1e-16:
                t *= 0.5
                counter.f += 1
            x = x + t * d
            counter.f += 1
            f_hist.append(float(f(x)))

        if verbose:
            print(f"[cycle {cycle}] after Newton f={f(x):.3e}")

        # ---- 2) evaluate entry measure ----
        H = _hess(x)
        val, metric, psd = _lam(x, H)
        thr = _entry_threshold(metric, tau, tau_g)
        f_hist.append(float(f(x))); counter.f += 1
        if verbose:
            print(f"  evaluate {metric}={val:.3e} thr={thr:.3e} psd={psd}")

        # ---- 3) in solution space? ---- 
        if val <= thr:
            if verbose:
                print("  -> in solution space; converged")
            break

        # ---- 4) far: Nesterov until entry ----
        num_steps = 0
        t_fista = 1.0
        while True:
            # Nesterov K_ck steps
            for _ in range(K_ck):
                if momentum == "fista":
                    t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_fista ** 2)) / 2.0
                    beta_k = (t_fista - 1.0) / t_next
                    t_fista = t_next
                else:  # "nag" (constant lookahead momentum, paper default)
                    beta_k = 0.9 if num_steps > 0 else 0.0
                y = x + beta_k * (x - x_prev)
                gy = grad(y); counter.g += 1
                if alpha is None:
                    a = 1.0 / L if L is not None else 1.0
                    f_y = f(y); counter.f += 1
                    gnorm2 = np.dot(gy, gy)
                    while a > 1e-14:
                        x_new = y - a * gy
                        f_new = f(x_new); counter.f += 1
                        if f_new <= f_y - 0.5 * a * gnorm2:
                            break
                        a *= 0.5
                else:
                    a = alpha
                    x_new = y - a * gy
                x_prev = x.copy()
                x = x_new
                f_hist.append(float(f(x))); counter.f += 1
                num_steps += 1

            # re-check entry measure (fresh Hessian each check here)
            H = _hess(x)
            val, metric, psd = _lam(x, H)
            thr = _entry_threshold(metric, tau, tau_g)
            f_hist.append(float(f(x))); counter.f += 1
            if verbose:
                print(f"    nesterov block: {metric}={val:.3e}")
            if val <= thr:
                break
            if num_steps > 200000:
                break

        # ---- 5) loop back to Newton ----

    return x, {
        "f_hist": f_hist,
        "grad_evals": counter.g,
        "f_evals": counter.f,
        "hess_evals": counter.h,
        "iters": len(f_hist),
        "k_ck": K_ck,
    }

# ----------------------------------------------------------------------
# Convenience: total cost model
# ----------------------------------------------------------------------
class CostModel:
    """Abstract cost of gradient and Hessian evaluation for fair comparison.

    For dense full Hessian with solve:
        grad_cost  : cost of one gradient = O(n)  (set 1)
        hess_cost  : cost of hessian + solve = O(n^3) dominant
    walltime ~ grad_evals*grad_cost + hess_evals*hess_cost
    """

    def __init__(self, n, hess_mult=None):
        self.n = n
        # default: full-dense Hessian+factorize cost proportional to n^3
        if hess_mult is None:
            hess_mult = max(1.0, n ** 3 / max(n, 1.0) / max(n, 1.0))
        self.hess_cost = hess_mult
        self.grad_cost = 1.0

    def walltime(self, info):
        return (info["grad_evals"] * self.grad_cost
                + info["hess_evals"] * self.hess_cost)


def walltime(n, info):
    # Cost model: one Hessian build + dense solve costs ~O(n^2) flops vs
    # ~O(n) for a gradient; ratio n. One documented line, no dead assignment.
    hess_cost = max(1.0, float(n))
    return info["grad_evals"] * 1.0 + info["hess_evals"] * hess_cost
