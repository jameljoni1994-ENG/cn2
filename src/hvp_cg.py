"""
Hessian-vector-product machinery for the large-n regime of CN2.

Replaces dense Hessian formation H(x) (O(n^2) memory, O(n^3) solve) with
finite-difference products  (H(x)+eps I) v  computed in O(n) memory and
O(n) gradient calls per solve, via conjugate gradients.  The CN2 dual gate
(lambda = sqrt(g^T (H+eps I)^{-1} g)) stays available because it is exactly
one extra CG solve on (H+eps I) with right-hand side g.

Cost model (relative to 1 gradient call):
    hvp(x, v)            : 2 gradient calls + 2 vector adds   -> O(n) memory
    cg_solve(g, hvp_op)  : ~k iterations * 2 grad calls        -> O(n) memory
"""

from __future__ import annotations

import numpy as np


def hvp_fd(grad, x, v, h=1e-6):
    """(H(x) v) via central finite differences of the gradient.

    Returns (H+eps*I)v in-place of the raw product when combined upstream.
    """
    g_p = grad(x + h * v)
    g_m = grad(x - h * v)
    return (g_p - g_m) / (2 * h)


class HvpOperator:
    """Linear operator v -> (H(x)+eps I) v without ever forming H(x)."""

    def __init__(self, grad, x, eps=1e-8, h=1e-6):
        self.grad = grad
        self.x = np.asarray(x, dtype=float)
        self.eps = eps
        self.h = h

    def __call__(self, v):
        return hvp_fd(self.grad, self.x, v, self.h) + self.eps * v


def cg_solve(A, b, x0=None, tol=1e-6, max_iter=None):
    """Conjugate-gradient solve A x = b for SPD operator A (callable).
    Returns (x, iters); a non-converged solve still returns its best iterate."""
    b = np.asarray(b, dtype=float)
    n = b.size
    if max_iter is None:
        max_iter = min(n, 100)
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = b - A(x)
    if np.linalg.norm(r) < tol * max(1.0, np.linalg.norm(b)):
        return x, 0
    p = r.copy()
    rs = np.dot(r, r)
    iters = 0
    for it in range(max_iter):
        Ap = A(p)
        alpha = rs / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        iters = it + 1
        if np.sqrt(rs_new) < tol * max(1.0, np.linalg.norm(b)):
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, iters


def newton_decrement_cg(g, Aop, tol=1e-6):
    """lambda = sqrt(g^T (H+eps I)^{-1} g) using one CG solve.

    Returns (value, converged, iters): converged=False means CG hit max_iter
    without reaching the residual tolerance, so the value is NOT trustworthy
    (occurs when A is indefinite / negative curvature). iters is the number
    of HVP iterations the gate's extra CG solve consumed."""
    z, iters, converged = cg_solve_full(Aop, g, tol=tol)
    val = float(np.sqrt(max(np.dot(g, z), 0.0)))
    return val, converged, iters


def cg_solve_full(A, b, tol=1e-6, max_iter=None):
    """CG solve returning (x, iters, converged)."""
    b = np.asarray(b, dtype=float)
    n = b.size
    if max_iter is None:
        max_iter = min(n, 100)
    x = np.zeros(n)
    r = b - A(x)
    bnrm = np.linalg.norm(b)
    if bnrm and np.linalg.norm(r) < tol * bnrm:
        return x, 0, True
    p = r.copy()
    rs = np.dot(r, r)
    for it in range(max_iter):
        Ap = A(p)
        alpha = rs / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if bnrm and np.sqrt(rs_new) < tol * bnrm:
            return x, it + 1, True
        if rs_new <= 0.0:  # indefiniteness detected
            return x, it + 1, False
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, max_iter, False


def lanczos_min_eig(Aop, n, k=25, tol=1e-6, seed=0):
    """
    Limited Lanczos to estimate the minimum eigenvalue of a symmetric operator Aop.
    
    Returns (lambda_min, converged, T_diag, T_offdiag):
    - lambda_min: estimated minimum eigenvalue of Aop
    - converged: True if beta_j < tol (Krylov subspace captured invariant subspace)
    - T_diag, T_offdiag: diagonal/off-diagonal of the tridiagonal matrix T_k
    
    Cost: k HVP evaluations. For n=1000, k=25 → 25 HVPs.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v = v / max(np.linalg.norm(v), 1e-12)
    v_prev = np.zeros(n)
    beta = 0.0
    alphas = []
    betas = []
    
    for j in range(k):
        w = Aop(v)
        alpha = float(np.dot(v, w))
        alphas.append(alpha)
        
        w = w - alpha * v - beta * v_prev
        beta = float(np.linalg.norm(w))
        betas.append(beta)
        
        if beta < tol:
            # Early convergence — invariant subspace found
            T_diag = np.array(alphas)
            T_offdiag = np.array(betas[:-1])
            eigvals = _tridiag_eigvals(T_diag, T_offdiag)
            return float(np.min(eigvals)), True, T_diag, T_offdiag
        
        v_prev = v
        v = w / beta
    
    # Full k steps done
    T_diag = np.array(alphas)
    T_offdiag = np.array(betas[:-1])
    eigvals = _tridiag_eigvals(T_diag, T_offdiag)
    return float(np.min(eigvals)), False, T_diag, T_offdiag


def _tridiag_eigvals(diag, offdiag):
    """Eigenvalues of symmetric tridiagonal matrix."""
    if len(offdiag) == 0:
        return diag
    # Use numpy's eigvalsh for symmetric tridiagonal
    T = np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
    return np.linalg.eigvalsh(T)