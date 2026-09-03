"""
Test problem definitions with analytic gradient and analytic Hessian where
cheap, otherwise finite differences (via core.grad_fd / core.hess_fd).
"""

from __future__ import annotations

import numpy as np

from src.core import grad_fd, hess_fd


# ----------------------------------------------------------------------
# Rosenbrock (n-dimensional, analytic)
# ----------------------------------------------------------------------
class Rosenbrock:
    name = "Rosenbrock"

    def __init__(self, n=2):
        self.n = n

    def f(self, x):
        x = np.asarray(x, dtype=float)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        n = self.n
        g = np.zeros(n)
        # contributed formulation
        for i in range(n - 1):
            g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
            g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
        return g

    def hess(self, x):
        x = np.asarray(x, dtype=float)
        n = self.n
        H = np.zeros((n, n))
        for i in range(n - 1):
            H[i, i] += 1200.0 * x[i] ** 2 - 400.0 * x[i + 1] + 2.0
            H[i, i + 1] += -400.0 * x[i]
            H[i + 1, i] += -400.0 * x[i]
            H[i + 1, i + 1] += 200.0
        return H


# ----------------------------------------------------------------------
# Ill-conditioned quadratic  f = 0.5 x^T diag(1, kappa,...) x
# ----------------------------------------------------------------------
class QuadIllCond:
    name = "Quadratic (ill-conditioned)"

    def __init__(self, n=10, kappa=1e4):
        self.n = n
        self.diag = np.geomspace(1.0, kappa, n)

    def f(self, x):
        x = np.asarray(x, dtype=float)
        return 0.5 * np.dot(x * self.diag, x)

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        return self.diag * x

    def hess(self, x):
        return np.diag(self.diag)


# ----------------------------------------------------------------------
# Beale function (2D, nonconvex, multiple local structure)
# ----------------------------------------------------------------------
class Beale:
    name = "Beale"

    def __init__(self):
        self.n = 2

    def f(self, x):
        x = np.asarray(x, dtype=float)
        x1, x2 = x[0], x[1]
        return (1.5 - x1 + x1 * x2) ** 2 \
            + (2.25 - x1 + x1 * x2 ** 2) ** 2 \
            + (2.625 - x1 + x1 * x2 ** 3) ** 2

    def grad(self, x):
        return grad_fd(self.f, x)

    def hess(self, x):
        return hess_fd(self.f, x)


# ----------------------------------------------------------------------
# Himmelblau function (2D, 4 global minima)
# ----------------------------------------------------------------------
class Himmelblau:
    name = "Himmelblau"

    def __init__(self):
        self.n = 2

    def f(self, x):
        x = np.asarray(x, dtype=float)
        x1, x2 = x[0], x[1]
        return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2

    def grad(self, x):
        return grad_fd(self.f, x)

    def hess(self, x):
        return hess_fd(self.f, x)


# ----------------------------------------------------------------------
# Logistic regression (empirical risk, d features + intercept)
# ----------------------------------------------------------------------
class LogisticRegression:
    name = "Logistic Regression"

    def __init__(self, X, y, reg=1e-3):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n = self.X.shape[1] + 1  # + bias
        self.reg = reg

    def _theta(self, x):
        return x[:-1], x[-1]  # weights, bias

    def f(self, x):
        w, b = self._theta(x)
        z = self.X @ w + b
        # stable log-loss
        m = self.X.shape[0]
        loss = np.sum(np.logaddexp(0.0, -self.y * z)) / m
        reg = 0.5 * self.reg * np.dot(w, w)
        return loss + reg

    def grad(self, x):
        w, b = self._theta(x)
        z = self.X @ w + b
        m = self.X.shape[0]
        sig = 1.0 / (1.0 + np.exp(-z))  # p(y=1)
        err = sig - (self.y > 0)         # derivative of log-loss
        gw = (self.X.T @ err) / m + self.reg * w
        gb = np.sum(err) / m
        return np.concatenate([gw, [gb]])

    def hess(self, x):
        return hess_fd(self.f, x, h=1e-4)


# ----------------------------------------------------------------------
# Helper to build a synthetic logistic dataset
# ----------------------------------------------------------------------
def make_logistic_data(n_samples=200, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    w_true = rng.standard_normal(n_features)
    z = X @ w_true + 0.1 * rng.standard_normal(n_samples)
    y = np.where(z > 0, 1.0, -1.0)
    return X, y
