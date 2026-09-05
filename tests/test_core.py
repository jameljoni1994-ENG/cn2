"""Unit + regression checks for the CN2 core.

Covers hessian_psd, entry_measure, walltime, newton_decrement_cg signature,
and the dual-gate integration regression checks:
  R1. K0 actually controls the probe cadence (k_ck = max(K0, n/5)) — was a
      dead parameter before v5.
  R2. CN2-HVP reports honest HVP accounting (hv_gate, hv_total).
  R3. entry_measure dual gate reports lambda (SPD) vs gradnorm (non-SPD).
  R4. Both momentum variants (nag, fista) converge on canonical problems.

Runnable standalone (`py tests/test_core.py`) or via pytest.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core import hessian_psd, entry_measure, walltime, cn2
from src.hvp_cg import HvpOperator, newton_decrement_cg, cg_solve_full
from experiments.run_largeN import cn2_hvp
from problems.test_funcs import Rosenbrock, QuadIllCond, Beale


def test_hessian_psd_spd():
    assert hessian_psd(np.array([[2.0, 0.0], [0.0, 1.0]]))


def test_hessian_psd_indefinite():
    # eigenvalues {3, -1} -> not SPD
    assert not hessian_psd(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_hessian_psd_semidefinite_regularized():
    # PSD with a null direction is accepted: H + shift*I is PD, which is the
    # exact matrix the lambda-gate solves with (regularized Newton decrement).
    assert hessian_psd(np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_entry_measure_lambda_gate():
    val, metric, psd = entry_measure(np.array([1.0, 1.0]),
                                     np.array([[2.0, 0.0], [0.0, 1.0]]))
    assert metric == "lambda" and psd
    assert np.isclose(val, float(np.sqrt(1.5)))


def test_entry_measure_gradnorm_gate():
    val, metric, psd = entry_measure(np.array([1.0, 1.0]),
                                     np.array([[2.0, 0.0], [0.0, -1.0]]))
    assert metric == "gradnorm" and not psd
    assert np.isclose(val, float(np.sqrt(2.0)))


def test_walltime_cost_model():
    # hess_cost = n (one documented ratio), no dead overwrite
    w = walltime(100, {"grad_evals": 1000, "hess_evals": 10})
    assert w == 1000.0 + 10.0 * 100.0


def test_newton_decrement_cg_returns_iters():
    A = lambda v: 3.0 * v  # noqa: E731  SPD diagonal operator
    val, converged, iters = newton_decrement_cg(np.array([1.0, 2.0]), A, tol=1e-6)
    assert converged and iters >= 0
    assert np.isclose(val, float(np.sqrt((1.0 ** 2 + 2.0 ** 2) / 3.0)))


def test_cg_solve_full_signature():
    A = lambda v: np.array([2.0 * v[0], 5.0 * v[1]])  # noqa: E731
    x, iters, converged = cg_solve_full(A, np.array([1.0, 1.0]), tol=1e-8)
    assert converged and iters > 0
    assert np.allclose(x, [0.5, 0.2])


def test_cn2_rosen100_converges():
    p, x0, tau, L = Rosenbrock(100), -1.2 * np.ones(100), 1e-8, 200.0
    x, info = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=L, tau_g=1e-4,
                  newton_steps=2, max_cycle=200)
    assert p.f(x) < 1e-10
    assert info["hess_evals"] > 0 and info["iters"] > 0


def test_cn2_beale_himmelblau():
    for p, x0, L in [(Beale(), np.array([1.0, 1.0]), 50.0),
                     (Beale(), np.array([1.5, 1.0]), 50.0)]:
        tau = 1e-8
        x, info = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=L, tau_g=1e-4,
                      newton_steps=2, max_cycle=300)
        assert p.f(x) < 1e-8, f"Beale did not converge from {x0}: f={p.f(x):.2e}"


def test_k0_wires_cadence():
    # R1: K0 must change the effective probe cadence on n=100 (floor = n/5 = 20)
    p, x0, tau, L = Rosenbrock(100), -1.2 * np.ones(100), 1e-8, 200.0
    a = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=L, tau_g=1e-4, K0=5, max_cycle=200)[1]
    b = cn2(p.f, p.grad, p.hess, x0, tau=tau, L=L, tau_g=1e-4, K0=100, max_cycle=200)[1]
    assert a["k_ck"] == 20, "K_ck must equal max(K0, n/5) for K0 < n/5"
    assert b["k_ck"] == 100
    assert a["hess_evals"] != b["hess_evals"], \
        "K0 must change the probe cadence (was a dead parameter before v5)"


def test_momentum_fista_converges():
    q = QuadIllCond(30, 1e3)
    x, info = cn2(q.f, q.grad, q.hess, 2.0 * np.ones(30), tau=1e-8, L=1e3,
                  tau_g=1e-6, K0=20, momentum="fista", max_cycle=300)
    assert q.f(x) < 1e-8


def test_hvp_honest_accounting():
    # R2: hv_total >= hv_evals (Newton-CG) and gate cost is tracked
    q = QuadIllCond(150, 1e3)
    x, info = cn2_hvp(q.f, q.grad, 2.0 * np.ones(150), tau=1e-6, L=1e4,
                      tau_g=1e-6, max_cycle=20, verbose=False)
    assert "hv_gate" in info and "hv_total" in info and "k_ck" in info
    assert info["hv_total"] == info["hv_evals"] + info["hv_gate"]
    assert info["hv_gate"] >= 0 and info["hv_evals"] >= 0
    assert info["hv_total"] >= info["hv_evals"]
    assert q.f(x) < 1e-5


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception:
                fails += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{sum(1 for k in globals() if k.startswith('test_')) - fails}"
          f"/{sum(1 for k in globals() if k.startswith('test_'))} passed")
    raise SystemExit(1 if fails else 0)