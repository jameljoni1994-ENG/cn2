# CN² Roadmap — Plan & Suggestions

Living plan for the CN² project: research → publication → open-source growth.
Status icons: ✅ done · 🔄 in progress · ⬜ planned · 💡 suggested.

---

## 1. Status snapshot

| Area | State |
|---|---|
| Algorithm v4 (dual gate, 2-Newton steps/cycle, K_ck cadence) | ✅ done |
| Benchmark suite (Rosenbrock/Beale/Himmelblau/Quad/Logistic) | ✅ done |
| Manual baselines (GD, NAG, L-BFGS, Newton, Newton-CG) | ✅ done |
| Counter correctness audit (f/g/hess exact) | ✅ done |
| 6 figures + 7 CSV tables + benchmark.json | ✅ done |
| Two-phase theory (checkpoint subsequence `{x_{jK}}`) | ✅ added to both papers |
| English paper (8 pp, clean PDF) | ✅ done |
| Arabic paper (8 pp, clean PDF) | ✅ done |
| Public GitHub repo + banner + MIT license | ✅ done |

---

## 2. Publication track (targets)

### Track A — SIAM Journal on Optimization (best fit)
- ✅ Theory section with Thm (checkpoint convergence + two-phase decay)
- 🔄 **Add complexity analysis per cycle** (Lemma-style: cost(K_ck, n) per cycle)
- ⬜ Write abstract in submission style; add "main theorem + proof outline only" letter-length presentation
- ⬜ Convert benchmarks to the journal experiment standards (larger n, more trials, seed reports)
- ⬜ Add a real-data application (e.g., logistic/MLP training with Hessian-vector products)

### Track B — NeurIPS / ICML (if positioning as ML optimizer)
- ⬜ Large-n experiments: Hessian-vector-product Newton-CG inner solver
- ⬜ Compare against OPTAMI/NATA (2025) and ICN (NOLTA 2026) numerically — currently only cited
- 🕑 Requires GPU-friendly re-implementation (PyTorch), item 4.3 below

> **Suggestion.** Submit to **SIAM J. Optim as short paper now**, keep NeurIPS as fallback after
> the large-n extension. SIAM reviewers value exactly the checkpoint-subsequence theory we now have.

---

## 3. Theory roadmap

| # | Item | Priority | Status |
|---|---|---|---|
| T1 | Nexus of checkpoint rate + Hessian cost (complexity per cycle) | P0 | ⬜ |
| T2 | Global (nonconvex) far-field rate via Łojasiewicz/KL exponent | P1 | 💡 |
| T3 | Bound `p_j` in terms of local conditioning κ_j along the block | P1 | 💡 |
| T4 | Blockwise (K-strided) subsequence extension of two-phase Thm | P1 | 💡 |
| T5 | Adaptive-τ adaptive-gate convergence (make τ a function of ‖g‖) | P2 | 💡 |
| T6 | Sharp constants: C ~ M/(2μ), ρ = 1-√(μ/L) verified numerically | P2 | 💡 |

**Suggested flagships:** T2 would make the far-field result *global*, separating CN² from
Cubic-Newton theory; T1 is the standard "cost per ε" reviewers will demand.

---

## 4. Algorithm & engineering roadmap

### 4.1 Hessian-vector products (large-n) — TOP priority
- ⬜ `hvp(f, x, v) ≈ (H(x)+εI)v` via finite differences: `(g(x+hv)-g(x-hv)) / (2h)`
- ⬜ Newton-CG inner solver using HVPs for the **Newton phases** (no full H)
- ⬜ Keep `lambda = sqrt(gᵀH⁻¹g)` computable via one CG solve — this preserves the gate
- Expected: n = 10⁴–10⁵ feasible on a laptop

### 4.2 Line-search robustness
- ⬜ Add safe guard: if NAG backtracking fails to find decrease in N steps → fall back to
  gradient step (not Newton) — protects nonconvex escape

### 4.3 PyTorch/JAX backend (GPU) — for NeurIPS track
- 💡 Re-implement `core` ops with `torch.func.hessian`/`jax.jacfwd` so the method integrates
  with deep-learning autodiff; benchmark small MLP/MNIST

### 4.4 Package engineering
- ⬜ `pyproject.toml` + `pip install -e .` (package `cn2`)
- ⬜ CI: GitHub Actions (lint + pytest + a smoke benchmark on ubuntu/mac/win)
- ⬜ Re-enable tests: port `test_rosenbrock.py`, `compare_probe.py` to the current `(x, info)` API
- ⬜ Type hints + docstring cleanup pass
- 💡 Optional `scipy` extras for first-order comparisons (not needed for core)

### 4.5 Reproducibility hardening
- ⬜ Pinned random seeds per problem (Logistic already `seed=0`; make explicit across suite)
- ⬜ Record machine/NumPy version in `benchmark.json` metadata
- ⬜ Add `--quick` smoke mode for CI

---

## 5. Experiment roadmap

| # | Item | Priority | Status |
|---|---|---|---|
| E1 | Condition-number sweep κ ∈ {1e2..1e8} × n {10..1000} | P1 | 💡 |
| E2 | Real datasets (LIBSVM/sklearn): logistic, ridge, small DNN | P1 | ⬜ |
| E3 | Add CRN (cubic-regularized Newton) as baseline for nonconvex claims | P1 | 💡 |
| E4 | Add OPTAMI/NATA + ICN numerical comparison (Track B) | P1 | 💡 |
| E5 | Hessian-evaluation profile with HVP-cost model (walls vs counts) | P1 | ⬜ |
| E6 | Ablation: newton_steps ∈ {1,2,3} effect on Hessian economy | P2 | 💡 |
| E7 | Trajectory figure: λ and ‖g‖ vs cycle, colored by phase (L vs S) | P2 | 💡 |

---

## 6. Open-source & community roadmap

| # | Item | Priority | Status |
|---|---|---|---|
| C1 | README + banner + license + topics | ✅ | done |
| C2 | Contributing guide + CODE_OF_CONDUCT | P2 | ⬜ |
| C3 | GitHub Issues from roadmap items (trackable work) | P1 | 💡 |
| C4 | Continuous integration (Actions: tests + PDF build) | P1 | ⬜ |
| C5 | PyPI publishing (name `cn2` likely taken; consider `cn2-opt`) | P2 | 💡 |
| C6 | Zenodo DOI (make paper+cited-version citable) | P2 | 💡 |
| C7 | Citation file (`CITATION.cff`) + `paper` DOI in README | P2 | 💡 |

---

## 7. Prioritized next moves (proposed order)

1. 🔄 **Update paper conclusions** (items ii/iii now done) — fast, both languages
2. ⬜ **T1 + T2 theory** (complexity/cycle + global nonconvex rate) → strengthens submission
3. ⬜ **4.1 HVP large-n** → unlocks n≥10⁴ + NeurIPS track + E5
4. ⬜ **4.4 packaging + CI** → community-readiness (C4)
5. ⬜ **E2 real data** → SIAM realism
6. ⬜ Create GitHub Issues from this roadmap for tracked work (C3)

---

## 8. Risk matrix

| Risk | Likelihood | Mitigation |
|---|---|---|
| "Hybrid already exists" (OPTAMI/ICN) | Medium | Numerically compare; claim is *stage-switching on geometric gate*, not fusion |
| Large-n HVP loses economy advantage | Medium | Report wall-time vs count; HVP preserves the λ gate via CG |
| Reviewers want global nonconvex proof | Medium | T2 (Łojasiewicz) — realistic, standard path |
| No GPU scale for NeurIPS | Medium | Reposition as "moderate-n Hessian-efficient" to SIAM |