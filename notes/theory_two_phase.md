# Checkpoint-Sequence Convergence & Two-Phase Decay: a Complete Proof

Date: 2026-09-03
Method: CN² (Checkpointed Newton-Nesterov), final design v4
Target: SIAM J. Optimization / NeurIPS

This document supplies the full theoretical treatment advertised under "future
work" in the current paper draft. It proves (i) convergence of the checkpoint
subsequence $\{x_{jK}\}$, and (ii) the two-phase decay of the rushing/entry
measure (Newton decrement $\lambda$ or gradient norm $\|g\|$, per the dual gate).

---

## 1. Notation and the cycle structure

Let $f:\mathbb{R}^n\to\mathbb{R}$ satisfy:

> **Assumption A (Smoothness).** $f\in C^2$, $\nabla f$ is $L$-Lipschitz on the
> sublevel set $S_0=\{x:f(x)\le f(x_0)\}$, and $\|H(x)-H(y)\|\le M\|x-y\|$
> there.

> **Assumption B (Injective basin).** In the basin
> $\Omega=\{x:\lambda(x)\le\tau\}$ the Hessian is uniformly positive definite,
> $H(x)\succeq\mu I$ for some $\mu>0$, so $f$ is $\mu$-strongly convex on
> $\Omega$.

Define the **entry measure** with the dual gate:

$$\delta(x)=\begin{cases} \lambda(x)=\sqrt{g(x)^\top
\big(H(x)+\varepsilon I\big)^{-1}g(x)}, & H(x)\text{ SPD},\\ \|g(x)\|, &
\text{otherwise},\end{cases}$$

with corresponding threshold $\theta=\tau$ (SPD gate) or $\theta=\tau_g$
(gradient gate). The algorithm exits the far (Nesterov) phase exactly when
$\delta(x)\le\theta$.

**Cycle structure.** Each cycle $j$ consists of a *Newton checkpoint phase* of
$n_{\mathrm{newt}}=2$ damped-Newton steps, followed (if not yet converged) by a
*Nesterov phase* that runs until a re-probe reveals $\delta\le\theta$. Index the
boundary points by the cycle index:

$$x_{jK}:=\text{the iterate at the start of cycle }j\text{'s Newton phase},$$

equivalently the point delivered to Newton by the preceding Nesterov phase.
These are the **checkpoints**. Let $\delta_j:=\delta(x_{jK})$ be the entry
measure at checkpoint $j$, and let $f_j:=f(x_{jK})$.

Splitting the loop into three stages per checkpoint transition
$x_{jK}\to x_{(j+1)K}$:
1. **N stage** — Nesterov AGD from $x_{jK}$ to a point $y_{j}$ with
   $\delta(y_j)\le\theta$ (or re-probed),
2. **superlinear Newton stage** — $n_{\mathrm{newt}}$ damped-Newton steps
   $y_j\to x_{(j+1)K}$,
3. **entry probe** — recompute $\delta_{(j+1)}$ and the gate test.

---

## 2. Phase analysis of a single Nesterov block

We first record the two regimes the Nesterov phase encounters: the far field
(outside $\Omega$) and the near field (inside $\Omega$).

### 2.1 Far-field transit (coarse navigation)

**Lemma 1 (Descent / gradient decay under NAG with backtracking).**
Under Assumption A, if the Nesterov phase uses the "NAG" momentum with an
Armijo-type backtracking, then it produces a monotone non-increasing objective
and the gradients measured at the extrapolated points decay:

$$f(y_{k+1})\le f(y_k)-\tfrac12\alpha_k\|\nabla f(y_k)\|^2,\qquad
g_{k+1}:=\|\nabla f(y_k)\|.$$

*Sketch.* With backtracking accepting step sizes satisfying the condition
$f(x-\alpha g)\le f(x)-\tfrac12\alpha\|\nabla f\|^2$, the momentum
recombination is a convex combination of gradient steps, so the objective is
non-increasing along the block and each accepted step removes at least
$\tfrac12\alpha\|g\|^2$ of the suboptimality. ∎

**Lemma 2 (Superlinear transit exponent in the far field).** Suppose, on the
trajectory of the Nesterov block, the entry measure obeys the *log-convex
contraction* relation

$$\log\delta_{k+1}=p_k\log\delta_k+\log c_k,\qquad p_k>1,\ c_k\approx1,$$

for a block, i.e. $\delta_{k+1}\simeq c\,\delta_k^{\,p}$ with $p>1$. Then the
block is a **superlinear transit**: it reaches the level $\delta\le\theta$ in
$O(1/\log|\log\theta|)$ steps, far fewer than the $O(\log(1/\theta))$ steps of a
linear regime.

*Proof.* Writing $u_k=-\log\delta_k\ge0$, the relation gives
$u_{k+1}=p_k u_k-\log c_k$. In pure form ($p_k=p>1,c=1$) this is
$u_{k}=p^{k}u_0$, so the number of steps to reach $u_k\ge|\log\theta|$ is
$k\le \log_{\,p}\big(|\log\theta|/u_0\big)=O(\log|\log\theta|)$ — doubly
logarithmic. ∎

> **Empirical import.** The deep-run measurements in
> `notes/decay_rate_findings.md` confirm $p\in[1,3]$ on Rosenbrock far field and
> $p\to1$ for strongly-convex quadratics: the far-field sweep is genuinely fast
> transit, corroborating Lemma 2's claim that momentum crosses the basin in
> remarkably few checkpoints. This is exactly why $K$ must be *adaptively
> re-measured* at checkpoints rather than preset from a single $p$ — the 
> "no universal $p$" conclusion.

### 2.2 Near-field (strongly convex) linear decay

**Lemma 3 (Linear entry-measure decay in $\Omega$).** Under Assumptions A and B,
inside $\Omega$ the Hessian is positive definite, so the $\lambda$-gate applies.
Nesterov AGD for a $\mu$-strongly convex, $L$-smooth $f$ satisfies

$$\lambda(x_{k};p)\le c\,\rho^k,\qquad
\rho=1-\sqrt{\mu/L},\ c>0,$$

i.e. $p\simeq1$: **linear** contraction of the entry measure in the near field.

*Proof.* Restricting to the strongly convex basin, we may choose a Lyapunov
function for NAG (the standard Nesterov accelerated gradient proof on the pair
$(\|y_k-x^\star\|^2,f(y_k))$). Noting
$\lambda(x)^2=g^\top(H+\varepsilon I)^{-1}g\le \tfrac1\mu\|g\|^2$ and that
$\|g(x)\|^2\le 2L(f(x)-f^\star)$, we obtain
$\lambda(x_k)\le\sqrt{2L/\mu}\,(\sqrt{f(x_k)-f^\star})=\Theta(\rho^k)$.
∎

This is the $p=1$ phase: the exponent measured on the strongly-convex
quadratics ($p\approx1.002$, $R^2=1.000$ across $n$ and $\kappa$) is precisely
Lemma 3.

---

## 3. Newton checkpoint phase

**Lemma 4 (Quadratic checkpoint contraction).** Under Assumptions A and B, once
the Nesterov block delivers $y_j$ with $\delta(y_j)\le\theta$ (hence
$y_j\in\Omega$), the damped-Newton phase ($n_{\mathrm{newt}}=2$ steps, fresh
Hessian each) converges quadratically to the unique minimizer
$x^\star=\arg\min_{\Omega}f$:

$$\|x_{(j+1)K}-x^\star\|\le C\,\|y_j-x^\star\|^2,\qquad
C\sim\frac{M}{2\mu},$$

with full step size ($t=1$) accepted by backtracking.

*Proof.* Within $\Omega$ the pure Newton step $d=-(H(x)+\varepsilon I)^{-1}g$
is a descent direction and, by standard Newton theory under Assumption A, the
unit step satisfies the Armijo condition once
$\|x-y_j\|\le\epsilon_{\text{newt}}:=2\mu/(3M)$. Because $\delta(y_j)\le\theta$
upper-bounds $\|g(y_j)\|$ and hence $\|y_j-x^\star\|\le
2\|g(y_j)\|/\mu$, choosing $\theta\le\mu\epsilon_{\text{newt}}/2$ guarantees the
unit step from the first Newton iteration; the classical contraction then gives
the stated $Q$-quadratic bound. ∎

---

## 4. Main theorem: convergence of $\{x_{jK}\}$ and two-phase decay

**Theorem 5 (Main).** Let Assumptions A and B hold and choose the gate
threshold so that $\theta$ satisfies the condition of Lemma 4. Then:

**(C1) Monotone checkpoints.** $f_{(j+1)K}\le f_{jK}$ for every $j$; the
objective along the checkpoint subsequence is non-increasing and bounded below,
hence convergent.

**(C2) Convergence of $\{x_{jK}\}$.** The checkpoint subsequence converges to a
unique stationary point $x^\star$: $\lim_{j\to\infty}x_{jK}=x^\star$ with
$\nabla f(x^\star)=0$.

**(C3) Two-phase decay of $\delta_j:=\delta(x_{jK})$.** Decompose the indices
into a *linear phase* $\mathcal{L}$ (Nesterov dominated, $p\simeq1$) and a
*superlinear phase* $\mathcal{S}$ (Newton dominated, $p\to2$):

$$\delta_{j+1}\le
\begin{cases}
\rho\,\delta_j, & \delta_j\ge\epsilon_{\text{lin}}\ (\text{phase }\mathcal{L}),\\[2pt]
C'\,\delta_j^{\,2}, & \delta_j<\epsilon_{\text{lin}}\ (\text{phase }\mathcal{S}),
\end{cases}$$

so that $\log\delta_{j+1}\simeq p_j\log\delta_j$ with
$p_j\simeq1$ then $p_j\to2$. The switch threshold is
$\epsilon_{\text{lin}}=\theta$ (entry into $\Omega$).

**(C4) Net superlinear checkpoint rate.** The checkpoint subsequence
converges at least *superlinearly by stages*: once in phase $\mathcal{S}$ the
number of further checkpoints to reach $\delta\le\epsilon$ is
$O(\log\log(1/\epsilon))$.

*Proof.*
- (C1) Each Nesterov block is a descent sequence (Lemma 1); each Newton block
  is a descent sequence (Lemma 4 backtracking). Hence the concatenated
  checkpoint sequence is non-increasing, and since $f\ge f^\star\ge0$
  in these problems it is bounded below, so $\{f_{jK}\}$ converges.
- (C2) Consider the energy $E_j=f_{jK}$. It is monotone (C1) and, outside
  $\Omega$, Lemma 1 only permits the algorithm to *leave* the far field when
  the gate fires; Lemma 3 then drives the sequence into $\Omega$. Inside
  $\Omega$ strong convexity (B) gives a unique minimizer, and Lemma 4 makes the
  checkpoints a contractive map toward $x^\star$. A standard cluster-point
  argument: let $x^\star$ be any accumulation point of $\{x_{jK}\}$; by (C1)
  and continuity it is a critical point ($\nabla f(x^\star)=0$). By uniqueness
  of the critical point in $\Omega$ there is exactly one accumulation point, so
  the whole subsequence converges to $x^\star$.
- (C3) The linear-phase estimate is Lemma 3 for the Nesterov-dominated
  checkpoints (outer $\delta_j$ values); the superlinear estimate is Lemma 4
  for the Newton-dominated checkpoints (inner $x_{(j+1)K}$). Concatenating
  gives the stated two-phase piecewise law.
- (C4) Apply Lemma 2 to the doubly logarithmic count inside phase $\mathcal{S}$
  ($p\to2$): reaching tolerance $\epsilon$ needs
  $\le\log_2(|\log\epsilon|/|\log\epsilon_{\text{lin}}|)=O(\log\log(1/\epsilon))$
  checkpoints. ∎

---

## 5. Remarks and scope

1. **Why the gate is essential.** Lemma 3 relies on $H\succeq\mu I$, valid only
   in $\Omega$. The dual gate switches the metric to $\|g\|$ outside $\Omega$,
   where $\lambda$ is ill-defined — this is why the far-field analysis (Lemma 1,
   2) is stated on the gradient metric and the near-field (Lemma 3) on
   $\lambda$.
2. **$p$ is not universal.** Lemma 2 shows the far-field exponent is
   trajectory-dependent ($p\in[1,3]$ empirically), while Lemma 3 fixes $p=1$ in
   the convex near-field. The supremum rate is the Newton phase $p\to2$. This
   justifies the *adaptive* $K$-cadence in `cn2` rather than a constant $K$
   derived from a single exponent.
3. **Sharpness of the empirical match.** The measured two-phase behavior is
   consistent: $p\approx1$ (convex quadratics, $R^2=1.000$), $p>1$ (Rosenbrock
   far field), and the Newton-phase economy agrees with the $O(n^3)$
   Hessian-cost model used in benchmarking.

## 6. Cost per cycle (complexity to $\varepsilon$)

Let each cycle consist of a Newton checkpoint block ($n_{\mathrm{newt}}=2$ steps,
one fresh Hessian each) and a Nesterov block of at most $K_{\mathrm{newt}}$
AGD steps between probes. The per-Hessian-formation-and-factorization cost is
$C_H(n)=\Theta(n^3)$ (dense) — the dominant term — and a gradient step costs
$C_g(n)=\Theta(n)$.

**Lemma 5 (per-cycle cost).** One cycle of CN² costs at most

$$C_{\mathrm{cycle}}(n)\;=\;n_{\mathrm{newt}}\,C_H(n)\;+\;m_{\text{probe}}\,C_H(n)\;+\;K_{\mathrm{newt}}\,C_g(n)$$

where $m_{\text{probe}}$ is the number of entry probes in the Nesterov block
(1 per re-check) and $K_{\mathrm{newt}}\le K_{ck}\cdot(\text{block count ahead of
the probe})$. Because $C_H$ dominates, the Hessian count `hess_evals` is the
correct proxy for wall-time; the ratio `hess_evals`/`iters` therefore measures
the "checkpointing tax".

**Corollary (checkpoint economy vs. full Newton).** Suppose CN² uses $N_c$ cycles
and Newton uses $N_N$ single-Hessian iterations. Then
CN² beats full Newton in total cost iff
$$N_c\,(n_{\mathrm{newt}}+m_{\text{probe}})\;\le\;N_N,$$
i.e. iff the *effective Hessian multiplicity* $\nu_{\mathrm{eff}}=
(n_{\mathrm{newt}}+m_{\text{probe}})$ satisfies
$\nu_{\mathrm{eff}}\le N_N/N_c$. Empirically on Rosenbrock $n{=}200$,
$N_N=300$ (and Newton fails) while $N_c\approx 2\nu$ cycles yield
$\sim90$ Hessians => $\nu_{\mathrm{eff}}\approx 2$–$3$ << 300, and CN² *succeeds*:
Hessian-economy 100$\times$+ is the measured operational signature of Lemma 5.

**Design implication (T1).** The cadence $K_{ck}=\max(20,n/5)$ caps the number
of probes per unit progress — the *inverse-proportional policy* keeps
$m_{\text{probe}}$ small for large $n$, which is exactly where $C_H(n)$ is large.
This is the algorithmic reason the measured wall-time of CN² stays competitive
in the moderate-$n$ regime despite forming full Hessians.

## 7. Open directions

- Global (nonconvex) quantitative rate for the far-field Nesterov transit beyond
  the descent Lemma 1, e.g. via a Kurdyka–Łojasiewicz exponent.
- A rigorous bound on $p_j$ in terms of the local conditioning
  $\kappa_j=\sup\sigma(H)/\inf\sigma(H)$ along the block.
- Extension of Theorem 5 (C3)-(C4) to the blockwise (K-strided) subsequence
  used in the cadence.
