# Vanilla pysisyphus and the transactional TDenTrack–ORCA extension

## Theory, implementation, efficiency, and scientific scope

Report date: 22 July 2026  
Vanilla reference: pysisyphus commit `a4ce10dd`  
Modified reference: pysisyphus commit `1a5ea218`  
TDenTrack optimizer API: the current `excited_state_diabatizer/state_tracking.py`

## Executive summary

The modified optimization is best described as a **transactional, state-aware trust-region optimization**, not as an ordinary line search. At a gradient-evaluated geometry \(q_k\), pysisyphus constructs its normal restricted-step rational-function optimization (RFO) proposal \(p_k\). The state-aware controller first examines only the unscaled endpoint

\[
q_k^{(1)} = q_k+p_k.
\]

That examination is an isolated, all-root, energy-only ORCA calculation followed by an exact adjacent-geometry transition-density comparison. If the endpoint has a unique acceptable electronic identity and passes the controller's energy and step guards, it is staged immediately; none of the other scale factors is calculated. Only when that primary endpoint is electronically ambiguous, incomplete, too weakly matched, too long, or uphill under the configured policy does the controller examine alternatives

\[
q_k^{(\lambda)}=q_k+\lambda p_k,
\qquad
\lambda\in\{0.5,0.75,1.25,1.5\}
\]

by default. The accepted endpoint then receives exactly one selected-root ORCA `EnGrad` calculation. The geometry, selected root, wavefunction reference, and quasi-Newton history are committed only after that gradient succeeds at the exact geometry that was surveyed.

Therefore, the proposed efficiency improvement—**do the additional scaled search only in a mixed-state region**—is already the default behavior through `fallback_only=True` and `primary_factor=1.0`. The unavoidable operation at every nonconverged step is the single survey at \(\lambda=1\). Without it, the program does not know whether the proposed endpoint is mixed or whether the state has changed root number. Skipping that survey would amount to trusting an adiabatic root ordinal or ORCA's native root follower, which is the failure mode this project is intended to avoid.

There is still room for optimization. At present, once the primary endpoint fails, all allowed fallback factors are normally evaluated so the controller can rank acceptable alternatives by energy, similarity, margin, and proximity to \(\lambda=1\). A sequential "first electronically acceptable endpoint" policy, an adaptive bracketing policy, or parallel fallback surveys could reduce wall time. Those variants trade information or computational resources for speed and should be exposed as explicit policies rather than silently changing the scientific decision rule.

The modified code retains pysisyphus's mature geometry machinery—Cartesian, redundant, delocalized, and translation/rotation internal coordinates; iterative backtransformation; model and updated Hessians; restricted-step RFO; trust-radius adaptation; convergence tests; and restart/output infrastructure. It adds a new authorization boundary around the electronic state. The important conceptual change is:

> Vanilla pysisyphus asks which root corresponds to the state after an electronic calculation has been made. The modified workflow asks which root is authorized before its production gradient is allowed to enter the optimization.

The extension is deliberately conservative. It can follow a state character through changes in adiabatic root order and can sometimes bridge a narrow mixed region by choosing a shorter or longer clean endpoint. It does not construct a globally diabatic Hamiltonian, compute nonadiabatic couplings, define a differentiable state-specific surface at an exact degeneracy, or replace a multistate/conical-intersection treatment.

## 1. Scope and terminology

### 1.1 Nuclear and electronic variables

Let

\[
x\in\mathbb R^{3N}
\]

be the Cartesian nuclear coordinates in bohr and let

\[
q=q(x)\in\mathbb R^m
\]

denote the coordinate vector used by an optimizer. Depending on the pysisyphus geometry, \(q\) may be Cartesian, redundant internal coordinates (RICs), delocalized internal coordinates (DLCs), or translation/rotation internal coordinates (TRICs).

For a TDDFT/TDA calculation at fixed \(x\), let \(E_I(x)\) be the energy of adiabatic root \(I\). The root ordinal is energy/order metadata local to a calculation. It is not a persistent electronic-state identity. A state with a recognizable charge-transfer or ligand-field character can change from root \(I\) to root \(J\) as the nuclei move.

In this report, **tracked state** or **quasi-diabatic state** means a sequence of adiabatic roots selected by maximum numerical continuity of their transition-density character. This is weaker than a formal diabatic representation. A formal diabatization would define a smooth multistate basis, and often a diabatic Hamiltonian, across a region. The present optimizer instead selects one adiabatic gradient at each accepted endpoint.

### 1.2 Three distinct uses of the word “line search”

The code contains three ideas that should not be conflated:

1. Vanilla RFO can use an **implicit polynomial line search** based on earlier energies and gradients. This is a numerical optimization accelerator.
2. The Campetella SDNTO algorithm reduces its steepest-descent scalar \(\alpha\) when an endpoint energy rises. This is a monotonic step-length policy.
3. The modified controller evaluates a finite set of scale factors \(\lambda\) along one RFO proposal. This is a **discrete electronic-state endpoint search**.

The third operation does not minimize a continuous scalar function along a line. It does not calculate a derivative with respect to \(\lambda\), bracket a one-dimensional minimum, or interpolate an energy polynomial. Its primary question is whether the endpoint has an unambiguous state identity. Calling it a “line search” is understandable, but “scaled endpoint survey” is more precise.

## 2. Vanilla pysisyphus as an external optimizer

### 2.1 Architectural separation

pysisyphus is an external optimizer. It does not normally supply the electronic energy itself. Its main layers are:

- `Geometry`: owns atoms, Cartesian coordinates, the chosen internal-coordinate representation, cached energy/derivative results, constraints, and coordinate transformations.
- calculators such as `ORCA`: generate input, run an electronic-structure program, parse energies and derivatives, retain files, and optionally reuse a previous wavefunction.
- optimizers such as `RFOptimizer`: construct steps from energies, gradients, and an approximate Hessian, decide convergence, update the trust radius and Hessian, and manage history/restart files.
- path and stationary-point machinery: minima, transition states, chains of states, and intrinsic reaction coordinates share common geometry and calculator interfaces.

For an ordinary optimization cycle, the conceptual flow is

\[
x_k
\xrightarrow{\text{calculator}}
\{E_k,g_{x,k}\}
\xrightarrow{\text{coordinate transform}}
\{E_k,g_{q,k}\}
\xrightarrow{\text{optimizer}}
p_k
\xrightarrow{\text{backtransform}}
x_{k+1}.
\]

The electronic-structure package can therefore be exchanged without rewriting the optimization algorithm, provided that its calculator returns a consistent energy and derivative.

### 2.2 Stationary points and the local quadratic model

A minimum of one potential-energy surface satisfies

\[
\min_q E(q),
\qquad
g(q_*)=0,
\qquad
H(q_*)\succeq 0,
\]

where

\[
g_k=\nabla_qE(q_k),
\qquad
H_k=\nabla_q^2E(q_k).
\]

The second-order Taylor model is

\[
m_k(p)=E_k+g_k^Tp+\frac12p^TH_kp.
\]

If \(H_k\) is nonsingular and positive definite, its stationary point is the Newton step

\[
p_k^{\mathrm N}=-H_k^{-1}g_k.
\]

Far from a minimum, an approximate Hessian can be indefinite or the quadratic model can be reliable only locally. An unprotected Newton step may then be uphill, too large, or chemically unreasonable. pysisyphus's `RFOptimizer` uses rational-function optimization and a trust radius to regularize this problem.

### 2.3 Rational-function optimization

The unscaled minimum-search RFO equation is

\[
\begin{pmatrix}
H_k & g_k\\
g_k^T & 0
\end{pmatrix}
\begin{pmatrix}
p_k\\1
\end{pmatrix}
=
\nu
\begin{pmatrix}
p_k\\1
\end{pmatrix}.
\]

For a minimization, the eigenvector associated with the lowest augmented-Hessian eigenvalue is chosen and divided by its final component so that the homogeneous coordinate equals one. The rational model used by current pysisyphus to predict the energy change is

\[
\Delta m_k^{\mathrm{RFO}}(p)
=
\frac{g_k^Tp+\tfrac12p^TH_kp}{1+p^Tp}.
\]

The denominator suppresses the pathological behavior of the unbounded quadratic model and guarantees an appropriate downhill direction under the RFO construction. It does not by itself make an arbitrarily long step trustworthy.

### 2.4 Restricted-step RFO and the trust radius

Let

\[
H_k=V\operatorname{diag}(\epsilon_i)V^T,
\qquad
\tilde g=V^Tg_k.
\]

The implementation solves a scaled augmented problem in this eigenbasis. For a positive scaling parameter \(\alpha\), its matrix is effectively

\[
A(\alpha)=
\begin{pmatrix}
\operatorname{diag}(\epsilon_i)/\alpha & \tilde g/\alpha\\
\tilde g^T & 0
\end{pmatrix}.
\]

Because this restricted-step form is not necessarily symmetric, `HessianOptimizer.solve_rfo` uses a general eigensolver rather than a symmetric one. The code iterates \(\alpha\) until the RFO step satisfies

\[
\lVert p_k(\alpha)\rVert_2\leq\Delta_k
\]

within a small tolerance. If those microiterations fail, pysisyphus falls back to a shifted Newton trust-region solution. In the latter formulation, in the Hessian eigenbasis,

\[
\tilde p_i(\mu)=-\frac{\tilde g_i}{\epsilon_i+\mu},
\]

and \(\mu\) is chosen so that the Hessian is suitably shifted and, when necessary,

\[
\lVert p(\mu)\rVert_2=\Delta_k.
\]

The initial RFO trust radius is 0.5 in the active coordinate units by default, bounded by `trust_min=0.1` and `trust_max=1.0`.

After an accepted geometry is evaluated, the trust agreement ratio is

\[
\rho_k=
\frac{E(q_k+p_k)-E(q_k)}{\Delta m_k(p_k)}.
\]

The current implementation follows the standard qualitative policy:

- if \(\rho_k<0.25\), reduce \(\Delta_k\) by a factor of four, not below `trust_min`;
- if \(\rho_k>0.75\) and the preceding step was essentially on the trust boundary, double \(\Delta_k\), not above `trust_max`;
- otherwise retain the current trust radius.

This is model globalization: it regulates where the approximate Hessian and energy model can be trusted. It is separate from electronic-state continuity.

### 2.5 Hessian initialization and quasi-Newton updates

An exact Hessian is expensive, so pysisyphus can initialize an internal-coordinate Hessian from diagonal model force constants. The default for the relevant RFO workflow is Fischer's model. For example, the implemented bond force constant has the form

\[
h_r=0.3601\exp[-1.944(r-r_{\mathrm{cov}})],
\]

and the implemented bend and torsion diagonals are functions of bond lengths, covalent radii, and local connectivity. The important practical feature is not the exact empirical formula but that chemically stiff stretches start with larger curvature than soft torsions and interfragment motions.

After the first step, let

\[
s_k=q_{k+1}-q_k,
\qquad
y_k=g_{k+1}-g_k.
\]

The default BFGS Hessian update is

\[
H_{k+1}=H_k+
\frac{y_ky_k^T}{y_k^Ts_k}
-
\frac{H_ks_ks_k^TH_k}{s_k^TH_ks_k}.
\]

It satisfies the secant equation

\[
H_{k+1}s_k=y_k
\]

and, under the curvature condition \(y_k^Ts_k>0\), preserves positive definiteness. pysisyphus also implements damped BFGS, SR1, PSB, a flowchart choice among updates, Bofill mixing, and periodic/adaptive exact-Hessian recalculation.

For state following, the secant vectors must come from the same physical surface. A root error contaminates \(y_k\), and therefore the Hessian, even if all individual gradients are numerically converged. This is one reason transactional root authorization is more than a cosmetic labeling feature.

### 2.6 Convergence

The default `gau_loose` convergence thresholds in the present branch are

\[
\max|f|\leq2.5\times10^{-3},
\qquad
\operatorname{RMS}(f)\leq1.7\times10^{-3},
\]

\[
\max|p|\leq1.0\times10^{-2},
\qquad
\operatorname{RMS}(p)\leq6.7\times10^{-3},
\]

with forces in \(E_h/a_0\) for Cartesian components and the corresponding internal-coordinate units for angular components. The modified optimizer also supports an optional calculator `state_convergence_ok` hook. `TDenTrackORCA` does not currently implement a separate final hook; instead, every noninitial committed point has already passed an `ACCEPT` decision and a successful audited gradient transaction. Adding a final confidence policy remains possible without changing the geometric thresholds.

### 2.7 Other vanilla pysisyphus capabilities

The pysisyphus framework is broader than minimum optimization. For a first-order saddle point, partitioned RFO separates a maximizing subspace, normally one reaction mode, from the remaining minimizing subspace and solves appropriate augmented problems in each. Chain-of-states methods operate on images \(x_i\) between endpoints. The central nudged-elastic-band decomposition is schematically

\[
F_i^{\mathrm{NEB}}
=
-\nabla E(x_i)_\perp
+F^{\mathrm{spring}}_{i,\parallel},
\]

which removes the true force along the path tangent and the spring force perpendicular to it. String methods instead redistribute images along a path parameter. Intrinsic reaction coordinates integrate a mass-weighted steepest-descent path away from a transition state, schematically

\[
\frac{dX}{ds}=-\frac{\nabla_XE}{\lVert\nabla_XE\rVert}.
\]

These methods share the same calculator and coordinate infrastructure. The transactional extension discussed here is implemented for single-geometry optimization steps; extending its semantics to multiple simultaneous images would require a separate transaction for every electronically active image.

## 3. Coordinate systems in vanilla pysisyphus

### 3.1 Wilson's B matrix

For internal coordinates \(q_i(x)\), the Wilson matrix is

\[
B_{i\alpha}=\frac{\partial q_i}{\partial x_\alpha},
\qquad
\Delta q\simeq B\Delta x.
\]

For a redundant coordinate set, \(m\) can exceed the physical vibrational dimension. pysisyphus therefore uses generalized inverses. Its `RedundantCoords.inv_B` evaluates the algebraic form

\[
B^+=(B^TB)^+B^T,
\]

with an SVD threshold, and defines the internal-space projector

\[
P=BB^+.
\]

Forces are negative gradients. The implemented transformation is

\[
f_q=P(B^T)^+f_x.
\]

The projection removes components that cannot correspond to a Cartesian displacement and is modified when internal constraints are present.

### 3.2 Hessian transformation

The Cartesian and internal Hessians are related by

\[
H_x=B^TH_qB+K,
\]

where the curvature term is

\[
K_{\alpha\beta}
=
\sum_i(g_q)_i
\frac{\partial^2q_i}{\partial x_\alpha\partial x_\beta}.
\]

Thus the implemented Cartesian-to-internal transformation is

\[
H_q=(B^T)^+(H_x-K)B^+.
\]

pysisyphus generates or evaluates the first and second derivatives of primitive coordinates rather than treating the coordinate transformation as linear. In a redundant space the optimization Hessian is projected as

\[
H_q^{\mathrm{proj}}
=PH_qP+\eta(I-P),
\]

with \(\eta=1000\) by default. The large shift prevents redundant null-space directions from entering the RFO step.

### 3.3 Redundant internal coordinates

RICs contain primitive stretches, bends, linear bends, proper and improper torsions, out-of-plane coordinates, and optional Cartesian/fragment coordinates. They usually reduce coupling relative to raw Cartesians and permit chemically informed model Hessians, but they are not a minimal basis.

Their principal numerical complication is that an internal displacement is not exactly realizable by a single linear Cartesian update. Given a target

\[
q^*=q(x_0)+\Delta q,
\]

pysisyphus iterates

\[
r^{(\ell)}=q^*-q(x^{(\ell)}),
\qquad
\Delta x^{(\ell)}=(B^{(\ell)T})^+r^{(\ell)},
\]

\[
x^{(\ell+1)}=x^{(\ell)}+\Delta x^{(\ell)}.
\]

It corrects periodic dihedral differences, watches bends and rotations for singular behavior, enforces frozen atoms and constraints, and stops when the RMS Cartesian correction is below \(10^{-6}\) bohr or when the iteration ceases to improve. A failed transformation can trigger rebuilding of the internal coordinates.

### 3.4 Delocalized internal coordinates

For primitive Wilson matrix \(B_p\), DLC construction diagonalizes

\[
G=B_pB_p^T.
\]

If \(U\) contains the retained eigenvectors, then

\[
q_D=U^Tq_p,
\qquad
B_D=U^TB_p.
\]

The full molecular set normally retains \(3N-6\) directions. A DLC optimizer step is converted back to primitive space as

\[
\Delta q_p=U\Delta q_D
\]

before iterative Cartesian backtransformation. DLCs remove explicit redundancy at the geometry where they are defined, although their basis can itself change when coordinates are rebuilt.

### 3.5 TRIC

TRIC augments fragment-local internal coordinates with three translations and three rotations for each fragment. This is useful for weakly bound, solvated, or multicomponent systems because it avoids inventing a dense network of artificial interfragment bonds. The rotational coordinates depend on a reference frame. In pysisyphus, TRIC also recalculates the B matrix at every internal-to-Cartesian backtransformation microcycle.

This frame dependence caused an important integration bug during development: copying a `Geometry` before converting a trial internal step reinitialized the TRIC rotation references and DLC basis. The numeric step from the live optimizer was then interpreted in a different coordinate frame. Commit `1a5ea218` changed state-aware endpoint generation to use

```python
geometry.get_temporary_coords(geometry.coords + lambda_ * step)
```

on the live geometry with a pure, nonmutating backtransformation. The applied Cartesian geometry is later checked against the exact Cartesian endpoint that was electronically surveyed.

## 4. Excited-state tracking in vanilla pysisyphus

### 4.1 CIS-like states

For a CIS/TDA-like state \(I\), suppressing spin temporarily,

\[
|\Psi_I\rangle
=
\sum_{ia}d^I_{ia}|\Phi_i^a\rangle,
\]

where \(i\) and \(a\) label occupied and virtual orbitals. In linear-response TDDFT, the transition-density-like amplitudes used for overlap tracking are normally \(X+Y\); in TDA, \(Y=0\).

Root tracking compares the state at two geometries, \(A\) and \(B\). The reference can be the first geometry, the previous geometry, or an adaptively updated earlier geometry.

### 4.2 Many-electron wavefunction overlap

The formal overlap is

\[
\langle\Psi_I^A|\Psi_J^B\rangle
=
\sum_{ia,jb}
d^{A,I}_{ia}d^{B,J}_{jb}
\langle\Phi_i^a(A)|\Phi_j^b(B)\rangle.
\]

Slater-determinant overlaps reduce to determinants of occupied-orbital overlap matrices. If MO coefficients are columns of \(C_A\) and \(C_B\), the cross-geometry MO overlap is

\[
S_{AB}^{\mathrm{MO}}
=C_A^TS_{AB}^{\mathrm{AO}}C_B,
\]

where

\[
(S_{AB}^{\mathrm{AO}})_{\mu\nu}
=
\langle\chi_\mu(R_A)|\chi_\nu(R_B)\rangle.
\]

The many-electron approach is the most complete of pysisyphus's historical tracking choices and, unlike transition densities and NTOs, can in principle compare a ground state with excited states. It is also more expensive because it requires determinant algebra; the vanilla implementation interfaces Plasser's `wfoverlap` program.

### 4.3 Transition-density overlap

Let \(T^{A,I,\sigma}_{ia}\) and \(T^{B,J,\sigma}_{jb}\) be spin-resolved transition amplitudes, and partition the cross-MO overlap into occupied and virtual blocks. The transition-density contraction used in the project can be written

\[
O_{IJ}
=
\sum_{\sigma}\sum_{ia,jb}
T^{A,I,\sigma}_{ia}
S_{ij}^{\mathrm{occ},\sigma}
S_{ab}^{\mathrm{virt},\sigma}
T^{B,J,\sigma}_{jb}.
\]

This avoids determinant calculations and evaluates all root pairs efficiently by tensor or matrix contractions. It measures similarity of one-electron transition character rather than the full many-electron wavefunction.

Historically, when a true cross-geometry AO matrix was unavailable, vanilla pysisyphus reconstructed a same-geometry AO metric from one MO coefficient matrix:

\[
S^{\mathrm{AO}}\approx(C^{-1})^TC^{-1},
\]

using a pseudoinverse in code, and optionally renormalized the other set of MOs in that metric. This can be reasonable for very small displacements, but it is not the actual integral

\[
\langle\chi_\mu(R_A)|\chi_\nu(R_B)\rangle.
\]

The older `double_mol` route could calculate an explicit supermolecule overlap for supported programs. The modified branch also adds an opt-in exact BSON cross-overlap path to the legacy tracker, but the new transactional backend does not depend on this mutable class.

### 4.4 Natural transition orbitals

For a transition matrix \(T\), the singular-value decomposition is

\[
T=U\Sigma V^T.
\]

The occupied NTOs are the occupied MOs rotated by \(U\), and the particle NTOs are the virtual MOs rotated by \(V\). If singular values or their squares are strongly concentrated in a few pairs, NTOs give a compact hole-particle representation of the excitation.

Campetella and Sanz García proposed state comparison through weighted hole and particle NTO overlaps. In their approximation, off-diagonal NTO-pair mixing is neglected and the weights at adjacent geometries are taken to be similar. Schematically,

\[
S_h^{AB}
\approx
\sum_k c_{k,h}^{A}
\langle\phi_{k,h}^{A}|\phi_{k,h}^{B}\rangle,
\]

\[
S_p^{AB}
\approx
\sum_k c_{k,p}^{A}
\langle\phi_{k,p}^{A}|\phi_{k,p}^{B}\rangle,
\qquad
S_{\mathrm{NTO}}=S_h+S_p.
\]

Absolute orbital overlap or spatial-overlap variants remove arbitrary phase sensitivity. NTO tracking is inexpensive and chemically interpretable, but multiple states can share very similar dominant NTO pairs. Campetella and Sanz García therefore recommend extra diagnostics such as transition dipoles when several similar contributions are present.

### 4.5 Vanilla root assignment and reference policies

The vanilla `OverlapCalculator.track_root` takes the absolute value of an overlap matrix and either:

- selects the largest element in the tracked reference row, or
- uses a Hungarian assignment over the whole matrix and takes the assigned column for the tracked row.

The root is changed to that column. For an ORCA gradient calculation, `ORCA.store_and_track` first stores the all-root data from the completed calculation. If tracking detects a root flip, ORCA repeats the electronic calculation at the same geometry with the updated root, so the returned gradient corresponds to the newly selected root.

The reference policies are:

- `first`: always compare with the initial state;
- `previous`: compare adjacent geometries;
- `adapt`: update the reference only if the largest overlap exceeds 0.5 and the ratio of second-largest to largest overlap lies between 0.3 and 0.6 by default.

The adaptive logic tries to avoid updating the reference when states are either trivially distinct or strongly mixed.

### 4.6 Strengths and limitations of the vanilla tracker

The historical implementation is computationally economical. A normal cycle can obtain the gradient and the data required for tracking in one electronic calculation; a detected root change adds a second gradient calculation. It supports several electronic-structure packages and several overlap definitions.

Its decision, however, is fundamentally an `argmax` or assignment. There is no mandatory lower bound on state similarity, no required separation from the second candidate, no explicit same-multiplicity guard, no fail-closed complete-root-window contract, no immutable pending/committed distinction, and no rule preventing an ambiguous root from entering optimizer history. Its approximate AO metric is also weaker than an exact adjacent-geometry integral when the displacement is appreciable.

## 5. Relation to the Campetella and Steinmetzer algorithms

### 5.1 Campetella's SDNTO

The SDNTO optimization uses steepest descent:

\[
x_{n+1}=x_n-\alpha\nabla E_{\mathrm{TES}}(x_n).
\]

At \(x_{n+1}\), it calculates the NTOs of all requested excited states, compares them with the target excited state (TES) at \(x_n\), and makes the highest-overlap root the next TES. If the energy rises, \(\alpha\) is halved and the displacement is retried. If the selected root changes, an additional calculation obtains the gradient of the new TES. Convergence is based on the largest force component.

The conceptual contributions inherited by this project are:

- state identity must be based on numerical electronic character rather than root number;
- the comparison must be made between adjacent nuclear geometries;
- a changed root must receive the correct gradient;
- step length can be adapted when the electronic/energetic endpoint is undesirable.

The modified code replaces steepest descent with RFO/TRIC, replaces dominant-NTO similarity with full transition-density overlap, adds exact cross-geometry AO integrals, and makes state selection transactional.

### 5.2 Steinmetzer's pysisyphus formulation

Steinmetzer, Kupfer, and Gräfe generalized excited-state tracking to pysisyphus's Hessian-based external optimizer. Their paper formulates RFO, trust-radius control, internal coordinates, WFO/TDen/NTO tracking, and first/previous/adaptive references. It demonstrates that a robust optimizer plus state tracking can require dramatically fewer expensive gradients than steepest descent for difficult excited-state optimizations.

The present work follows that architecture: pysisyphus remains the geometry optimizer and an external electronic-structure program remains the energy/gradient provider. The main departure is prospective authorization. The original tracker discovers and corrects a root after one calculation at the endpoint; the transactional version first performs a root-complete survey, refuses ambiguity, and only then permits the selected-root gradient.

### 5.3 Why the intergeometry overlap is essential

The electronic state lives in an orbital basis that moves with the atoms. Comparing two coefficient arrays directly would assume that basis function \(\chi_\mu(R_A)\) is the same vector as \(\chi_\mu(R_B)\), which is false. The correct comparison must transport information between the two AO spaces through

\[
S_{AB}^{\mathrm{AO}}
=
\left[
\langle\chi_\mu(R_A)|\chi_\nu(R_B)\rangle
\right].
\]

This is the mathematical content of the adjacent-geometry “superposition” or double-molecule technique. It is not merely a numerical embellishment. It defines the metric needed to compare orbitals and transition densities located on different nuclear geometries.

Its caveats are equally important:

- atom order and basis-function ordering must be identical;
- basis exponents, contraction coefficients, spherical/Cartesian conventions, and ECP definitions must match;
- diffuse functions and large displacements can make cross metrics ill-conditioned;
- a large overlap certifies similarity in the chosen one-electron transition-density metric, not equality of full correlated wavefunctions;
- if the root of interest leaves the calculated root window, exact integrals cannot recover missing electronic information;
- state phase is arbitrary, so selection normally uses the absolute normalized overlap, while the signed matrix must be retained for subspace analysis.

The current BSON implementation checks these structural assumptions and also evaluates the reverse integral to require

\[
S_{AB}^{\mathrm{AO}}
\simeq
(S_{BA}^{\mathrm{AO}})^T.
\]

## 6. The modified transactional electronic-state model

### 6.1 Immutable snapshots

`ElectronicSnapshot` associates one exact geometry with:

- the roots actually parsed;
- the roots requested from the calculation;
- a selected root, if committed;
- corrected state energies and excitation energies;
- multiplicities and optional spin diagnostics;
- retained `.cis`, BSON, GBW, input, output, and audit paths;
- scalar provenance metadata.

Coordinates and numerical arrays are copied into immutable byte-backed storage. This prevents an optimizer or callback from silently changing the geometry represented by an electronic record.

The distinction between `roots` and `requested_roots` is deliberate. If the job was expected to return roots \(1,\ldots,N\) but only a subset was parsed, the missing states can include the true continuation of the tracked state. The default selector therefore fails closed on an incomplete root window.

### 6.2 Tracking sessions and transaction states

`TrackingSession` owns:

- one committed snapshot;
- a stable anchor snapshot;
- immutable pending surveys;
- a generation counter;
- committed history.

Calling `survey` registers a pending candidate but does not move the reference. Calling `select` is also read-only. Only

```text
ACCEPT decision + successful selected-root gradient
```

allows `commit`. Commit re-evaluates the decision, installs the selected candidate, clears all alternatives, and increments the generation. `MANIFOLD`, `RETRY`, and `HALT` cannot be committed.

This gives the electronic analogue of a database transaction:

```text
committed reference
    -> read-only probes
    -> stage exactly one accepted endpoint
    -> run exact selected-root gradient
    -> atomic commit
```

Any failure before the final operation discards pending data and restores the previous root/GBW/geometry state.

### 6.3 Exact ORCA 6.1.1 artifact loading

The production backend is explicitly gated to ORCA 6.1.1. It loads:

- MO coefficients, AO shell metadata, self-overlap matrices, atom positions, basis ordering, and ECP information from BSON;
- signed TDA \(X\) or TDDFT \(X+Y\) amplitudes from `.cis`;
- state energies, excitation energies, final-energy markers, multiplicities, root markers, and termination data from output.

The active occupied and virtual orbital ranges in ORCA `.cis` headers are treated as **zero-based and inclusive**. A range \(r_0\ldots r_1\) therefore maps to Python slice `r0:r1+1`. This corrected the earlier one-index inconsistency that could either produce a shape failure or silently contract the wrong orbitals.

The loader requires the MOs to satisfy, for each spin,

\[
C_\sigma^TS^{\mathrm{AO}}C_\sigma\simeq I.
\]

It also verifies CIS amplitude dimensions against the active orbital slices and checks excitation energies encoded in `.cis` against those parsed from output.

### 6.4 Exact adjacent-geometry transition-density overlap

For committed geometry \(A\) and candidate geometry \(B\), the backend evaluates

\[
S_{AB}^{\mathrm{AO}}
=\langle\chi(A)|\chi(B)\rangle
\]

directly from the two retained BSON shell sets using `Wavefunction.S_with`. For spin \(\sigma\),

\[
S_{AB}^{\mathrm{MO},\sigma}
=(C_A^\sigma)^T
S_{AB}^{\mathrm{AO}}
C_B^\sigma.
\]

Let \(S_{AB}^{o,\sigma}\) and \(S_{AB}^{v,\sigma}\) be the active occupied and virtual blocks. The full signed root-overlap block is

\[
O_{rs}^{AB}
=
\sum_\sigma
\sum_{ia,jb}
T_{r,ia}^{A,\sigma}
(S_{AB}^{o,\sigma})_{ij}
(S_{AB}^{v,\sigma})_{ab}
T_{s,jb}^{B,\sigma}.
\]

The implementation is the tensor contraction

```python
einsum("ria,ij,ab,sjb->rs", T_ref, S_occ, S_virt, T_cur)
```

summed over alpha and beta spin blocks.

The same contraction at one geometry gives transition-density Gram matrices

\[
G^A_{rs}=\langle T_r^A,T_s^A\rangle,
\qquad
G^B_{rs}=\langle T_r^B,T_s^B\rangle.
\]

Electronic eigenstates are orthogonal as many-electron wavefunctions, but their one-electron transition densities need not be mutually orthogonal in this metric. Retaining the full Gram matrices is therefore necessary for mathematically valid subspace comparisons.

### 6.5 Scalar normalization and root selection

For the committed reference root \(r\) and candidate root \(s\), define

\[
\widetilde O_{rs}
=
\frac{O_{rs}}
{\sqrt{G^A_{rr}G^B_{ss}}}.
\]

The signed value is retained, but the root score is

\[
c_s=|\widetilde O_{rs}|,
\]

because a global phase change of one transition vector has no physical significance. Values exceeding one beyond a numerical tolerance are rejected as an inconsistent overlap/metric combination.

Let \(s_1\) and \(s_2\) be the best and second-best multiplicity-compatible roots. The assignment margin is

\[
m=c_{s_1}-c_{s_2}.
\]

The default `SelectionConfig` uses

\[
c_{s_1}\geq0.65,
\qquad
m\geq0.10
\]

for an ordinary unique assignment, requires a complete root window, and requires the same multiplicity when the reference multiplicity is known. Energy availability and strict monotonicity are optional in the selector itself; the pysisyphus step controller applies its own descent policy by default.

### 6.6 Near-degenerate manifolds and principal angles

#### 6.6.1 Why individual-root overlaps can become ambiguous

Suppose that at geometry \(A\), two nearly degenerate states span the
transition-density subspace

\[
\mathcal M_A
=
\operatorname{span}\{T_2^A,T_3^A\},
\]

where \(T_i^A\) is the transition-density representation of adiabatic state
\(i\). At the next geometry, the electronic-structure solver may return a
different orthonormal representation of the same physical subspace:

\[
\begin{pmatrix}
T_2^B\\
T_3^B
\end{pmatrix}
=
\begin{pmatrix}
\cos\phi & \sin\phi\\
-\sin\phi & \cos\phi
\end{pmatrix}
\begin{pmatrix}
T_2^A\\
T_3^A
\end{pmatrix}.
\]

For \(\phi=45^\circ\), the original state \(T_2^A\) overlaps equally with the
two states returned at \(B\):

\[
\left|\langle T_2^A,T_2^B\rangle\right|
\approx
\left|\langle T_2^A,T_3^B\rangle\right|
\approx
\frac{1}{\sqrt 2}.
\]

A single-root selector therefore has approximately zero assignment margin.
There is no basis-invariant way to declare either new adiabatic root to be the
unique continuation of old root 2. Nevertheless,

\[
\operatorname{span}\{T_2^B,T_3^B\}
=
\operatorname{span}\{T_2^A,T_3^A\}.
\]

The individual eigenvectors have rotated, but the two-dimensional electronic
subspace is perfectly preserved. Principal angles are used to distinguish this
situation from the loss of the followed electronic character from the root
window.

This ambiguity is not merely a numerical inconvenience. At an exact
degeneracy, any unitary rotation within the degenerate eigenspace is another
equally valid set of adiabatic eigenvectors. Near a degeneracy, small changes in
geometry, convergence thresholds, or numerical noise can still produce large
changes in the reported individual eigenvectors even when their collective
subspace varies smoothly.

#### 6.6.2 Mathematical construction

Let the columns of \(A\) and \(B\) contain transition densities belonging to
the proposed reference and candidate manifolds:

\[
A=
\begin{bmatrix}
T_{r_1}^{A} & \cdots & T_{r_k}^{A}
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
T_{s_1}^{B} & \cdots & T_{s_k}^{B}
\end{bmatrix}.
\]

Their within-geometry Gram matrices and intergeometry cross-overlap are

\[
G_A=A^\dagger A,
\qquad
G_B=B^\dagger B,
\qquad
O=A^\dagger B.
\]

The transition densities are not required to be orthonormal in this metric.
Consequently, applying an ordinary singular-value decomposition directly to
\(O\) would mix physical subspace continuity with unequal norms and
within-manifold nonorthogonality. The metric must first be removed by
whitening. Conceptually, the whitened cross-overlap is

\[
W=G_A^{-1/2}OG_B^{-1/2}.
\]

The implementation first performs explicit diagonal normalization. With

\[
D_A=\operatorname{diag}(G_A),
\qquad
D_B=\operatorname{diag}(G_B),
\]

define

\[
\widehat G_A=D_A^{-1/2}G_AD_A^{-1/2},
\qquad
\widehat G_B=D_B^{-1/2}G_BD_B^{-1/2},
\]

\[
\widehat O=D_A^{-1/2}OD_B^{-1/2}.
\]

The whitened cross-overlap is

\[
W=\widehat G_A^{-1/2}
\widehat O
\widehat G_B^{-1/2}.
\]

This is algebraically the metric-aware comparison above, with improved
diagnostics for the individual normalized overlaps. If

\[
W=U\Sigma V^\dagger,
\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_k),
\]

then

\[
\sigma_i=\cos\theta_i
\]

are the cosines of the principal angles between the two transition-density
subspaces. Therefore:

- \(\sigma_i\approx1\), equivalently \(\theta_i\approx0\), means that the
  corresponding direction in the reference subspace is well represented in
  the candidate subspace;
- a small \(\sigma_i\) means that at least one direction of the reference
  manifold is missing from, or poorly represented by, the candidate manifold;
- \(\sigma_{\min}\) is a conservative measure of whole-manifold continuity;
- \(\theta_{\max}=\arccos(\sigma_{\min})\) is the worst principal angle.

For the pure two-state rotation in the preceding example, both singular values
are one and both principal angles are zero, even though the overlap of one old
root with the two new roots is evenly divided. The result is invariant to
permutations, rotations, and sign or phase changes inside the chosen subspaces.

#### 6.6.3 When manifold analysis is invoked

The selector always attempts an ordinary single-root assignment first. If the
best and second-best normalized scores are \(c_1\) and \(c_2\), it defines

\[
m=c_1-c_2.
\]

With the current defaults, a possible near-degenerate manifold is considered
when

\[
c_1\geq0.65,
\qquad
m<0.10,
\qquad
|\Delta E|\leq0.10\ \mathrm{eV},
\]

where the energy condition refers to the relevant competing roots. With
`manifold_when_energy_unavailable=True`, an unavailable competitor gap is also
treated conservatively as a possible manifold rather than as evidence of
separation.

Candidate manifold members are multiplicity-compatible roots that are both
score-close to the best candidate and energy-close to it, or have an
unavailable gap under that policy. A corresponding reference manifold is
constructed from multiplicity-compatible roots close in energy to the
committed root. Principal-angle analysis requires the full cross-overlap and
both Gram matrices and, by default, equal reference and candidate manifold
dimensions.

A one-state reference apparently splitting over two candidate roots is thus
not automatically accepted as a continuous two-state manifold. It is a
dimension-changing ambiguity and currently produces `RETRY`. The optimizer can
then test another endpoint rather than expanding the definition of the tracked
state after the fact.

By default, `min_subspace_singular_value` is unset. Principal-angle diagnostics
can still be calculated, but no additional numerical singular-value gate is
imposed. A production study that wants such a gate must configure and validate
the threshold on representative trajectories. Passing that gate would certify
subspace continuity; it would not create a unique root.

#### 6.6.4 What the current optimizer does with a manifold

Principal angles answer a different question from single-root selection. They
can answer

> Is this ambiguity caused by a preserved multistate subspace whose individual
> adiabatic roots have rotated?

They do not answer

> Which single adiabatic root supplies the energy and gradient for the next
> nuclear step?

The present implementation therefore **diagnoses** a continuous manifold but
does not optimize a manifold-valued electronic state. Even a perfectly
preserved manifold remains `MANIFOLD` with `selected_root=None`; it is never
silently converted to `ACCEPT`. In the transactional optimization protocol:

1. no unique root is authorized at the manifold endpoint;
2. no selected-root gradient is requested there;
3. the trial geometry and electronic state are not committed;
4. the step controller may survey another displacement factor, for example
   \(0.5\), \(0.75\), \(1.25\), or \(1.5\);
5. if another endpoint has a unique acceptable root, ORCA calculates that
   root's analytic gradient and the corresponding geometry step can commit;
6. if every permitted endpoint remains ambiguous or otherwise unacceptable,
   the optimizer raises `NoAcceptableStateStep` and retains the last committed
   geometry, energy, gradient, and electronic reference.

This behavior allows a shorter step to stop before a mixed region or a longer
step to land beyond a localized mixed region, while preventing a gradient on
an electronically unauthorized root from contaminating the geometry or
quasi-Newton Hessian.

#### 6.6.5 Why a continuous manifold does not define one gradient

Consider a state defined as a geometry-dependent rotation of adiabatic states,

\[
|\Phi_\alpha\rangle
=
\sum_i U_{i\alpha}|\Psi_i\rangle.
\]

The matrix \(U\) supplied by a subspace-alignment construction describes how
two representations may be aligned, but it does not by itself define a unique
physical diabatic Hamiltonian or its nuclear derivative. Differentiating the
rotated Hamiltonian introduces off-diagonal derivative information—usually
expressed through derivative couplings or equivalent interstate derivative
matrix elements. Consequently, the gradient of a chosen rotated state is not,
in general, obtained by simply selecting one adiabatic gradient or rotating a
list of diagonal gradients.

A genuine manifold-aware geometry optimization would first have to define its
objective. Examples include:

- a fixed-weight state-averaged or ensemble energy,

  \[
  E_{\mathrm{avg}}=\sum_iw_iE_i,
  \qquad
  \nabla E_{\mathrm{avg}}=\sum_iw_i\nabla E_i,
  \]

  which optimizes the specified ensemble rather than one diabatic electronic
  state;
- a diabatic-state optimization based on a physically specified diabatic
  transformation and the derivative information needed for its Hamiltonian;
- a conical-intersection or other multistate objective using multiple
  energies, gradients, and, where required, derivative couplings.

Transition-density overlaps and their principal angles provide none of these
objectives by themselves. They diagnose continuity of electronic character.
For this reason the current backend rejects ORCA `TGradList` multigradient
operation rather than treating an arbitrary member's adiabatic gradient as the
gradient of the manifold.

### 6.7 Selection outcomes

The four statuses have distinct meanings:

- `ACCEPT`: one root is uniquely and consistently identified and may be staged.
- `MANIFOLD`: the state belongs to a meaningful near-degenerate subset, but no unique adiabatic root is authorized.
- `RETRY`: the endpoint may be recoverable with another step, larger root window, or corrected data—for example low overlap, low nondegenerate margin, missing roots, multiplicity mismatch, or unequal manifold dimensions.
- `HALT`: the survey is internally invalid or inconsistent with its reference and should not be retried unchanged.

## 7. The ORCA survey/gradient protocol

### 7.1 Isolated all-root survey

Every surveyed endpoint is run in a unique audit directory by a child ORCA calculator. The survey input removes `IRoot`, sets a complete `NRoots` window, and can use the committed GBW as an SCF guess. It is an energy-only TDDFT/TDA job: no analytic nuclear gradient is requested.

The retained artifacts include input, output, `.cis`, BSON, GBW, densities when available, and a JSON manifest containing geometry, roots, energies, multiplicities, active spaces, full signed overlap matrix, both Gram matrices, norms, and file paths. A rejected endpoint therefore remains scientifically inspectable without mutating the live parent calculator.

### 7.2 Root-numbering conventions

ORCA root numbering is not universal across reference types.

- A restricted, spin-adapted triplet calculation uses a multiplicity-local root ordinal. Triplet root \(k\) must be requested as `IRoot k` and `IRootMult triplet`.
- An unrestricted open-shell TDDFT calculation uses the global printed `STATE N` ordinal, because states of different approximate multiplicity can be interleaved.

The `.cis` loader retains the global root, root within multiplicity, ORCA gradient root, multiplicity, and response-vector metadata so that these conventions cannot be accidentally mixed. The selected gradient output is audited against the echoed `IRoot`, `IRootMult`, `DE(CIS)` marker, state-of-interest report, and normal-termination marker. `FollowIRoot true` and `TGradList` are rejected because they would make the externally authorized root ambiguous.

### 7.3 Consistent energy scale

ORCA's printed TDDFT state table is constructed from a reference energy and excitation energies:

\[
E_I^{\mathrm{table}}=E_{\mathrm{SCF}}+\omega_I.
\]

Late state-independent terms, notably D3(BJ), can be absent from that table but present in the native `.engrad` scalar. The backend parses the numeric correction \(\delta\) and uses

\[
E_I=E_{\mathrm{SCF}}+\omega_I+\delta
\]

for every root. Excitation gaps remain unchanged. ORCA 6.1.1 can anchor `FINAL SINGLE POINT ENERGY` to different roots in different TDDFT/TDA paths, so the final scalar is used to identify and audit its provenance, not to select the tracked state.

At a selected-root gradient endpoint, the raw `.engrad` energy must match the corrected selected-root survey energy within tolerance. The calculator retains the raw scalar as `last_orca_engrad_energy`, returns only the corrected selected-state energy and forces to `Geometry`, and removes the legacy uncorrected all-energy vector from that result. This keeps optimizer energy differences, descent decisions, and trust updates on one surface.

### 7.4 Consistent LR-CPCM surface

ORCA's defaults for excited-state energy-only and analytic-gradient jobs can use different solvent response regimes. Because this workflow deliberately alternates those job types, `CPCMEQ` must be explicitly stated whenever CPCM/SMD is active. Otherwise the survey energy and selected-root gradient could describe different potential-energy surfaces. The usual relaxed excited-state optimization uses `CPCMEQ true`; a frozen nonequilibrium response must be chosen explicitly and used consistently everywhere.

### 7.5 Selected-root gradient and commit

After staging, `TDenTrackORCA.get_forces` requires the requested coordinates to match the staged coordinates. It temporarily sets the authorized root and candidate GBW, runs one ORCA `EnGrad`, audits the root identity and energy, and validates that all force components are finite and have the expected dimension.

Only then does it call `TrackingSession.commit`. On any exception, it restores the previous root and GBW and leaves the electronic reference unchanged. A force call at a new geometry with no accepted stage raises `UnstagedGeometryError`. A harmless reevaluation at the already committed geometry is allowed.

## 8. State-aware RFO endpoint control

### 8.1 Position of the controller in an optimization cycle

For a committed gradient point \((q_k,E_k,g_k,H_k)\), the modified cycle is:

1. Update the trust radius and Hessian using only previously committed points.
2. Construct the ordinary RFO proposal \(p_k\).
3. If the current point is already geometrically converged, and any calculator-supplied electronic convergence hook also passes, stop without another survey.
4. Construct the exact Cartesian endpoint for \(q_k+p_k\) by nonmutating backtransformation in the live coordinate frame.
5. Run the all-root survey at \(\lambda=1\).
6. If it passes all controller guards, stage it immediately.
7. Otherwise survey allowed shorter/longer factors.
8. Rank acceptable alternatives and stage exactly one.
9. Apply its scaled internal step and verify that the resulting Cartesian geometry equals the surveyed endpoint.
10. At the start of the next cycle, run the staged selected-root gradient.
11. Commit electronic and geometric state; only then can the new point enter the next Hessian secant update.

The production gradient occurs at the next optimizer `housekeeping` call because that is the normal pysisyphus cycle structure. Transaction bookkeeping bridges that boundary.

### 8.2 Primary-first fallback behavior

The default controller is initialized with

```python
factors=(0.5, 0.75, 1.0, 1.25, 1.5)
primary_factor=1.0
fallback_only=True
require_descent=True
```

Internally it orders the probes as

\[
1.0,\;0.5,\;0.75,\;1.25,\;1.5.
\]

After the primary survey, it immediately stops the probe loop if `_passes_controller_guards` is true. Thus, in a clean descending region, only one survey is run.

The fallback factors are launched when any controller guard fails, not solely when the selector returns `MANIFOLD`. The reasons include:

- `RETRY` or `HALT` electronic decision;
- normalized similarity below a controller override;
- margin below a controller override;
- an uphill candidate when `require_descent=True`;
- unavailable candidate energy under a required descent check;
- scaled step beyond `max_step_norm` or the optimizer's `trust_max`;
- failed/rebuilt internal-coordinate backtransformation.

This explains why a log can show an electronically `ACCEPT`ed \(\lambda=1\) and nevertheless contain all five surveys: `ACCEPT` is TDenTrack's electronic decision, while the pysisyphus controller can separately reject it as uphill.

### 8.3 Why the \(\lambda=1\) survey cannot normally be skipped

The program cannot infer strong mixing without evaluating the candidate states. At \(q_k\), it knows the committed state and gradient. It does not know at \(q_k+p_k\):

- which adiabatic root carries that state character;
- whether the character has split between roots;
- whether the correct root lies outside the requested window;
- whether a spin-incompatible root appears deceptively similar;
- whether ORCA's root follower would jump to a remote state.

Root energy gaps and NTOs at \(q_k\) are at best predictors. An avoided crossing can occur within the proposed displacement even if the starting point is well separated. Therefore a production-safe policy must survey the primary endpoint before authorizing its gradient.

One could deliberately implement **periodic auditing**—for example, use the previous root ordinal for several steps and run all-root surveys only when a predictor signals danger. That would be cheaper, but it would change the safety contract. The missed-event probability would need to be quantified on real trajectories, and any wrong gradient could corrupt both the geometry and quasi-Newton Hessian before the next audit. Given the stated motivation that ORCA's native follower is unreliable, periodic auditing should be an experimental performance mode, not the default.

### 8.4 Shorter and longer endpoints

A shorter factor can stop before a narrow mixed region:

\[
0<\lambda<1.
\]

A longer factor can land beyond it:

\[
\lambda>1.
\]

The longer-step idea is scientifically plausible when state mixing is localized and the same quasi-diabatic character re-emerges at a lower-energy geometry. It remains a heuristic because the optimizer does not know the electronic path between the two endpoints. A clean large-\(\lambda\) endpoint does not prove that a unique differentiable adiabatic state connects the endpoints, nor that the RFO model is accurate over the enlarged displacement.

For this reason, the default controller requires scaled proposals to remain below `trust_max`. Going farther requires both `respect_trust_max=False` and an explicit finite `max_step_norm`. This separation prevents an electronic bridge heuristic from silently overriding geometric model validity.

### 8.5 Ranking fallback endpoints

After a primary failure, acceptable alternatives are ranked lexicographically by:

1. lowest selected-state endpoint energy;
2. highest normalized similarity;
3. largest assignment margin;
4. factor closest to one.

This requires evaluating all allowed fallback factors. It is more informative than taking the first clean root, but it can be expensive.

### 8.6 Interaction with the RFO trust model

If the controller scales \(p_k\) to \(\lambda p_k\), `RFOptimizer.on_step_control` recomputes the predicted RFO energy change as

\[
\Delta m_k^{\mathrm{RFO}}(\lambda p_k)
=
\frac{
\lambda g_k^Tp_k+	frac12\lambda^2p_k^TH_kp_k
}{1+\lambda^2p_k^Tp_k}.
\]

The subsequent trust ratio therefore compares the actual accepted endpoint with the model prediction for the actual controlled displacement, not the abandoned unscaled proposal.

The transactional controller disables vanilla RFO's implicit polynomial line search, GDIIS, and GEDIIS. It also disables analogous unsurveyed preapplication/line-search behavior in L-BFGS and SQNM paths. Those accelerators can construct an interpolation or extrapolation geometry that has not crossed the state-authorization boundary. Re-enabling them would require making every implicit electronic probe transactional.

Post-step reparametrization is disabled for the same reason: it could change a geometry after its electronic survey but before its selected-root gradient.

### 8.7 Rollback and restart

Before applying a controlled step, the optimizer stores the last gradient-evaluated internal and Cartesian coordinates and the electronic tracking revision. If execution stops, an exception occurs, internal coordinates rebuild, output writing fails, or the maximum cycle count is reached before the selected-root force call commits, pysisyphus discards pending surveys and restores that geometry.

If the electronic revision has already advanced, the gradient and state have committed; rolling back only the geometry would create an inconsistent pair, so the committed geometry is retained while the later error is propagated.

Restart serialization includes the committed electronic snapshot and controller history, but not pending or staged trials. Such endpoints must be surveyed again after restart. This is conservative and reproducible.

## 9. Efficiency analysis and proposed variants

### 9.1 Cost model

Let

- \(C_S(N)\) be the cost of one all-root, energy-only TDDFT/TDA survey with \(N\) roots;
- \(C_G(I)\) be the cost of one analytic gradient for selected root \(I\);
- \(n_S\) be the number of endpoint surveys performed in a cycle.

Ignoring small Python/integral-analysis overhead,

\[
C_{\mathrm{cycle}}
\approx
n_SC_S(N)+C_G(I).
\]

With the primary-first default:

\[
n_S=1
\]

in a clean descending region, and at most five with the default factor set after a primary failure. An always-exhaustive implementation would pay five surveys in every cycle. Primary-first therefore removes 80% of survey jobs in the common clean case.

Vanilla pysisyphus is cheaper in the clean case: it usually pays roughly one gradient calculation and extracts all-root tracking data from it. On a detected root change, it repeats the gradient. The modified scheme intentionally pays an extra all-root survey to guarantee that no production gradient is taken on an unauthorized state.

For the retained remote Ru calculations, an all-root survey was on the order of 25 minutes and a selected-root gradient about 50 minutes. These timings are system- and hardware-specific, but they illustrate the scale: a clean primary-first cycle is roughly 75 minutes, whereas five serial surveys plus a gradient would be roughly 175 minutes. Avoiding unnecessary fallback surveys is therefore essential.

### 9.2 Separate electronic fallback from energy fallback

The default `require_descent=True` makes the endpoint controller monotonic. This is stricter than a conventional trust-region method, which can tolerate small uphill steps and then adjust \(\Delta\) from \(\rho_k\). Two coherent policies are possible:

**Conservative monotonic policy**

- survey \(\lambda=1\);
- if the state is clean but energy rises, try alternative factors;
- never spend the selected-root gradient on an uphill surveyed endpoint.

This saves expensive gradients at the price of more surveys and may be robust far from a minimum.

**Trust-region policy**

- set `require_descent=False` in the step controller;
- use fallback only for electronic ambiguity, incompleteness, multiplicity, or step bounds;
- allow the RFO predicted/actual energy mechanism to respond to a small uphill committed step.

This is closer to vanilla quasi-Newton theory and can reduce fallback surveys, especially near convergence where numerical noise can produce tiny nonmonotonic energy changes. It also risks paying for a gradient at an energetically poor endpoint. A useful middle ground would permit

\[
E(q_k+p_k)-E(q_k)\leq\epsilon_E
\]

with a small method-dependent tolerance, or use a nonmonotone window rather than exact descent.

The choice should be explicit because “strong mixing” and “uphill” are different failure diagnoses.

### 9.3 Sequential first-acceptable fallback

After the primary endpoint fails, a `first_acceptable` mode could survey factors in a configured order and stop as soon as one passes. For example:

```text
1.0 -> 0.75 -> 0.5 -> 1.25 -> 1.5
```

prioritizes caution, whereas

```text
1.0 -> 1.25 -> 0.75 -> 1.5 -> 0.5
```

tests the “bridge beyond mixing” hypothesis earlier. The benefit is lower expected \(n_S\). The cost is that the chosen endpoint is order-dependent and may not be the lowest-energy clean alternative.

This should be a new policy such as

```python
fallback_selection="first_acceptable"
```

alongside the current exhaustive `lowest_energy` behavior.

### 9.4 Adaptive electronic bracketing

Rather than use a fixed factor list, an adaptive policy could treat electronic confidence as a diagnostic along \(\lambda\):

1. evaluate \(\lambda=1\);
2. if mixed, evaluate one shorter and one longer probe, for example 0.75 and 1.25;
3. compare status, score, margin, energy, and manifold membership;
4. continue only in the direction whose electronic identity improves;
5. stop at the first clean endpoint or a strict job/step budget.

Because state identity is not a smooth scalar, this is not classical bisection. Nevertheless, it can avoid calculating both remote ends when the first directional probes clearly show where the tracked character re-emerges.

### 9.5 Parallel fallback surveys

Once \(\lambda=1\) fails, the remaining isolated survey jobs are independent because all compare against the same committed reference. They can be submitted in parallel. This reduces wall time but not CPU time and increases instantaneous memory/license/core demand. A practical HPC policy would:

- run the primary survey synchronously;
- on failure, submit bounded fallback surveys as a job array;
- collect all successful manifests;
- apply the same deterministic ranking;
- cancel remaining jobs only if using a first-acceptable policy.

### 9.6 Dynamic root windows

The cost of TDDFT response calculations grows with the number of roots. A two-tier policy could use a smaller root window in clearly isolated regions and expand it when the selected root approaches a window edge, the score falls, or an energy-density predictor detects crowding. This can be safe only if the small window has a guard band around the selected state. A state outside the calculated window has zero opportunity to be selected, so aggressive window reduction can create false confidence.

### 9.7 Reuse and batching

The current surveys already reuse the committed GBW as a guess and preserve candidate GBWs for the selected gradient. Further possibilities include:

- reuse converged orbitals more aggressively across nearby fallback factors;
- group independent fallback jobs at the scheduler level;
- avoid repeated parsing/loading of the unchanged committed `.cis` and BSON within one proposal by caching immutable validated reference data;
- cache its self Gram matrix and active-space validation keyed by snapshot identity;
- retain exact integral objects if their setup cost is material.

These optimizations do not change the scientific policy and are preferable to skipping the primary survey.

### 9.8 Recommended default policy

For production development, the recommended sequence is:

1. Keep the mandatory \(\lambda=1\) all-root survey.
2. Keep `fallback_only=True`.
3. Decide explicitly whether energy monotonicity belongs in endpoint selection; test both `require_descent=True` and a small-tolerance trust-region variant.
4. Add `fallback_selection="first_acceptable"` as an optional performance mode while retaining exhaustive lowest-energy ranking as the conservative reference.
5. Add adaptive/parallel fallback only after the serial decision logs are validated on scan geometries with known root crossings.
6. Cache committed BSON/CIS parsing and Gram data.

## 10. Scientific limitations and failure modes

### 10.1 An accepted endpoint is not a continuous diabatic path

The overlap criterion compares endpoints. If \(\lambda=1.5\) lands on a clean state beyond a mixed region, the code has not integrated a state through the intervening geometries. It has established only that the candidate transition density resembles the committed one. The accepted analytic gradient is the gradient of the adiabatic root at the candidate endpoint.

### 10.2 Degeneracy and differentiability

Near an exact degeneracy, individual adiabatic eigenvectors can rotate arbitrarily within the degenerate subspace. Pairwise root overlaps are then representation-dependent, even though the subspace is well defined. Principal-angle analysis correctly recognizes subspace continuity, but it cannot manufacture a unique state-specific derivative. A true treatment may require state averaging, multistate optimization, nonadiabatic couplings, or conical-intersection algorithms.

### 10.3 Transition-density scope

Transition densities and NTOs are excited-state quantities relative to a reference. They do not generally capture crossings between the ground state and an excited state. The Steinmetzer paper explicitly notes that a genuine many-electron WFO is required for that capability. The present trusted backend is transition-density based.

### 10.4 TDDFT response limitations

No root-tracking algorithm fixes deficiencies of the electronic-structure model. Linear-response TDDFT can be problematic for strong double-excitation character, long-range charge transfer with an unsuitable functional, spin contamination, instabilities, or regions where a single-reference description breaks down. A numerically continuous transition density may still correspond to an inaccurate physical surface.

### 10.5 Multiplicity

Same-multiplicity filtering is only as good as the multiplicity metadata encoded or inferred for the response roots. In unrestricted calculations, \(\langle S^2\rangle\) contamination can blur spin character. A strict integer multiplicity label should therefore be supplemented by spin diagnostics for difficult open-shell regions.

### 10.6 Root-window completeness

The selector can prove only that the best match among calculated roots is unique. It cannot prove that an uncalculated state would not match better. The complete contiguous window contract prevents missing records within the requested range but cannot guarantee that \(N\) roots are enough. Tracking a state near root \(N\), falling overlap, or systematic movement toward the window edge should trigger a larger root calculation.

### 10.7 Energy consistency

Survey ranking and trust updates are meaningful only if all jobs use the same Hamiltonian and energy convention: functional, basis/ECP, integration grid, SCF settings, dispersion, solvent response, point charges, charge, multiplicity, and TDA/TDDFT choice. The modified backend audits several of these, but user-supplied per-call inputs must still be passed identically to surveys and gradients.

### 10.8 Internal-coordinate singularities

RIC/DLC/TRIC improve optimization conditioning but can rebuild or become singular at large structural changes, linear bends, or rotations approaching their reference limit. A longer electronic bridge step is especially likely to stress backtransformation. The controller treats a required coordinate rebuild as a rejected trial rather than surveying a geometry defined in a new uncommitted internal basis.

### 10.9 Thresholds are model parameters

`min_score=0.65`, `min_margin=0.10`, and `manifold_gap_ev=0.10` are defensible initial values, not universal constants. Their calibration should examine:

- known smooth scan segments;
- known avoided crossings and strong mixing regions;
- dependence on root-window size;
- TDA versus full TDDFT amplitudes;
- basis and functional dependence;
- whether the selected state has a single dominant transition or a distributed transition density.

The report manifest should always retain the raw signed matrices so later threshold changes can be evaluated without rerunning ORCA.

## 11. Validation strategy for the modified code

### 11.1 Algebraic/unit validation

Essential unit tests include:

- zero-based inclusive `.cis` active ranges;
- exact identity-MO and rotated-MO overlap cases;
- forward/reverse AO cross-overlap transpose symmetry;
- phase flips, root permutations, and rotations inside a manifold;
- nonorthogonal transition-density Grams and principal-angle whitening;
- incomplete windows, wrong multiplicities, singular Grams, and normalized values above one;
- stage/gradient coordinate mismatch and rollback;
- root-number translation for restricted triplets and unrestricted global states;
- dispersion-corrected energy matching and alternative ORCA final-energy anchors;
- explicit LR-CPCM response consistency;
- `Geometry.set_results` accepting only the normalized energy/forces mapping.

### 11.2 Real ORCA fixtures

Synthetic tests cannot validate undocumented binary/output conventions. Retained ORCA 6.1.1 fixtures should cover:

- restricted TDA singlets;
- restricted spin-adapted triplets;
- unrestricted open-shell states with interleaved multiplicities;
- full TDDFT \(X+Y/X-Y\) storage;
- D3(BJ) and no-dispersion cases;
- CPCM equilibrium and explicitly nonequilibrium cases;
- at least one ECP transition-metal system.

### 11.3 Scan replay before live optimization

The relaxed Ru scan is especially valuable because roots 1–3 are relatively straightforward and roots 4 onward contain more difficult rearrangements. For every adjacent pair, replay should record:

- normalized score of the expected state;
- second score and margin;
- energy gap to the competitor;
- selected global/local root and multiplicity;
- minimum manifold singular value where relevant;
- whether shortening or lengthening the geometry interval improves uniqueness.

This establishes whether the thresholds distinguish known clean and difficult regions before analytic gradients and Hessian updates are allowed to compound errors.

### 11.4 Optimization validation ladder

A prudent progression is:

1. optimize an isolated, well-separated low root with primary-only surveys;
2. reproduce a vanilla pysisyphus result where both methods follow the same root;
3. optimize a state that changes adiabatic root once but remains unambiguous;
4. use a scan-derived geometry immediately before a known mixed region and test shorter-only fallback;
5. test longer fallback with a strict Cartesian displacement cap and inspect the intervening scan points;
6. test restart immediately before survey, after staging, and after electronic commit;
7. only then attempt the problematic higher Ru roots.

Energy decrease alone is not validation. The final evidence should include root-overlap audit matrices, NTO interpretation, multiplicity/spin behavior, energy/gradient agreement, and repeatability under smaller trust radii and larger root windows.

## 12. Original versus modified pysisyphus

| Aspect | Vanilla pysisyphus at `a4ce10dd` | Modified pysisyphus at `1a5ea218` |
|---|---|---|
| Primary role | General external optimizer for minima, transition states, paths, and IRCs | Same optimizer, with an opt-in transactional state-authorization layer |
| Geometry theory | Cartesian, RIC, DLC, model Hessians, iterative backtransformation | Retained; TRIC trial endpoints are now generated nonmutatingly in the live coordinate frame |
| Minimum step | Restricted-step RFO with adaptive trust radius | Same RFO proposal; accepted scale can be changed by state-aware endpoint control |
| Hessian update | BFGS by default, plus several alternatives | Same, but only successfully committed state-consistent gradients enter secant history |
| Electronic packages | Broad calculator support | Transactional production backend currently specific to ORCA 6.1.1; vanilla calculators remain available |
| Historical state metrics | WFO, transition density, NTO, TOP | Trusted transactional backend uses signed transition-density overlap; legacy metrics remain separate |
| Cross-geometry AO metric | Often reconstructed from one MO matrix; explicit double-molecule only for some calculators | Exact BSON shell integral \(S_{AB}\), with basis/ECP/order/coordinate/transpose validation |
| State comparison | Absolute overlap row `argmax` or Hungarian assignment | Self-normalized scores, signed block retention, complete window, multiplicity, margin, and optional energy guards |
| Near degeneracy | No explicit noncommittable manifold state | Explicit `MANIFOLD`; Gram-whitened principal-angle analysis; never authorizes an arbitrary member |
| Timing of selection | Calculate endpoint gradient, track, and rerun gradient if root flips | Survey all roots first, stage one root, then allow exactly one selected-root production gradient |
| Mutation model | Root/history are mutable during calculation sequence | Immutable snapshots; pending surveys are read-only; generation-checked atomic commit |
| Rejected trial effect | Ambiguity can still influence root/history unless handled by caller | Rejected probes do not change root, GBW, geometry, optimizer history, or Hessian |
| Step alternatives | RFO trust adjustment, GDIIS/GEDIIS, polynomial line search | Discrete primary-first shorter/longer state surveys; implicit unsurveyed accelerators disabled |
| Clean-cycle electronic cost | Usually one gradient job | One all-root energy survey plus one selected-root gradient |
| Root-flip/mixed cost | Usually a repeated gradient on detected flip; ambiguous `argmax` still returns a root | Up to five energy surveys plus one gradient by default; no gradient if no acceptable endpoint exists |
| Energy handling | Calculator-dependent all-energy/gradient parsing | Corrected state-energy scale, dispersion audit, native EnGrad match, scalar provenance |
| Solvent handling | User responsibility | Explicit `CPCMEQ` required under implicit solvent to keep survey/gradient surfaces identical |
| ORCA root audit | Root input generated, but not the full transactional output certification | Echoed `IRoot`/`IRootMult`, `DE(CIS)`, state-of-interest, termination, geometry, and energy all audited |
| Restart | Geometry/optimizer/calculator history | Adds committed electronic snapshot and controller history; omits pending/staged trials |
| Scientific behavior at degeneracy | Tends to choose a largest-overlap root | Refuses a unique root and requests another endpoint or halts after alternatives are exhausted |
| Principal limitation | Can silently force weak/ambiguous assignments; approximate cross metric | More expensive and ORCA-specific; still not a formal multistate or diabatic optimization |

## 13. Conclusions

Vanilla pysisyphus supplies the right optimization foundation for this project: robust internal coordinates, restricted-step RFO, trust-region globalization, quasi-Newton curvature, multiple electronic-structure backends, and mature optimizer bookkeeping. Its historical state tracking already embodies the central insight of the Campetella and Steinmetzer work—that electronic character, not adiabatic root number, should define continuity.

The modified branch changes the safety model. Exact adjacent-geometry AO integrals place transition densities from different nuclear geometries in the correct common metric. Normalization, assignment margins, multiplicity, root completeness, and principal-angle manifold analysis turn a raw largest-overlap choice into an auditable decision. The survey/stage/gradient/commit transaction prevents rejected or ambiguous electronic states from entering the geometry or Hessian history.

The additional fallback search is already conditional. In ordinary clean regions, the code performs one \(\lambda=1\) survey and stops. That survey is the minimum cost of externally certifying the root. The most promising efficiency work is therefore not to remove it, but to reduce what happens after it fails: distinguish electronic ambiguity from strict energy descent, offer a first-acceptable or adaptive fallback policy, parallelize truly independent fallbacks, cache committed electronic data, and tune root windows conservatively.

Finally, bridging a narrow mixed region with a longer step is a useful and testable heuristic, but it should retain its present epistemic status. It can find a clean lower-energy endpoint with familiar transition-density character. It cannot prove that the skipped interval defines a smooth unique state-specific surface. Where every bounded endpoint remains a manifold, stopping is not an algorithmic failure; it is the correct indication that the scientific problem has become multistate.

## References and implementation map

### Primary papers

1. J. Steinmetzer, S. Kupfer, S. Gräfe, “pysisyphus: Exploring potential energy surfaces in ground and excited states,” *International Journal of Quantum Chemistry* **121** (2021), e26390. DOI: 10.1002/qua.26390.
2. M. Campetella, J. Sanz García, “Following the evolution of excited states along photochemical reaction pathways,” *Journal of Computational Chemistry* **41** (2020), 1156–1164. DOI: 10.1002/jcc.26162.

### Main source locations

- Vanilla/modified optimizer loop: `pysisyphus/optimizers/Optimizer.py`
- RFO and trust mechanics: `pysisyphus/optimizers/RFOptimizer.py`, `HessianOptimizer.py`
- RIC/TRIC and backtransformation: `pysisyphus/intcoords/RedundantCoords.py`, `intcoords/update.py`
- DLC basis: `pysisyphus/intcoords/DLC.py`
- Historical tracker: `pysisyphus/calculators/OverlapCalculator.py`
- ORCA calculator/parser integration: `pysisyphus/calculators/ORCA.py`
- Discrete state-aware endpoint controller: `pysisyphus/optimizers/step_control.py`
- Transactional ORCA adapter: `pysisyphus/calculators/TDenTrackORCA.py`
- ORCA 6.1.1 all-root survey and exact cross overlap: `pysisyphus/calculators/ORCA6StateSurvey.py`
- Immutable selection and manifold mathematics: `tdentrack/excited_state_diabatizer/state_tracking.py`
- User-facing integration description: `pysisyphus/docs/es_optimization.rst` and `tdentrack/docs/MANUAL.md`
