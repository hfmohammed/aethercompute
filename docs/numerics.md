# Numerics

## Comparison metrics

`aether_core::validate::compare` reports, for candidate `a` vs reference `b`:

* `max_abs  = max |a - b|`
* `rms      = sqrt(mean (a - b)^2)`
* `max_rel  = max |a - b| / max(|b|, 1e-3)` (denominator floored so values near
  zero do not dominate)
* `nonfinite` = number of NaN/Inf elements (any makes the run fail)

An element passes when `|a - b| <= abs + rel * |b|`. A comparison passes when
every element passes, `rms <= tol.rms`, and `nonfinite == 0`. There is no
"percent correct" figure anywhere; results are the raw error metrics.

Particle snapshots are compared on position and velocity (mass is constant,
flags are integer).

## Tolerances

| Comparison        | abs                | rel  | rms                |
|-------------------|--------------------|------|--------------------|
| Rayon vs scalar   | 0 (bit-exact)      | 0    | 0                  |
| GPU, local        | 1e-5 * sqrt(steps) | 1e-4 | 1e-6 * sqrt(steps) |
| GPU, all-pairs    | 5e-5 * sqrt(steps) | 1e-4 | 1e-5 * sqrt(steps) |
| GPU, heat         | 1e-6 * sqrt(steps) | 1e-4 | 1e-7 * sqrt(steps) |

CPU and GPU are *not* expected to be bit-identical: the GPU compiler may fuse
multiply-adds and reassociate, and Metal enables fast-math by default. The
tolerances grow with sqrt(steps) as a heuristic for accumulated rounding. They
are deliberately tight; the benchmark JSON records the achieved `max_abs` and
`rms` for each configuration so the margin is visible (see `results/`).

The all-pairs sum uses the same accumulation order on CPU and GPU, which keeps
its error small; the floor on `max_rel` is `REL_FLOOR = 1e-3`.

## Heat: stability and boundaries

Explicit FTCS in 2D is stable iff `rx + ry <= 1/2`; for equal spacing this is
`alpha dt / dx^2 <= 1/4`. `HeatParams::new` enforces the sum form (so
anisotropic settings are handled correctly) and rejects non-positive or
non-finite values. The CLI default is `r = 0.2`.

* **Dirichlet:** the outer ring is held at `t_boundary` (0). The initial field
  is zeroed on the ring so the initial state is consistent.
* **Neumann (insulated):** the missing neighbour of an edge cell equals the
  edge cell (ghost = self), i.e. zero normal gradient. This is symmetric, so
  total heat is conserved up to rounding.

## Invariants covered by tests

* Heat: uniform field is steady; Neumann conserves total heat (CPU 1e-4,
  GPU 1e-3 relative); Dirichlet total heat is non-increasing; discrete maximum
  principle (values stay within the initial min/max); symmetric spike
  spreading; convergence to the mean under Neumann; boundary ring held.
* Particles: zero force is ballistic; wall reflection and restitution; drag
  reduces speed; state stays finite and inside the bounds over thousands of
  steps; initial state is deterministic per seed and differs across seeds.
* N-body: pair symmetry (equal and opposite impulses), zero self-force,
  softening keeps near-coincident forces finite, total momentum conserved,
  bounded energy drift for semi-implicit Euler (kinetic + softened potential,
  f64 diagnostics).
* Differential: scalar vs Rayon (bit-exact) and scalar vs GPU for 1, 2, 31, 63,
  64, 65, 127, 1000, 1024, 10 000 and 100 000 particles; heat grids from 3x3 to
  1024x1024; non-multiple-of-workgroup counts; workgroup sizes 1-256; both
  boundary modes; long runs (1000 steps).

## Known limitations

* Velocity Verlet (listed as optional in the spec) is not implemented;
  semi-implicit Euler is the only integrator, so energy drift comparisons
  between integrators are future work.
* The all-pairs kernel is the simple global-memory version. Workgroup-memory
  tiling is not implemented.
* GPU timestamp queries are available on the M2 adapter but not used;
  compute timing is wall-clock around submit + device idle.
