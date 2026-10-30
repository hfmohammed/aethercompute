# Design

## Layers

```
aether CLI / viewer
        |
ExperimentConfig  --(seed)-->  deterministic scenario generator   (aether-core)
        |
Workload trait: initialize / step / sync / snapshot / validate    (aether-core)
   |-----------------|------------------------|
 Scalar CPU        Rayon CPU               wgpu + WGSL           (aether-cpu, aether-gpu)
   |-----------------|------------------------|
 Snapshot + error metrics  ->  benchmark records  ->  JSON/CSV/SVG (aether-core, aether-bench)
```

* `aether-core` has no GPU or window dependencies, so identical scenarios run
  in tests, the CLI and the viewer.
* The scalar and Rayon backends call the *same* per-element kernel functions
  (`local_update`, `nbody_update`, `cell_update`); Rayon only changes how the
  loop is scheduled. Each output element depends only on the previous buffer,
  so Rayon results are bit-identical to scalar (tested with `Tolerance::EXACT`).
* The WGSL shaders mirror those kernels operation for operation.

## Workloads

### Particles, local mode (`particles-local`, O(N))

Each particle is updated independently with a constant-cost force: a harmonic
well plus a tangential swirl, linear drag, and reflecting walls.

```
a   = -k * x + s * perp(x)          perp(x, y) = (-y, x)
v'  = (v + a dt) (1 - c dt)         semi-implicit Euler with linear drag
x'  = x + v' dt
walls at +-1: clamp x, v <- -e v    e = restitution = 0.8
```

This is a throughput showcase. It is **not** a gravitational solver.

### Particles, all-pairs N-body (`nbody-allpairs`, O(N^2))

Exact softened 2D gravity, one thread per target particle, global-memory reads:

```
a_i = G * sum_j m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2)
```

`G` defaults to `1/N` so accelerations stay O(1) as `N` grows. Cost grows
quadratically, so this mode is capped at 65,536 bodies and benchmarked at
256-4,096. Throughput is reported in particle-updates/s like the local mode;
because each update costs N interactions, the two modes' numbers are **not**
comparable (the JSON/plots label complexity explicitly).

### Heat diffusion (`heat`)

Explicit FTCS 5-point stencil on the unit square with `dx = dy = 1/(n-1)`:

```
T'[i,j] = T[i,j] + rx (T[i-1,j] - 2T[i,j] + T[i+1,j]) + ry (T[i,j-1] - 2T[i,j] + T[i,j+1])
rx = alpha dt / dx^2,  ry = alpha dt / dy^2
```

The configuration is expressed through the diffusion number `r = alpha dt/dx^2`
(default 0.2); the physical `dt` follows from it. See `numerics.md` for the
stability guard and boundary modes.

## Memory layout (host == WGSL)

| Struct           | Size | Layout |
|------------------|------|--------|
| `Particle`       | 32 B | `pos: vec2 @0, vel: vec2 @8, mass: f32 @16, flags: u32 @20, pad: vec2 @24` |
| `ParticleParams` | 48 B | 8 x f32, `count: u32 @32`, 3 x u32 pad (uniform block, 16-byte multiple) |
| `HeatParams`     | 32 B | `nx, ny: u32, rx, ry: f32, boundary: u32, t_boundary: f32, 2 x u32 pad` |
| `View`           | 32 B | camera/colour uniform for the renderer |

All are `#[repr(C)]` + `bytemuck::Pod`; sizes and key offsets are asserted in
unit tests, and the GPU differential tests would fail immediately on any
host/WGSL layout mismatch.

## GPU execution flow

1. Create an instance, select the high-performance adapter, log backend and
   limits (`aether info` prints the same report). No optional features are
   requested; limits are taken from the adapter.
2. Allocate persistent buffers once: two storage buffers (state A/B,
   ping-pong), one uniform buffer (written once at creation), one staging
   buffer used only for readback.
3. Compile the WGSL module. The workgroup size is a WGSL `override` constant
   set at pipeline creation, so 32/64/128/256 are tuned without editing shaders.
   Pipeline creation runs inside a validation error scope and surfaces as a
   normal `Err` (F09).
4. `step(n)`: one command encoder, a single `queue.submit`. In the default
   `batched` pass mode all `n` dispatches share one compute pass (alternating
   bind groups A->B, B->A; wgpu inserts the storage barriers); `per-step` mode
   uses one pass per step. Dispatch is `ceil(items/wg)` workgroups;
   counts above the per-dimension limit are folded into a 2D dispatch.
5. `sync()` blocks until the device is idle. `snapshot()` copies the current
   buffer to staging, maps it, and copies out. Nothing reads back per frame.
6. The viewer draws from the *same* storage buffer in a render pass: particles
   as instanced quads with additive blending (overdraw becomes density, colour
   encodes speed), heat as a full-screen colour-mapped field.

## Buffer lifetime

State buffers, uniform and staging buffers live as long as the `GpuWorkload`.
`initialize()` re-uploads the deterministic initial state with `write_buffer`
into buffer A. The renderer creates one cheap bind group per frame so it always
points at whichever ping-pong buffer holds the current state.

## Failure handling

* No adapter / device creation failure: `GpuContext::new` returns an error;
  `info` reports it, `validate` skips GPU unless `--require-gpu`, `bench`
  records the error per GPU row and still completes the CPU rows.
* State larger than the adapter's buffer limits, or a workgroup size above its
  limits, returns a descriptive `Err` before any allocation.
* Invalid or unstable configurations are rejected by `ExperimentConfig::validate`.
