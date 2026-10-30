# Decisions and trade-offs

## Why wgpu + Metal rather than CUDA

* The development and target machine is an Apple Silicon MacBook (M2). CUDA and
  PTX require NVIDIA hardware and are not available on macOS.
* wgpu gives one Rust API and one shader language (WGSL), with a native Metal
  backend on macOS; naga translates the shaders to Metal Shading Language. The
  same code is vendor-neutral and portable to other GPU APIs.
* Vulkan and DX12 backends are available through wgpu but are not yet tested;
  Metal on Apple Silicon is the verified target.

## Why compute only through `wgpu` and not a raw Metal crate

Portability and a single code path for compute and rendering. The cost is that
wgpu's abstraction hides some Metal-specific tuning (e.g. threadgroup memory
details, GPU counters). Deeper profiling is done with Xcode's Metal tools.

## Plain winit + wgpu viewer instead of egui

The spec lists egui as optional. The viewer uses keyboard/mouse controls and a
live window title (fps, steps/frame, isolated ms/step) rather than an in-window
GUI, which keeps the dependency set small and avoids a second, version-coupled
graphics dependency. The headless `aether render` command produces screenshots.

## Workgroup size as a WGSL `override`

One shader source serves every workgroup size; the value is injected at pipeline
creation. Sweeps (`--workgroup-sweep 64,128,256`) need no shader edits.

## Pass batching as the default

Encoding every step as its own compute pass costs a fixed ~35 us per step on the
M2/Metal stack, which dominates small problems. Recording all steps of a batch as
dispatches inside one pass removes most of that cost and is bit-identical
(tested). Both modes stay selectable (`--pass-mode per-step|batched`) so the
experiment is reproducible. See the README for the measured outcome.

## Tolerances over exact equality

CPU/GPU results are compared with documented, tight tolerances rather than bit
equality (`numerics.md`). Rayon vs scalar *is* bit-exact because they share the
same per-element kernels.

## Scope

Not yet implemented: velocity Verlet, workgroup-memory tiling for all-pairs, an
in-window text overlay, and GPU testing on Linux and Windows.
