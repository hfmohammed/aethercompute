// Explicit FTCS 5-point heat step with ping-pong buffers.
// Mirrors `aether_core::heat::cell_update`.

struct Params {
    nx: u32,
    ny: u32,
    rx: f32,
    ry: f32,
    boundary: u32, // 0 = Dirichlet (outer ring held), 1 = Neumann (insulated)
    t_boundary: f32,
    pad0: u32,
    pad1: u32,
}

override WG: u32 = 64;

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.y * (nwg.x * WG) + gid.x;
    let nx = params.nx;
    let ny = params.ny;
    if (idx >= nx * ny) {
        return;
    }
    let i = idx % nx;
    let j = idx / nx;
    let t = src[idx];
    let edge = i == 0u || j == 0u || i == nx - 1u || j == ny - 1u;
    if (edge && params.boundary == 0u) {
        dst[idx] = t;
        return;
    }
    var w = t;
    var e = t;
    var s = t;
    var n = t;
    if (i > 0u) { w = src[idx - 1u]; }
    if (i + 1u < nx) { e = src[idx + 1u]; }
    if (j > 0u) { s = src[idx - nx]; }
    if (j + 1u < ny) { n = src[idx + nx]; }
    dst[idx] = t + params.rx * (w - 2.0 * t + e) + params.ry * (s - 2.0 * t + n);
}
