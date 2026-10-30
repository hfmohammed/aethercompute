// O(N) independent particle update: harmonic well + swirl, drag, wall bounce.
// One invocation reads one particle from `src` and writes one to `dst`.
// Mirrors `aether_core::particles::local_update` operation for operation.

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    flags: u32,
    pad: vec2<f32>,
}

struct Params {
    dt: f32,
    damping: f32,
    force_scale: f32,
    softening: f32,
    half_w: f32,
    half_h: f32,
    restitution: f32,
    swirl: f32,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

override WG: u32 = 64;

@group(0) @binding(0) var<storage, read> src: array<Particle>;
@group(0) @binding(1) var<storage, read_write> dst: array<Particle>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // Dispatches larger than the per-dimension limit are folded into y.
    let i = gid.y * (nwg.x * WG) + gid.x;
    if (i >= params.count) {
        return;
    }
    let p = src[i];
    let ax = -params.force_scale * p.pos.x - params.swirl * p.pos.y;
    let ay = -params.force_scale * p.pos.y + params.swirl * p.pos.x;
    let drag = 1.0 - params.damping * params.dt;
    var vx = (p.vel.x + ax * params.dt) * drag;
    var vy = (p.vel.y + ay * params.dt) * drag;
    var x = p.pos.x + vx * params.dt;
    var y = p.pos.y + vy * params.dt;
    if (x > params.half_w) {
        x = params.half_w;
        vx = -vx * params.restitution;
    } else if (x < -params.half_w) {
        x = -params.half_w;
        vx = -vx * params.restitution;
    }
    if (y > params.half_h) {
        y = params.half_h;
        vy = -vy * params.restitution;
    } else if (y < -params.half_h) {
        y = -params.half_h;
        vy = -vy * params.restitution;
    }
    var out = p;
    out.pos = vec2<f32>(x, y);
    out.vel = vec2<f32>(vx, vy);
    dst[i] = out;
}
