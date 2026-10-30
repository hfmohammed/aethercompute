// Exact all-pairs softened 2D gravity, O(N^2). One thread per target particle,
// global-memory reads (the simple baseline; tiling in workgroup memory is a
// documented follow-up optimisation). Mirrors `nbody_update` in aether-core.

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
    let i = gid.y * (nwg.x * WG) + gid.x;
    if (i >= params.count) {
        return;
    }
    let p = src[i];
    let eps2 = params.softening * params.softening;
    var ax = 0.0;
    var ay = 0.0;
    for (var j = 0u; j < params.count; j = j + 1u) {
        let o = src[j];
        let dx = o.pos.x - p.pos.x;
        let dy = o.pos.y - p.pos.y;
        let r2 = dx * dx + dy * dy + eps2;
        let inv = 1.0 / (r2 * sqrt(r2));
        let s = o.mass * inv;
        ax = ax + dx * s;
        ay = ay + dy * s;
    }
    ax = ax * params.force_scale;
    ay = ay * params.force_scale;

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
