// Instanced quads straight from the simulation's storage buffer (no readback).
// Additive blending turns overdraw into density; colour encodes speed.

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    flags: u32,
    pad: vec2<f32>,
}

struct View {
    center: vec2<f32>,
    zoom: f32,
    aspect: f32,
    point_size: f32,
    vmax: f32,
    grid: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> view: View;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) speed: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi];
    let p = particles[ii];
    var ndc = (p.pos - view.center) * view.zoom;
    ndc.x = ndc.x / view.aspect;
    var off = c * view.point_size;
    off.x = off.x / view.aspect;
    var o: VsOut;
    o.clip = vec4<f32>(ndc + off, 0.0, 1.0);
    o.local = c;
    o.speed = length(p.vel) / view.vmax;
    return o;
}

fn ramp(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let a = vec3<f32>(0.10, 0.35, 1.00);
    let b = vec3<f32>(0.15, 0.95, 0.85);
    let c = vec3<f32>(1.00, 0.75, 0.20);
    let d = vec3<f32>(1.00, 0.25, 0.35);
    if (x < 0.5) {
        return mix(a, b, x * 2.0);
    }
    return mix(b, mix(c, d, clamp((x - 0.75) * 4.0, 0.0, 1.0)), clamp((x - 0.5) * 4.0, 0.0, 1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.local, in.local);
    if (r2 > 1.0) {
        discard;
    }
    let fall = (1.0 - r2);
    return vec4<f32>(ramp(in.speed) * fall * 0.35, 1.0);
}
