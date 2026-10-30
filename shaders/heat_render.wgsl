// Full-screen heatmap sampled directly from the simulation's storage buffer.

struct View {
    center: vec2<f32>,
    zoom: f32,
    aspect: f32,
    point_size: f32,
    vmax: f32,
    grid: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<uniform> view: View;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VsOut;
    o.clip = vec4<f32>(p[vi], 0.0, 1.0);
    o.ndc = p[vi];
    return o;
}

// Compact "inferno"-like polynomial colour map.
fn inferno(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.0002, 0.0016, -0.0194);
    let c1 = vec3<f32>(0.1065, 0.5639, 3.9327);
    let c2 = vec3<f32>(11.6024, -3.9728, -15.9423);
    let c3 = vec3<f32>(-41.7039, 17.4363, 44.3541);
    let c4 = vec3<f32>(77.1629, -33.4023, -81.8073);
    let c5 = vec3<f32>(-71.3194, 32.6260, 73.2095);
    let c6 = vec3<f32>(25.1311, -12.2426, -23.0703);
    return clamp(c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6))))), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Map the screen to world space [-1,1]^2 (square cells), then to the grid.
    var w = in.ndc;
    w.x = w.x * view.aspect;
    w = w / view.zoom + view.center;
    let uv = w * 0.5 + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x >= 1.0 || uv.y >= 1.0) {
        return vec4<f32>(0.02, 0.02, 0.03, 1.0);
    }
    let n = view.grid;
    let ix = min(u32(uv.x * f32(n)), n - 1u);
    let iy = min(u32(uv.y * f32(n)), n - 1u);
    let t = field[iy * n + ix] / view.vmax;
    return vec4<f32>(inferno(t), 1.0);
}
