//! Particle data model, deterministic initialisation and reference kernels.
//!
//! Both the scalar and Rayon CPU backends call exactly these functions, and the
//! WGSL shaders mirror them operation for operation, so CPU/GPU differences are
//! limited to floating-point contraction/rounding.

use bytemuck::{Pod, Zeroable};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::config::{ExperimentConfig, WorkloadKind};

/// One particle. 32 bytes, matches the WGSL `Particle` struct
/// (`vec2<f32>, vec2<f32>, f32, u32, vec2<f32>` padding keeps the size a
/// multiple of 16 so array stride is identical on host and device).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Particle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub mass: f32,
    pub flags: u32,
    pub _pad: [f32; 2],
}

impl Particle {
    pub fn new(pos: [f32; 2], vel: [f32; 2], mass: f32) -> Self {
        Self {
            pos,
            vel,
            mass,
            flags: 0,
            _pad: [0.0; 2],
        }
    }
}

/// Uniform parameter block. 48 bytes, matches the WGSL `Params` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ParticleParams {
    pub dt: f32,
    /// Linear drag coefficient; velocity is scaled by `1 - damping*dt` per step.
    pub damping: f32,
    /// Spring constant (local mode) or gravitational constant G (all-pairs).
    pub force_scale: f32,
    pub softening: f32,
    pub half_w: f32,
    pub half_h: f32,
    /// Velocity retained on a wall bounce.
    pub restitution: f32,
    /// Tangential force coefficient of the local mode.
    pub swirl: f32,
    pub count: u32,
    pub _pad: [u32; 3],
}

impl ParticleParams {
    pub fn from_config(cfg: &ExperimentConfig) -> Self {
        let nbody = cfg.workload == WorkloadKind::NbodyAllpairs;
        let default_dt = if nbody { 0.002 } else { 0.004 };
        let default_force = if nbody {
            // Keeps total acceleration O(1) as N grows.
            1.0 / cfg.count.max(1) as f32
        } else {
            1.5
        };
        Self {
            dt: cfg.dt.unwrap_or(default_dt),
            damping: cfg.damping,
            force_scale: cfg.force_scale.unwrap_or(default_force),
            softening: cfg.softening,
            half_w: 1.0,
            half_h: 1.0,
            restitution: 0.8,
            swirl: if nbody { 0.0 } else { 0.6 },
            count: cfg.count,
            _pad: [0; 3],
        }
    }
}

/// Deterministic initial state for a seed. Same seed => identical bytes.
pub fn initial_particles(cfg: &ExperimentConfig) -> Vec<Particle> {
    let p = ParticleParams::from_config(cfg);
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    (0..cfg.count)
        .map(|_| {
            let x = (rng.random::<f32>() * 2.0 - 1.0) * 0.8 * p.half_w;
            let y = (rng.random::<f32>() * 2.0 - 1.0) * 0.8 * p.half_h;
            let vx = (rng.random::<f32>() * 2.0 - 1.0) * 0.3;
            let vy = (rng.random::<f32>() * 2.0 - 1.0) * 0.3;
            let m = 0.5 + rng.random::<f32>();
            Particle::new([x, y], [vx, vy], m)
        })
        .collect()
}

/// Reflect a particle off the world bounds.
#[inline]
pub fn apply_bounds(
    mut pos: [f32; 2],
    mut vel: [f32; 2],
    p: &ParticleParams,
) -> ([f32; 2], [f32; 2]) {
    if pos[0] > p.half_w {
        pos[0] = p.half_w;
        vel[0] = -vel[0] * p.restitution;
    } else if pos[0] < -p.half_w {
        pos[0] = -p.half_w;
        vel[0] = -vel[0] * p.restitution;
    }
    if pos[1] > p.half_h {
        pos[1] = p.half_h;
        vel[1] = -vel[1] * p.restitution;
    } else if pos[1] < -p.half_h {
        pos[1] = -p.half_h;
        vel[1] = -vel[1] * p.restitution;
    }
    (pos, vel)
}

/// Semi-implicit Euler: v' = (v + a dt)(1 - c dt); x' = x + v' dt; then bounds.
#[inline]
fn integrate(src: &Particle, acc: [f32; 2], p: &ParticleParams) -> Particle {
    let drag = 1.0 - p.damping * p.dt;
    let vx = (src.vel[0] + acc[0] * p.dt) * drag;
    let vy = (src.vel[1] + acc[1] * p.dt) * drag;
    let (pos, vel) = apply_bounds(
        [src.pos[0] + vx * p.dt, src.pos[1] + vy * p.dt],
        [vx, vy],
        p,
    );
    Particle { pos, vel, ..*src }
}

/// O(1) per-particle update: harmonic well toward the origin plus a swirl.
#[inline]
pub fn local_update(src: &Particle, p: &ParticleParams) -> Particle {
    let ax = -p.force_scale * src.pos[0] - p.swirl * src.pos[1];
    let ay = -p.force_scale * src.pos[1] + p.swirl * src.pos[0];
    integrate(src, [ax, ay], p)
}

/// O(N) per-particle update: softened 2D gravity from every particle.
#[inline]
pub fn nbody_update(i: usize, all: &[Particle], p: &ParticleParams) -> Particle {
    let src = &all[i];
    let eps2 = p.softening * p.softening;
    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    for other in all {
        let dx = other.pos[0] - src.pos[0];
        let dy = other.pos[1] - src.pos[1];
        let r2 = dx * dx + dy * dy + eps2;
        let inv = 1.0 / (r2 * r2.sqrt());
        let s = other.mass * inv;
        ax += dx * s;
        ay += dy * s;
    }
    integrate(src, [ax * p.force_scale, ay * p.force_scale], p)
}

/// Total momentum (f64 accumulation for diagnostics).
pub fn total_momentum(ps: &[Particle]) -> [f64; 2] {
    ps.iter().fold([0.0; 2], |a, p| {
        [
            a[0] + p.mass as f64 * p.vel[0] as f64,
            a[1] + p.mass as f64 * p.vel[1] as f64,
        ]
    })
}

pub fn kinetic_energy(ps: &[Particle]) -> f64 {
    ps.iter()
        .map(|p| {
            0.5 * p.mass as f64 * (p.vel[0] as f64).powi(2)
                + 0.5 * p.mass as f64 * (p.vel[1] as f64).powi(2)
        })
        .sum()
}

/// Softened gravitational potential energy, O(N^2) in f64.
pub fn potential_energy(ps: &[Particle], g: f64, softening: f64) -> f64 {
    let eps2 = softening * softening;
    let mut u = 0.0;
    for i in 0..ps.len() {
        for j in (i + 1)..ps.len() {
            let dx = ps[j].pos[0] as f64 - ps[i].pos[0] as f64;
            let dy = ps[j].pos[1] as f64 - ps[i].pos[1] as f64;
            u -= g * ps[i].mass as f64 * ps[j].mass as f64 / (dx * dx + dy * dy + eps2).sqrt();
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(count: u32) -> ExperimentConfig {
        ExperimentConfig {
            count,
            ..Default::default()
        }
    }

    #[test]
    fn layout_sizes() {
        assert_eq!(std::mem::size_of::<Particle>(), 32);
        assert_eq!(std::mem::align_of::<Particle>(), 4);
        assert_eq!(std::mem::size_of::<ParticleParams>(), 48);
        assert_eq!(std::mem::size_of::<ParticleParams>() % 16, 0);
        assert_eq!(std::mem::offset_of!(Particle, vel), 8);
        assert_eq!(std::mem::offset_of!(Particle, mass), 16);
        assert_eq!(std::mem::offset_of!(ParticleParams, count), 32);
    }

    #[test]
    fn init_is_deterministic() {
        assert_eq!(initial_particles(&cfg(100)), initial_particles(&cfg(100)));
    }

    #[test]
    fn init_depends_on_seed() {
        let a = initial_particles(&cfg(100));
        let b = initial_particles(&ExperimentConfig {
            seed: 43,
            ..cfg(100)
        });
        assert_ne!(a, b);
    }

    #[test]
    fn init_within_bounds_and_finite() {
        for p in initial_particles(&cfg(5000)) {
            assert!(p.pos[0].abs() <= 0.8 && p.pos[1].abs() <= 0.8);
            assert!(p.mass >= 0.5 && p.mass <= 1.5);
            assert!(p.vel.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn zero_force_is_ballistic() {
        let c = ExperimentConfig {
            force_scale: Some(0.0),
            damping: 0.0,
            ..cfg(1)
        };
        let mut p = ParticleParams::from_config(&c);
        p.swirl = 0.0;
        let s = Particle::new([0.1, -0.2], [0.5, 0.25], 1.0);
        let n = local_update(&s, &p);
        assert!((n.pos[0] - (0.1 + 0.5 * p.dt)).abs() < 1e-7);
        assert!((n.pos[1] - (-0.2 + 0.25 * p.dt)).abs() < 1e-7);
        assert_eq!(n.vel, s.vel);
    }

    #[test]
    fn wall_reflection() {
        let mut p = ParticleParams::from_config(&cfg(1));
        p.force_scale = 0.0;
        p.swirl = 0.0;
        p.damping = 0.0;
        let s = Particle::new([0.999, 0.0], [10.0, 0.0], 1.0);
        let n = local_update(&s, &p);
        assert_eq!(n.pos[0], p.half_w);
        assert!((n.vel[0] + 10.0 * p.restitution).abs() < 1e-5);
        let s = Particle::new([0.0, -0.999], [0.0, -10.0], 1.0);
        let n = local_update(&s, &p);
        assert_eq!(n.pos[1], -p.half_h);
        assert!(n.vel[1] > 0.0);
    }

    #[test]
    fn damping_reduces_speed() {
        let mut p = ParticleParams::from_config(&cfg(1));
        p.force_scale = 0.0;
        p.swirl = 0.0;
        p.damping = 1.0;
        let s = Particle::new([0.0, 0.0], [1.0, 0.0], 1.0);
        let n = local_update(&s, &p);
        assert!(n.vel[0] < 1.0 && n.vel[0] > 0.0);
    }

    #[test]
    fn local_stays_bounded_and_finite() {
        let c = cfg(256);
        let p = ParticleParams::from_config(&c);
        let mut ps = initial_particles(&c);
        for _ in 0..2000 {
            ps = ps.iter().map(|s| local_update(s, &p)).collect();
        }
        for s in &ps {
            assert!(s.pos.iter().all(|x| x.is_finite() && x.abs() <= 1.0 + 1e-6));
            assert!(s.vel.iter().all(|x| x.is_finite()));
        }
    }

    fn nbody_cfg(count: u32) -> ExperimentConfig {
        ExperimentConfig {
            workload: WorkloadKind::NbodyAllpairs,
            damping: 0.0,
            ..cfg(count)
        }
    }

    #[test]
    fn nbody_pair_symmetry() {
        let c = nbody_cfg(2);
        let mut p = ParticleParams::from_config(&c);
        p.force_scale = 1.0;
        let ps = vec![
            Particle::new([-0.3, 0.0], [0.0; 2], 1.0),
            Particle::new([0.3, 0.0], [0.0; 2], 1.0),
        ];
        let a = nbody_update(0, &ps, &p);
        let b = nbody_update(1, &ps, &p);
        assert!(a.vel[0] > 0.0 && b.vel[0] < 0.0, "bodies attract");
        assert!((a.vel[0] + b.vel[0]).abs() < 1e-6);
        assert!(a.vel[1].abs() < 1e-9 && b.vel[1].abs() < 1e-9);
    }

    #[test]
    fn nbody_self_force_is_zero() {
        let c = nbody_cfg(1);
        let p = ParticleParams::from_config(&c);
        let ps = vec![Particle::new([0.1, 0.1], [0.0; 2], 1.0)];
        let n = nbody_update(0, &ps, &p);
        assert_eq!(n.vel, [0.0, 0.0]);
    }

    #[test]
    fn nbody_softening_bounds_force() {
        let c = nbody_cfg(2);
        let mut p = ParticleParams::from_config(&c);
        p.force_scale = 1.0;
        p.softening = 0.1;
        let ps = vec![
            Particle::new([0.0, 0.0], [0.0; 2], 1.0),
            Particle::new([1e-6, 0.0], [0.0; 2], 1.0),
        ];
        let n = nbody_update(0, &ps, &p);
        assert!(n.vel[0].is_finite() && n.vel[0].abs() < 1.0);
    }

    #[test]
    fn nbody_conserves_momentum() {
        let c = nbody_cfg(64);
        let mut p = ParticleParams::from_config(&c);
        p.half_w = 100.0;
        p.half_h = 100.0;
        let mut ps = initial_particles(&c);
        let before = total_momentum(&ps);
        for _ in 0..200 {
            let prev = ps.clone();
            ps = (0..prev.len())
                .map(|i| nbody_update(i, &prev, &p))
                .collect();
        }
        let after = total_momentum(&ps);
        let scale = ps.iter().map(|q| q.mass as f64).sum::<f64>();
        assert!(
            (after[0] - before[0]).abs() / scale < 1e-4,
            "{before:?} {after:?}"
        );
        assert!((after[1] - before[1]).abs() / scale < 1e-4);
    }

    #[test]
    fn nbody_energy_drift_is_small() {
        let c = nbody_cfg(48);
        let mut p = ParticleParams::from_config(&c);
        p.half_w = 100.0;
        p.half_h = 100.0;
        let mut ps = initial_particles(&c);
        let g = p.force_scale as f64;
        let e0 = kinetic_energy(&ps) + potential_energy(&ps, g, p.softening as f64);
        for _ in 0..300 {
            let prev = ps.clone();
            ps = (0..prev.len())
                .map(|i| nbody_update(i, &prev, &p))
                .collect();
        }
        let e1 = kinetic_energy(&ps) + potential_energy(&ps, g, p.softening as f64);
        assert!(
            ((e1 - e0) / e0.abs().max(1e-9)).abs() < 0.05,
            "e0={e0} e1={e1}"
        );
    }
}
