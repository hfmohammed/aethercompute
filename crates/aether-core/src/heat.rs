//! 2D heat diffusion: explicit FTCS 5-point stencil on a unit-square grid.
//!
//! `T' = T + r (T_W + T_E + T_N + T_S - 4T)` with `r = alpha*dt/dx^2`,
//! stable for `r <= 1/4`. Two boundary models:
//! * Dirichlet: the outer ring of cells holds a fixed temperature.
//! * Neumann: insulated; the missing neighbour of an edge cell is the edge
//!   cell itself (zero normal gradient), which conserves total heat exactly in
//!   exact arithmetic.

use bytemuck::{Pod, Zeroable};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::config::{BoundaryMode, ConfigError, ExperimentConfig};

/// Explicit scheme stability bound for equal spacing: alpha*dt/dx^2 <= 1/4.
pub const MAX_DIFFUSION_NUMBER: f32 = 0.25;

/// Uniform parameter block. 32 bytes, matches the WGSL `Params` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct HeatParams {
    pub nx: u32,
    pub ny: u32,
    /// alpha*dt/dx^2
    pub rx: f32,
    /// alpha*dt/dy^2
    pub ry: f32,
    /// 0 = Dirichlet, 1 = Neumann
    pub boundary: u32,
    pub t_boundary: f32,
    pub _pad: [u32; 2],
}

impl HeatParams {
    pub fn from_config(cfg: &ExperimentConfig) -> Result<Self, ConfigError> {
        Self::new(
            cfg.grid,
            cfg.grid,
            cfg.diffusion_number,
            cfg.diffusion_number,
            cfg.boundary,
        )
    }

    pub fn new(
        nx: u32,
        ny: u32,
        rx: f32,
        ry: f32,
        boundary: BoundaryMode,
    ) -> Result<Self, ConfigError> {
        // The 2D explicit bound is rx + ry <= 1/2.
        let r = rx + ry;
        if !(rx > 0.0 && ry > 0.0 && r.is_finite()) || r > 2.0 * MAX_DIFFUSION_NUMBER + 1e-6 {
            return Err(ConfigError::Unstable {
                r: rx.max(ry),
                max: MAX_DIFFUSION_NUMBER,
            });
        }
        Ok(Self {
            nx,
            ny,
            rx,
            ry,
            boundary: boundary.as_u32(),
            t_boundary: 0.0,
            _pad: [0; 2],
        })
    }

    /// Physical timestep implied by the diffusion number on the unit square.
    pub fn dt(&self, alpha: f32) -> f64 {
        let dx = 1.0 / (self.nx as f64 - 1.0);
        self.rx as f64 * dx * dx / alpha as f64
    }
}

/// Deterministic initial field: a few Gaussian hotspots seeded from `seed`.
pub fn initial_grid(cfg: &ExperimentConfig) -> Vec<f32> {
    let n = cfg.grid as usize;
    let params = HeatParams::from_config(cfg).expect("validated config");
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let spots: Vec<(f32, f32, f32, f32)> = (0..3)
        .map(|_| {
            (
                0.2 + 0.6 * rng.random::<f32>(),
                0.2 + 0.6 * rng.random::<f32>(),
                0.04 + 0.08 * rng.random::<f32>(),
                0.6 + 0.4 * rng.random::<f32>(),
            )
        })
        .collect();
    let mut g = vec![0.0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            let x = i as f32 / (n - 1) as f32;
            let y = j as f32 / (n - 1) as f32;
            let mut t = 0.0f32;
            for &(cx, cy, s, a) in &spots {
                let d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                t += a * (-d2 / (2.0 * s * s)).exp();
            }
            g[j * n + i] = t;
        }
    }
    if params.boundary == BoundaryMode::Dirichlet.as_u32() {
        for j in 0..n {
            for i in 0..n {
                if i == 0 || j == 0 || i == n - 1 || j == n - 1 {
                    g[j * n + i] = params.t_boundary;
                }
            }
        }
    }
    g
}

/// Update one cell. Shared by the scalar and Rayon backends.
#[inline]
pub fn cell_update(src: &[f32], i: usize, j: usize, p: &HeatParams) -> f32 {
    let nx = p.nx as usize;
    let ny = p.ny as usize;
    let idx = j * nx + i;
    let t = src[idx];
    let edge = i == 0 || j == 0 || i == nx - 1 || j == ny - 1;
    if edge && p.boundary == 0 {
        return t;
    }
    let w = if i > 0 { src[idx - 1] } else { t };
    let e = if i + 1 < nx { src[idx + 1] } else { t };
    let s = if j > 0 { src[idx - nx] } else { t };
    let n = if j + 1 < ny { src[idx + nx] } else { t };
    t + p.rx * (w - 2.0 * t + e) + p.ry * (s - 2.0 * t + n)
}

/// Total heat (sum of cell values), f64 accumulation.
pub fn total_heat(g: &[f32]) -> f64 {
    g.iter().map(|&v| v as f64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorkloadKind;

    fn cfg(grid: u32, boundary: BoundaryMode) -> ExperimentConfig {
        ExperimentConfig {
            workload: WorkloadKind::Heat,
            grid,
            boundary,
            ..Default::default()
        }
    }

    fn step(g: &[f32], p: &HeatParams) -> Vec<f32> {
        let (nx, ny) = (p.nx as usize, p.ny as usize);
        let mut out = vec![0.0; g.len()];
        for j in 0..ny {
            for i in 0..nx {
                out[j * nx + i] = cell_update(g, i, j, p);
            }
        }
        out
    }

    #[test]
    fn layout() {
        assert_eq!(std::mem::size_of::<HeatParams>(), 32);
        assert_eq!(std::mem::offset_of!(HeatParams, boundary), 16);
    }

    #[test]
    fn stability_guard() {
        assert!(HeatParams::new(8, 8, 0.25, 0.25, BoundaryMode::Neumann).is_ok());
        assert!(HeatParams::new(8, 8, 0.26, 0.26, BoundaryMode::Neumann).is_err());
        assert!(HeatParams::new(8, 8, 0.0, 0.1, BoundaryMode::Neumann).is_err());
        assert!(HeatParams::new(8, 8, f32::NAN, 0.1, BoundaryMode::Neumann).is_err());
        // anisotropic: rx + ry <= 1/2 is the real 2D bound
        assert!(HeatParams::new(8, 8, 0.4, 0.1, BoundaryMode::Neumann).is_ok());
        assert!(HeatParams::new(8, 8, 0.4, 0.2, BoundaryMode::Neumann).is_err());
    }

    #[test]
    fn dt_matches_definition() {
        let p = HeatParams::new(101, 101, 0.2, 0.2, BoundaryMode::Neumann).unwrap();
        assert!((p.dt(2.0) - 0.2 * 1e-4 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn init_deterministic_and_seeded() {
        let c = cfg(32, BoundaryMode::Neumann);
        assert_eq!(initial_grid(&c), initial_grid(&c));
        let d = ExperimentConfig {
            seed: 7,
            ..c.clone()
        };
        assert_ne!(initial_grid(&c), initial_grid(&d));
    }

    #[test]
    fn dirichlet_init_zeroes_boundary() {
        let g = initial_grid(&cfg(32, BoundaryMode::Dirichlet));
        for k in 0..32 {
            assert_eq!(g[k], 0.0);
            assert_eq!(g[31 * 32 + k], 0.0);
            assert_eq!(g[k * 32], 0.0);
            assert_eq!(g[k * 32 + 31], 0.0);
        }
    }

    #[test]
    fn uniform_field_is_steady() {
        for b in [BoundaryMode::Dirichlet, BoundaryMode::Neumann] {
            let p = HeatParams::from_config(&cfg(16, b)).unwrap();
            let g = vec![0.0f32; 256];
            assert_eq!(step(&g, &p), g);
            let g = vec![1.0f32; 256];
            // Dirichlet boundary is 0 but held; interior of a uniform 1.0 field
            // only changes next to the boundary, so check Neumann strictly.
            if b == BoundaryMode::Neumann {
                assert_eq!(step(&g, &p), g);
            }
        }
    }

    #[test]
    fn neumann_conserves_heat() {
        let c = cfg(48, BoundaryMode::Neumann);
        let p = HeatParams::from_config(&c).unwrap();
        let mut g = initial_grid(&c);
        let h0 = total_heat(&g);
        for _ in 0..500 {
            g = step(&g, &p);
        }
        assert!((total_heat(&g) - h0).abs() / h0 < 1e-4);
    }

    #[test]
    fn dirichlet_loses_heat_monotonically() {
        let c = cfg(48, BoundaryMode::Dirichlet);
        let p = HeatParams::from_config(&c).unwrap();
        let mut g = initial_grid(&c);
        let mut prev = total_heat(&g);
        for _ in 0..300 {
            g = step(&g, &p);
            let h = total_heat(&g);
            assert!(h <= prev + 1e-4);
            prev = h;
        }
    }

    #[test]
    fn maximum_principle() {
        let c = cfg(40, BoundaryMode::Neumann);
        let p = HeatParams::from_config(&c).unwrap();
        let mut g = initial_grid(&c);
        let (lo, hi) = g
            .iter()
            .fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        for _ in 0..200 {
            g = step(&g, &p);
            for &v in &g {
                assert!(v >= lo - 1e-6 && v <= hi + 1e-6);
            }
        }
    }

    #[test]
    fn dirichlet_boundary_is_held() {
        let c = cfg(24, BoundaryMode::Dirichlet);
        let p = HeatParams::from_config(&c).unwrap();
        let mut g = initial_grid(&c);
        for _ in 0..50 {
            g = step(&g, &p);
        }
        assert_eq!(g[0], 0.0);
        assert_eq!(g[24 * 24 - 1], 0.0);
    }

    #[test]
    fn diffuses_a_spike_symmetrically() {
        let p = HeatParams::new(9, 9, 0.2, 0.2, BoundaryMode::Neumann).unwrap();
        let mut g = vec![0.0f32; 81];
        g[4 * 9 + 4] = 1.0;
        let g = step(&g, &p);
        let c = g[4 * 9 + 4];
        assert!((c - (1.0 - 4.0 * 0.2)).abs() < 1e-6);
        for idx in [3 * 9 + 4, 5 * 9 + 4, 4 * 9 + 3, 4 * 9 + 5] {
            assert!((g[idx] - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn converges_to_mean_under_neumann() {
        let c = cfg(16, BoundaryMode::Neumann);
        let p = HeatParams::from_config(&c).unwrap();
        let mut g = initial_grid(&c);
        let mean = total_heat(&g) / 256.0;
        for _ in 0..20_000 {
            g = step(&g, &p);
        }
        for &v in &g {
            assert!((v as f64 - mean).abs() < 1e-3);
        }
    }
}
