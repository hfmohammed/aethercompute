use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::heat::{HeatParams, MAX_DIFFUSION_NUMBER};
use crate::particles::ParticleParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkloadKind {
    /// O(N) independent / local-force particle update (high-scale baseline).
    ParticlesLocal,
    /// Exact all-pairs softened gravity, O(N^2). Small N only.
    NbodyAllpairs,
    /// Explicit 5-point 2D heat diffusion.
    Heat,
}

impl WorkloadKind {
    pub const ALL: [WorkloadKind; 3] = [Self::ParticlesLocal, Self::NbodyAllpairs, Self::Heat];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ParticlesLocal => "particles-local",
            Self::NbodyAllpairs => "nbody-allpairs",
            Self::Heat => "heat",
        }
    }

    pub fn is_particles(self) -> bool {
        !matches!(self, Self::Heat)
    }

    /// Asymptotic cost per step, for labelling results.
    pub fn complexity(self) -> &'static str {
        match self {
            Self::ParticlesLocal => "O(N)",
            Self::NbodyAllpairs => "O(N^2)",
            Self::Heat => "O(cells)",
        }
    }
}

impl std::str::FromStr for WorkloadKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "particles-local" | "particles" | "local" => Ok(Self::ParticlesLocal),
            "nbody-allpairs" | "nbody" | "n-body" => Ok(Self::NbodyAllpairs),
            "heat" => Ok(Self::Heat),
            other => Err(format!(
                "unknown workload '{other}' (expected particles-local, nbody-allpairs or heat)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    Scalar,
    Rayon,
    Gpu,
}

impl BackendKind {
    pub const ALL: [BackendKind; 3] = [Self::Scalar, Self::Rayon, Self::Gpu];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Rayon => "rayon",
            Self::Gpu => "gpu",
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "scalar" | "cpu" => Ok(Self::Scalar),
            "rayon" => Ok(Self::Rayon),
            "gpu" | "wgpu" => Ok(Self::Gpu),
            other => Err(format!(
                "unknown backend '{other}' (expected scalar, rayon or gpu)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoundaryMode {
    /// Fixed temperature on the outer ring of cells.
    Dirichlet,
    /// Insulated: zero normal gradient (ghost cell equals edge cell).
    Neumann,
}

impl BoundaryMode {
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Dirichlet => 0,
            Self::Neumann => 1,
        }
    }
}

impl std::str::FromStr for BoundaryMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "dirichlet" | "fixed" => Ok(Self::Dirichlet),
            "neumann" | "insulated" => Ok(Self::Neumann),
            other => Err(format!(
                "unknown boundary '{other}' (expected dirichlet or neumann)"
            )),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum ConfigError {
    #[error("{0}")]
    Invalid(String),
    #[error("unstable heat timestep: diffusion number alpha*dt/dx^2 = {r} must be in (0, {max}]")]
    Unstable { r: f32, max: f32 },
}

/// Everything needed to reproduce a run on a given machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub workload: WorkloadKind,
    pub backend: BackendKind,
    /// Particle count (particle workloads).
    pub count: u32,
    /// Grid edge length in cells (heat); the grid is `grid x grid`.
    pub grid: u32,
    /// Simulation steps per measured sample.
    pub steps: u32,
    pub warmup: u32,
    pub repeats: u32,
    pub seed: u64,
    /// Particle timestep. `None` selects a per-workload default.
    pub dt: Option<f32>,
    pub softening: f32,
    /// Force scale / gravitational constant. `None` selects a default.
    pub force_scale: Option<f32>,
    pub damping: f32,
    /// GPU workgroup size (threads per workgroup).
    pub workgroup_size: u32,
    pub boundary: BoundaryMode,
    /// Heat diffusion number alpha*dt/dx^2 (stable for <= 0.25).
    pub diffusion_number: f32,
    /// Heat diffusivity alpha.
    pub alpha: f32,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            workload: WorkloadKind::ParticlesLocal,
            backend: BackendKind::Scalar,
            count: 10_000,
            grid: 256,
            steps: 100,
            warmup: 3,
            repeats: 20,
            seed: 42,
            dt: None,
            softening: 0.05,
            force_scale: None,
            damping: 0.05,
            workgroup_size: 64,
            boundary: BoundaryMode::Dirichlet,
            diffusion_number: 0.2,
            alpha: 1.0,
        }
    }
}

impl ExperimentConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: &str| Err(ConfigError::Invalid(m.to_string()));
        if self.workload.is_particles() {
            if self.count == 0 {
                return bad("count must be >= 1");
            }
            if self.count > 50_000_000 {
                return bad("count exceeds the 50M safety cap");
            }
            if self.workload == WorkloadKind::NbodyAllpairs && self.count > 65_536 {
                return bad("nbody-allpairs is O(N^2); count is capped at 65536 (use 256-4096 for benchmarks)");
            }
            if !(self.softening >= 0.0 && self.softening.is_finite()) {
                return bad("softening must be finite and >= 0");
            }
            if let Some(dt) = self.dt {
                if !(dt > 0.0 && dt.is_finite()) {
                    return bad("dt must be finite and > 0");
                }
            }
        } else {
            if self.grid < 3 {
                return bad("grid must be >= 3");
            }
            if self.grid > 16_384 {
                return bad("grid exceeds the 16384 safety cap");
            }
            HeatParams::from_config(self)?;
        }
        if !self.workgroup_size.is_power_of_two() || self.workgroup_size > 1024 {
            return bad("workgroup_size must be a power of two <= 1024");
        }
        if !(self.damping >= 0.0 && self.damping.is_finite()) {
            return bad("damping must be finite and >= 0");
        }
        Ok(())
    }

    /// Particle uniform block for the configured workload.
    pub fn particle_params(&self) -> ParticleParams {
        ParticleParams::from_config(self)
    }

    /// Heat uniform block (validates the stability condition).
    pub fn heat_params(&self) -> Result<HeatParams, ConfigError> {
        HeatParams::from_config(self)
    }

    /// Human-readable problem size, e.g. `100000 particles` / `512x512`.
    pub fn size_label(&self) -> String {
        if self.workload.is_particles() {
            format!("{} particles", self.count)
        } else {
            format!("{0}x{0}", self.grid)
        }
    }

    /// Work units per step (particle updates, or cell updates).
    pub fn updates_per_step(&self) -> u64 {
        if self.workload.is_particles() {
            self.count as u64
        } else {
            self.grid as u64 * self.grid as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_names() {
        assert_eq!("heat".parse::<WorkloadKind>().unwrap(), WorkloadKind::Heat);
        assert_eq!("cpu".parse::<BackendKind>().unwrap(), BackendKind::Scalar);
        assert_eq!(
            "insulated".parse::<BoundaryMode>().unwrap(),
            BoundaryMode::Neumann
        );
        assert!("nope".parse::<WorkloadKind>().is_err());
    }

    #[test]
    fn default_is_valid() {
        ExperimentConfig::default().validate().unwrap();
    }

    #[test]
    fn rejects_zero_count_and_big_nbody() {
        let mut c = ExperimentConfig::default();
        c.count = 0;
        assert!(c.validate().is_err());
        c.workload = WorkloadKind::NbodyAllpairs;
        c.count = 1_000_000;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_unstable_heat() {
        let c = ExperimentConfig {
            workload: WorkloadKind::Heat,
            diffusion_number: 0.3,
            ..Default::default()
        };
        assert!(matches!(c.validate(), Err(ConfigError::Unstable { .. })));
    }

    #[test]
    fn rejects_bad_workgroup() {
        let c = ExperimentConfig {
            workgroup_size: 100,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_json_roundtrip() {
        let c = ExperimentConfig::default();
        let s = serde_json::to_string(&c).unwrap();
        let d: ExperimentConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn labels() {
        let mut c = ExperimentConfig::default();
        assert_eq!(c.size_label(), "10000 particles");
        c.workload = WorkloadKind::Heat;
        assert_eq!(c.size_label(), "256x256");
        assert_eq!(c.updates_per_step(), 65536);
    }
}
