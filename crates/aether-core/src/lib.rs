//! Backend-independent data models, deterministic scenario generation,
//! reference kernels and validation for AetherCompute.
//!
//! Nothing in this crate touches a GPU or a window, so the exact same
//! scenarios run headlessly, in tests and in the interactive viewer.

pub mod config;
pub mod heat;
pub mod particles;
pub mod stats;
pub mod validate;
pub mod workload;

pub use config::{
    BackendKind, BoundaryMode, ConfigError, ExperimentConfig, PassMode, WorkloadKind,
};
pub use heat::{HeatParams, MAX_DIFFUSION_NUMBER};
pub use particles::{Particle, ParticleParams};
pub use stats::Summary;
pub use validate::{ErrorReport, Tolerance};
pub use workload::{Snapshot, Workload, WorkloadError};
