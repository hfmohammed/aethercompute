use bytemuck::cast_slice;
use thiserror::Error;

use crate::config::ExperimentConfig;
use crate::particles::Particle;
use crate::validate::{compare, ErrorReport, Tolerance};

#[derive(Debug, Error)]
pub enum WorkloadError {
    #[error("{0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("GPU error: {0}")]
    Gpu(String),
    #[error("{0}")]
    Other(String),
}

/// A copy of simulation state on the host.
#[derive(Debug, Clone, PartialEq)]
pub enum Snapshot {
    Particles(Vec<Particle>),
    Heat(Vec<f32>),
}

impl Snapshot {
    /// Flat `f32` view used for error metrics. Particle snapshots compare
    /// position and velocity (mass is constant, flags are integers).
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Snapshot::Particles(ps) => ps
                .iter()
                .flat_map(|p| [p.pos[0], p.pos[1], p.vel[0], p.vel[1]])
                .collect(),
            Snapshot::Heat(g) => g.clone(),
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Snapshot::Particles(ps) => cast_slice(ps),
            Snapshot::Heat(g) => cast_slice(g),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Snapshot::Particles(p) => p.len(),
            Snapshot::Heat(g) => g.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Compare against a reference snapshot of the same kind and size.
    pub fn compare(
        &self,
        reference: &Snapshot,
        tol: Tolerance,
    ) -> Result<ErrorReport, WorkloadError> {
        let same = matches!(
            (self, reference),
            (Snapshot::Particles(_), Snapshot::Particles(_))
                | (Snapshot::Heat(_), Snapshot::Heat(_))
        );
        if !same || self.len() != reference.len() {
            return Err(WorkloadError::Other(
                "cannot compare snapshots of different kind or size".into(),
            ));
        }
        Ok(compare(&self.to_f32(), &reference.to_f32(), tol))
    }
}

/// A runnable workload on some execution backend.
///
/// `step` may only *enqueue* work (the GPU backend does); call `sync` to wait
/// for completion. That split is what lets benchmarks separate dispatch time
/// from readback (`snapshot`) time.
pub trait Workload {
    /// Backend/adapter description, e.g. `gpu (Apple M2, Metal)`.
    fn describe(&self) -> String;

    fn config(&self) -> &ExperimentConfig;

    /// (Re)load the deterministic initial state from the config's seed.
    fn initialize(&mut self) -> Result<(), WorkloadError>;

    /// Advance `steps` timesteps.
    fn step(&mut self, steps: u32) -> Result<(), WorkloadError>;

    /// Block until all enqueued steps have finished executing.
    fn sync(&mut self) -> Result<(), WorkloadError>;

    /// Copy current state to the host (implies `sync`).
    fn snapshot(&mut self) -> Result<Snapshot, WorkloadError>;

    /// Compare current state against a reference snapshot.
    fn validate(
        &mut self,
        reference: &Snapshot,
        tol: Tolerance,
    ) -> Result<ErrorReport, WorkloadError> {
        self.snapshot()?.compare(reference, tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_particles() {
        let s = Snapshot::Particles(vec![Particle::new([1.0, 2.0], [3.0, 4.0], 1.0)]);
        assert_eq!(s.to_f32(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.as_bytes().len(), 32);
    }

    #[test]
    fn compare_rejects_mismatch() {
        let a = Snapshot::Heat(vec![0.0; 4]);
        let b = Snapshot::Heat(vec![0.0; 5]);
        assert!(a.compare(&b, Tolerance::EXACT).is_err());
        let c = Snapshot::Particles(vec![]);
        assert!(a.compare(&c, Tolerance::EXACT).is_err());
    }

    #[test]
    fn compare_heat() {
        let a = Snapshot::Heat(vec![1.0, 2.0]);
        assert!(a.compare(&a.clone(), Tolerance::EXACT).unwrap().passed);
    }
}
