//! CPU reference backends: single-threaded scalar and Rayon-parallel.
//!
//! Both run the identical per-element kernels from `aether-core`, so Rayon
//! results are bit-identical to scalar results (each output element is
//! computed independently from the previous step's buffer).

use aether_core::{
    heat::{self, HeatParams},
    particles::{self, Particle, ParticleParams},
    BackendKind, ExperimentConfig, Snapshot, Workload, WorkloadError, WorkloadKind,
};
use rayon::prelude::*;

enum State {
    Particles {
        params: ParticleParams,
        cur: Vec<Particle>,
        next: Vec<Particle>,
    },
    Heat {
        params: HeatParams,
        cur: Vec<f32>,
        next: Vec<f32>,
    },
}

pub struct CpuWorkload {
    cfg: ExperimentConfig,
    parallel: bool,
    state: State,
}

impl CpuWorkload {
    /// `cfg.backend` selects scalar vs Rayon; `Gpu` is rejected.
    pub fn new(cfg: ExperimentConfig) -> Result<Self, WorkloadError> {
        cfg.validate()?;
        let parallel = match cfg.backend {
            BackendKind::Scalar => false,
            BackendKind::Rayon => true,
            BackendKind::Gpu => {
                return Err(WorkloadError::Other(
                    "CpuWorkload cannot run the gpu backend".into(),
                ))
            }
        };
        let state = Self::fresh_state(&cfg)?;
        Ok(Self {
            cfg,
            parallel,
            state,
        })
    }

    fn fresh_state(cfg: &ExperimentConfig) -> Result<State, WorkloadError> {
        Ok(if cfg.workload.is_particles() {
            let cur = particles::initial_particles(cfg);
            State::Particles {
                params: cfg.particle_params(),
                next: cur.clone(),
                cur,
            }
        } else {
            let cur = heat::initial_grid(cfg);
            State::Heat {
                params: cfg.heat_params()?,
                next: cur.clone(),
                cur,
            }
        })
    }
}

fn step_particles(
    kind: WorkloadKind,
    parallel: bool,
    params: &ParticleParams,
    cur: &[Particle],
    next: &mut [Particle],
) {
    let local = kind == WorkloadKind::ParticlesLocal;
    let f = |(i, out): (usize, &mut Particle)| {
        *out = if local {
            particles::local_update(&cur[i], params)
        } else {
            particles::nbody_update(i, cur, params)
        };
    };
    if parallel {
        next.par_iter_mut().enumerate().for_each(f);
    } else {
        next.iter_mut().enumerate().for_each(f);
    }
}

fn step_heat(parallel: bool, params: &HeatParams, cur: &[f32], next: &mut [f32]) {
    let nx = params.nx as usize;
    let row = |(j, out): (usize, &mut [f32])| {
        for (i, o) in out.iter_mut().enumerate() {
            *o = heat::cell_update(cur, i, j, params);
        }
    };
    if parallel {
        next.par_chunks_mut(nx).enumerate().for_each(row);
    } else {
        next.chunks_mut(nx).enumerate().for_each(row);
    }
}

impl Workload for CpuWorkload {
    fn describe(&self) -> String {
        match (self.parallel, rayon::current_num_threads()) {
            (false, _) => "cpu-scalar (1 thread)".into(),
            (true, n) => format!("cpu-rayon ({n} threads)"),
        }
    }

    fn config(&self) -> &ExperimentConfig {
        &self.cfg
    }

    fn initialize(&mut self) -> Result<(), WorkloadError> {
        self.state = Self::fresh_state(&self.cfg)?;
        Ok(())
    }

    fn step(&mut self, steps: u32) -> Result<(), WorkloadError> {
        let kind = self.cfg.workload;
        let parallel = self.parallel;
        for _ in 0..steps {
            match &mut self.state {
                State::Particles { params, cur, next } => {
                    step_particles(kind, parallel, params, cur, next);
                    std::mem::swap(cur, next);
                }
                State::Heat { params, cur, next } => {
                    step_heat(parallel, params, cur, next);
                    std::mem::swap(cur, next);
                }
            }
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<(), WorkloadError> {
        Ok(())
    }

    fn snapshot(&mut self) -> Result<Snapshot, WorkloadError> {
        Ok(match &self.state {
            State::Particles { cur, .. } => Snapshot::Particles(cur.clone()),
            State::Heat { cur, .. } => Snapshot::Heat(cur.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_core::{BoundaryMode, Tolerance};

    fn cfg(workload: WorkloadKind, backend: BackendKind, size: u32) -> ExperimentConfig {
        ExperimentConfig {
            workload,
            backend,
            count: size,
            grid: size,
            ..Default::default()
        }
    }

    fn run(c: ExperimentConfig, steps: u32) -> Snapshot {
        let mut w = CpuWorkload::new(c).unwrap();
        w.step(steps).unwrap();
        w.snapshot().unwrap()
    }

    fn assert_scalar_eq_rayon(workload: WorkloadKind, size: u32, steps: u32) {
        let a = run(cfg(workload, BackendKind::Scalar, size), steps);
        let b = run(cfg(workload, BackendKind::Rayon, size), steps);
        assert!(
            a.compare(&b, Tolerance::EXACT).unwrap().passed,
            "{workload:?} n={size}"
        );
        assert_eq!(a, b);
    }

    #[test]
    fn rayon_matches_scalar_local() {
        for n in [1, 2, 31, 64, 65, 1000, 4097] {
            assert_scalar_eq_rayon(WorkloadKind::ParticlesLocal, n, 20);
        }
    }

    #[test]
    fn rayon_matches_scalar_nbody() {
        for n in [1, 2, 31, 65, 300] {
            assert_scalar_eq_rayon(WorkloadKind::NbodyAllpairs, n, 10);
        }
    }

    #[test]
    fn rayon_matches_scalar_heat() {
        for n in [3, 4, 17, 64, 129] {
            assert_scalar_eq_rayon(WorkloadKind::Heat, n, 30);
        }
    }

    #[test]
    fn rayon_matches_scalar_heat_neumann() {
        let mut c = cfg(WorkloadKind::Heat, BackendKind::Scalar, 50);
        c.boundary = BoundaryMode::Neumann;
        let a = run(c.clone(), 40);
        c.backend = BackendKind::Rayon;
        assert_eq!(a, run(c, 40));
    }

    #[test]
    fn rejects_gpu_backend() {
        assert!(CpuWorkload::new(cfg(WorkloadKind::Heat, BackendKind::Gpu, 8)).is_err());
    }

    #[test]
    fn rejects_invalid_config() {
        let mut c = cfg(WorkloadKind::Heat, BackendKind::Scalar, 8);
        c.diffusion_number = 0.9;
        assert!(CpuWorkload::new(c).is_err());
    }

    #[test]
    fn initialize_resets_state() {
        let c = cfg(WorkloadKind::ParticlesLocal, BackendKind::Scalar, 100);
        let mut w = CpuWorkload::new(c).unwrap();
        let s0 = w.snapshot().unwrap();
        w.step(25).unwrap();
        assert_ne!(w.snapshot().unwrap(), s0);
        w.initialize().unwrap();
        assert_eq!(w.snapshot().unwrap(), s0);
    }

    #[test]
    fn runs_are_repeatable() {
        let c = cfg(WorkloadKind::NbodyAllpairs, BackendKind::Rayon, 128);
        assert_eq!(run(c.clone(), 15), run(c, 15));
    }

    #[test]
    fn zero_steps_is_identity() {
        let c = cfg(WorkloadKind::Heat, BackendKind::Scalar, 16);
        let mut w = CpuWorkload::new(c).unwrap();
        let s0 = w.snapshot().unwrap();
        w.step(0).unwrap();
        assert_eq!(w.snapshot().unwrap(), s0);
    }

    #[test]
    fn step_batches_compose() {
        let c = cfg(WorkloadKind::ParticlesLocal, BackendKind::Scalar, 77);
        let mut w = CpuWorkload::new(c.clone()).unwrap();
        w.step(3).unwrap();
        w.step(4).unwrap();
        assert_eq!(w.snapshot().unwrap(), run(c, 7));
    }

    #[test]
    fn heat_state_stays_finite_long_run() {
        let c = cfg(WorkloadKind::Heat, BackendKind::Rayon, 64);
        let s = run(c, 500);
        assert!(s.to_f32().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn describe_names_backend() {
        let w = CpuWorkload::new(cfg(WorkloadKind::Heat, BackendKind::Scalar, 8)).unwrap();
        assert!(w.describe().contains("scalar"));
        let w = CpuWorkload::new(cfg(WorkloadKind::Heat, BackendKind::Rayon, 8)).unwrap();
        assert!(w.describe().contains("rayon"));
    }
}
