use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use aether_core::{
    validate::Tolerance, BackendKind, ExperimentConfig, Snapshot, Summary, Workload, WorkloadError,
    WorkloadKind,
};
use aether_cpu::CpuWorkload;
use aether_gpu::{GpuContext, GpuWorkload};
use thiserror::Error;

use crate::machine::{git_state, MachineInfo};
use crate::record::*;

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("{0}")]
    Config(String),
    #[error(transparent)]
    Workload(#[from] WorkloadError),
}

/// Build the workload for `cfg.backend`. The GPU backend needs a context.
pub fn create_workload(
    cfg: &ExperimentConfig,
    gpu: Option<&Arc<GpuContext>>,
) -> Result<Box<dyn Workload>, WorkloadError> {
    match cfg.backend {
        BackendKind::Scalar | BackendKind::Rayon => Ok(Box::new(CpuWorkload::new(cfg.clone())?)),
        BackendKind::Gpu => {
            let ctx = gpu.ok_or_else(|| WorkloadError::Gpu("no GPU context available".into()))?;
            Ok(Box::new(GpuWorkload::new(ctx.clone(), cfg.clone())?))
        }
    }
}

#[derive(Debug, Clone)]
pub struct SweepSpec {
    pub base: ExperimentConfig,
    pub backends: Vec<BackendKind>,
    /// Particle counts or grid edges; empty = the base config's size.
    pub sizes: Vec<u32>,
    /// GPU workgroup sizes to sweep; empty = the base config's.
    pub workgroups: Vec<u32>,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn with_size(base: &ExperimentConfig, size: u32) -> ExperimentConfig {
    let mut c = base.clone();
    if c.workload.is_particles() {
        c.count = size;
    } else {
        c.grid = size;
    }
    c
}

fn reference_run(cfg: &ExperimentConfig) -> Result<Snapshot, WorkloadError> {
    let mut c = cfg.clone();
    c.backend = BackendKind::Scalar;
    let mut w = CpuWorkload::new(c)?;
    w.step(cfg.steps)?;
    w.snapshot()
}

/// Run one configuration: cold start, warm-up, measured repeats, validation.
fn bench_one(
    cfg: &ExperimentConfig,
    gpu: Option<&Arc<GpuContext>>,
    reference: &Snapshot,
) -> BenchRecord {
    match bench_one_inner(cfg, gpu, reference) {
        Ok(r) => r,
        Err(e) => BenchRecord::failed(cfg.clone(), cfg.backend.as_str().to_string(), e.to_string()),
    }
}

fn bench_one_inner(
    cfg: &ExperimentConfig,
    gpu: Option<&Arc<GpuContext>>,
    reference: &Snapshot,
) -> Result<BenchRecord, WorkloadError> {
    // Cold start: construction (upload + pipeline compile) and first run.
    let t = Instant::now();
    let mut w = create_workload(cfg, gpu)?;
    let setup_ms = ms(t);
    let t = Instant::now();
    w.step(cfg.steps)?;
    w.sync()?;
    let first_run_ms = ms(t);

    // Correctness first: a fast but incorrect kernel is a failed result.
    let tol = match cfg.backend {
        BackendKind::Gpu => Tolerance::gpu_vs_cpu(cfg.workload, cfg.steps),
        _ => Tolerance::EXACT,
    };
    let validation = w.validate(reference, tol)?;

    for _ in 0..cfg.warmup {
        w.step(cfg.steps)?;
        w.sync()?;
    }

    let mut samples = Samples::default();
    for _ in 0..cfg.repeats {
        let t = Instant::now();
        w.step(cfg.steps)?;
        w.sync()?;
        let compute = ms(t);
        let t = Instant::now();
        let snap = w.snapshot()?;
        let readback = ms(t);
        std::hint::black_box(&snap);
        samples.compute_ms.push(compute);
        samples.readback_ms.push(readback);
        samples.end_to_end_ms.push(compute + readback);
    }

    let compute = Summary::from_samples(&samples.compute_ms);
    let median_step_ms = compute.as_ref().map(|s| s.median / cfg.steps.max(1) as f64);
    let median_mupdates_per_s = compute.as_ref().map(|s| {
        cfg.updates_per_step() as f64 * cfg.steps as f64 / (s.median / 1e3) / 1e6
    });
    Ok(BenchRecord {
        config: cfg.clone(),
        backend_description: w.describe(),
        complexity: cfg.workload.complexity().to_string(),
        cold_start: Some(ColdStart { setup_ms, first_run_ms }),
        readback: Summary::from_samples(&samples.readback_ms),
        end_to_end: Summary::from_samples(&samples.end_to_end_ms),
        compute,
        samples,
        median_step_ms,
        median_mupdates_per_s,
        validation: Some(validation),
        error: None,
    })
}

fn median_of(r: &BenchRecord, f: impl Fn(&BenchRecord) -> Option<&Summary>) -> Option<f64> {
    f(r).map(|s| s.median)
}

fn speedups(records: &[BenchRecord]) -> Vec<Speedup> {
    let size_of = |r: &BenchRecord| {
        if r.config.workload.is_particles() { r.config.count } else { r.config.grid }
    };
    let find = |kind: WorkloadKind, size: u32, backend: BackendKind| {
        records.iter().find(|r| {
            r.config.workload == kind && size_of(r) == size && r.config.backend == backend && r.error.is_none()
        })
    };
    let ratio = |cpu: Option<&BenchRecord>, gpu: &BenchRecord, f: fn(&BenchRecord) -> Option<&Summary>| {
        let (c, g) = (median_of(cpu?, f)?, median_of(gpu, f)?);
        (g > 0.0).then_some(c / g)
    };
    records
        .iter()
        .filter(|r| r.config.backend == BackendKind::Gpu && r.error.is_none())
        .map(|g| {
            let (kind, size) = (g.config.workload, size_of(g));
            let (scalar, rayon) = (find(kind, size, BackendKind::Scalar), find(kind, size, BackendKind::Rayon));
            Speedup {
                workload: kind.as_str().to_string(),
                size,
                workgroup_size: g.config.workgroup_size,
                vs_scalar_compute: ratio(scalar, g, |r| r.compute.as_ref()),
                vs_rayon_compute: ratio(rayon, g, |r| r.compute.as_ref()),
                vs_scalar_end_to_end: ratio(scalar, g, |r| r.end_to_end.as_ref()),
                vs_rayon_end_to_end: ratio(rayon, g, |r| r.end_to_end.as_ref()),
            }
        })
        .collect()
}

/// Run every (size, backend[, workgroup]) combination and assemble a report.
/// A missing GPU does not abort the sweep: GPU records carry the error.
pub fn run_sweep(spec: &SweepSpec, progress: &mut dyn FnMut(&str)) -> Result<BenchReport, BenchError> {
    spec.base.validate().map_err(|e| BenchError::Config(e.to_string()))?;
    if spec.base.repeats < 1 {
        return Err(BenchError::Config("repeats must be >= 1".into()));
    }
    let mut gpu_error = None;
    let gpu = if spec.backends.contains(&BackendKind::Gpu) {
        match GpuContext::new() {
            Ok(c) => Some(c),
            Err(e) => {
                gpu_error = Some(e.to_string());
                None
            }
        }
    } else {
        None
    };

    let default_size = if spec.base.workload.is_particles() { spec.base.count } else { spec.base.grid };
    let sizes = if spec.sizes.is_empty() { vec![default_size] } else { spec.sizes.clone() };
    let workgroups = if spec.workgroups.is_empty() {
        vec![spec.base.workgroup_size]
    } else {
        spec.workgroups.clone()
    };

    let mut records = Vec::new();
    for &size in &sizes {
        let cfg = with_size(&spec.base, size);
        cfg.validate().map_err(|e| BenchError::Config(e.to_string()))?;
        progress(&format!("[{}] {}: computing scalar reference", cfg.workload.as_str(), cfg.size_label()));
        let reference = reference_run(&cfg)?;
        for &backend in &spec.backends {
            let wgs: &[u32] = if backend == BackendKind::Gpu { &workgroups } else { &workgroups[..1] };
            for &wg in wgs {
                let mut c = cfg.clone();
                c.backend = backend;
                c.workgroup_size = wg;
                let label = if backend == BackendKind::Gpu {
                    format!("{} wg={wg}", backend.as_str())
                } else {
                    backend.as_str().to_string()
                };
                progress(&format!("[{}] {}: {label}", c.workload.as_str(), c.size_label()));
                let rec = if backend == BackendKind::Gpu && gpu.is_none() {
                    BenchRecord::failed(
                        c,
                        "gpu (unavailable)".into(),
                        gpu_error.clone().unwrap_or_else(|| "no GPU".into()),
                    )
                } else {
                    bench_one(&c, gpu.as_ref(), &reference)
                };
                if let Some(m) = rec.median_step_ms {
                    progress(&format!(
                        "    median {m:.4} ms/step, {:.1} M updates/s, validation {}",
                        rec.median_mupdates_per_s.unwrap_or(0.0),
                        match &rec.validation {
                            Some(v) if v.passed => "PASS".to_string(),
                            Some(v) => format!("FAIL (max_abs {:.3e})", v.max_abs),
                            None => "n/a".into(),
                        }
                    ));
                } else if let Some(e) = &rec.error {
                    progress(&format!("    skipped: {e}"));
                }
                records.push(rec);
            }
        }
    }

    let (git_sha, git_dirty) = git_state();
    let speedups = speedups(&records);
    Ok(BenchReport {
        schema_version: SCHEMA_VERSION,
        git_sha,
        git_dirty,
        timestamp_unix: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        machine: MachineInfo::detect(),
        adapter: gpu.as_ref().map(|g| g.report.clone()),
        timing_scope: TimingScope::default(),
        records,
        speedups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick(workload: WorkloadKind) -> ExperimentConfig {
        ExperimentConfig { workload, count: 200, grid: 32, steps: 5, warmup: 1, repeats: 3, ..Default::default() }
    }

    fn spec(workload: WorkloadKind, backends: &[BackendKind]) -> SweepSpec {
        SweepSpec { base: quick(workload), backends: backends.to_vec(), sizes: vec![], workgroups: vec![] }
    }

    #[test]
    fn cpu_sweep_validates_and_records_samples() {
        let r = run_sweep(&spec(WorkloadKind::ParticlesLocal, &[BackendKind::Scalar, BackendKind::Rayon]), &mut |_| {}).unwrap();
        assert_eq!(r.records.len(), 2);
        for rec in &r.records {
            assert!(rec.validated_ok(), "{rec:?}");
            assert_eq!(rec.samples.compute_ms.len(), 3);
            assert_eq!(rec.samples.readback_ms.len(), 3);
            assert!(rec.median_mupdates_per_s.unwrap() > 0.0);
            assert!(rec.cold_start.is_some());
        }
        assert!(r.failures(false).is_empty());
        assert_eq!(r.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn heat_sweep_over_sizes() {
        let mut s = spec(WorkloadKind::Heat, &[BackendKind::Scalar]);
        s.sizes = vec![16, 32, 48];
        let r = run_sweep(&s, &mut |_| {}).unwrap();
        assert_eq!(r.records.len(), 3);
        assert_eq!(r.records[2].config.grid, 48);
    }

    #[test]
    fn end_to_end_is_compute_plus_readback() {
        let r = run_sweep(&spec(WorkloadKind::Heat, &[BackendKind::Scalar]), &mut |_| {}).unwrap();
        let s = &r.records[0].samples;
        for i in 0..s.compute_ms.len() {
            assert!((s.end_to_end_ms[i] - s.compute_ms[i] - s.readback_ms[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn rejects_bad_config() {
        let mut s = spec(WorkloadKind::Heat, &[BackendKind::Scalar]);
        s.base.diffusion_number = 0.9;
        assert!(run_sweep(&s, &mut |_| {}).is_err());
        let mut s = spec(WorkloadKind::Heat, &[BackendKind::Scalar]);
        s.base.repeats = 0;
        assert!(run_sweep(&s, &mut |_| {}).is_err());
    }

    #[test]
    fn gpu_sweep_validates_and_reports_speedups_when_gpu_present() {
        let s = spec(WorkloadKind::ParticlesLocal, &BackendKind::ALL);
        let r = run_sweep(&s, &mut |_| {}).unwrap();
        assert_eq!(r.records.len(), 3);
        if r.adapter.is_some() {
            assert!(r.failures(true).is_empty(), "{:?}", r.failures(true));
            assert_eq!(r.speedups.len(), 1);
            assert!(r.speedups[0].vs_scalar_compute.unwrap() > 0.0);
        } else {
            assert!(r.failures(false).is_empty());
            assert!(!r.failures(true).is_empty());
        }
    }

    #[test]
    fn workgroup_sweep_makes_one_gpu_record_each() {
        let mut s = spec(WorkloadKind::Heat, &[BackendKind::Scalar, BackendKind::Gpu]);
        s.workgroups = vec![32, 64, 128];
        let r = run_sweep(&s, &mut |_| {}).unwrap();
        assert_eq!(r.records.iter().filter(|x| x.config.backend == BackendKind::Gpu).count(), 3);
        assert_eq!(r.records.iter().filter(|x| x.config.backend == BackendKind::Scalar).count(), 1);
    }

    #[test]
    fn failures_flag_bad_validation() {
        let mut r = run_sweep(&spec(WorkloadKind::Heat, &[BackendKind::Scalar]), &mut |_| {}).unwrap();
        r.records[0].validation.as_mut().unwrap().passed = false;
        assert_eq!(r.failures(false).len(), 1);
    }

    #[test]
    fn create_workload_requires_context_for_gpu() {
        let mut c = quick(WorkloadKind::Heat);
        c.backend = BackendKind::Gpu;
        assert!(create_workload(&c, None).is_err());
    }
}
