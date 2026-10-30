use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use aether_core::{
    validate::Tolerance, BackendKind, ExperimentConfig, Snapshot, Summary, Workload, WorkloadError,
    WorkloadKind,
};
use aether_cpu::CpuWorkload;
use aether_gpu::{GpuContext, GpuWorkload};
use rand::seq::SliceRandom;
use rand::SeedableRng;
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
    pub mode: BenchMode,
    /// Randomise backend/workgroup execution order within each size, so
    /// thermal state does not systematically favour one backend.
    pub shuffle: bool,
    pub order_seed: u64,
    /// Idle time between runs.
    pub cooldown_ms: u64,
    /// Command line to record in the report.
    pub command: String,
}

impl Default for SweepSpec {
    fn default() -> Self {
        Self {
            base: ExperimentConfig::default(),
            backends: BackendKind::ALL.to_vec(),
            sizes: vec![],
            workgroups: vec![],
            mode: BenchMode::default(),
            shuffle: true,
            order_seed: 1,
            cooldown_ms: 0,
            command: String::new(),
        }
    }
}

/// Fraction of physical RAM a single run may be estimated to use.
const RAM_BUDGET_FRACTION: f64 = 0.6;

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
    mode: BenchMode,
) -> BenchRecord {
    match bench_one_inner(cfg, gpu, reference, mode) {
        Ok(r) => r,
        Err(e) => BenchRecord::failed(cfg.clone(), cfg.backend.as_str().to_string(), e.to_string()),
    }
}

fn bench_one_inner(
    cfg: &ExperimentConfig,
    gpu: Option<&Arc<GpuContext>>,
    reference: &Snapshot,
    mode: BenchMode,
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

    // Continuous mode restarts from the initial state once, then evolves.
    if mode == BenchMode::Continuous {
        w.initialize()?;
        w.sync()?;
    }
    for _ in 0..cfg.warmup {
        w.step(cfg.steps)?;
        w.sync()?;
    }

    let mut samples = Samples::default();
    let mut last_snap = None;
    for _ in 0..cfg.repeats {
        if mode == BenchMode::ResetEachSample {
            // Untimed: every sample does identical work from identical state.
            w.initialize()?;
            w.sync()?;
        }
        let t = Instant::now();
        w.step(cfg.steps)?;
        w.sync()?;
        let compute = ms(t);
        let t = Instant::now();
        let snap = w.snapshot()?;
        let readback = ms(t);
        std::hint::black_box(&snap);
        last_snap = Some(snap);
        samples.compute_ms.push(compute);
        samples.readback_ms.push(readback);
        samples.end_to_end_ms.push(compute + readback);
    }

    // Check the state that was actually measured, not just the cold run.
    let (final_validation, final_nonfinite) = match &last_snap {
        Some(snap) => {
            let nonfinite = snap.to_f32().iter().filter(|v| !v.is_finite()).count();
            let fv = match mode {
                BenchMode::ResetEachSample => Some(snap.compare(reference, tol)?),
                BenchMode::Continuous => None,
            };
            (fv, Some(nonfinite))
        }
        None => (None, None),
    };

    let compute = Summary::from_samples(&samples.compute_ms);
    let median_step_ms = compute.as_ref().map(|s| s.median / cfg.steps.max(1) as f64);
    let median_mupdates_per_s = compute
        .as_ref()
        .map(|s| cfg.updates_per_step() as f64 * cfg.steps as f64 / (s.median / 1e3) / 1e6);
    Ok(BenchRecord {
        status: RecordStatus::Success,
        config: cfg.clone(),
        backend_description: w.describe(),
        complexity: cfg.workload.complexity().to_string(),
        cold_start: Some(ColdStart {
            setup_ms,
            first_run_ms,
        }),
        readback: Summary::from_samples(&samples.readback_ms),
        end_to_end: Summary::from_samples(&samples.end_to_end_ms),
        compute,
        samples,
        median_step_ms,
        median_mupdates_per_s,
        validation: Some(validation),
        final_validation,
        final_nonfinite,
        run_index: 0,
        error: None,
    })
}

fn median_of(r: &BenchRecord, f: impl Fn(&BenchRecord) -> Option<&Summary>) -> Option<f64> {
    f(r).map(|s| s.median)
}

fn speedups(records: &[BenchRecord]) -> Vec<Speedup> {
    let size_of = |r: &BenchRecord| {
        if r.config.workload.is_particles() {
            r.config.count
        } else {
            r.config.grid
        }
    };
    let find = |kind: WorkloadKind, size: u32, backend: BackendKind| {
        records.iter().find(|r| {
            r.config.workload == kind
                && size_of(r) == size
                && r.config.backend == backend
                && r.validated_ok()
        })
    };
    let ratio =
        |cpu: Option<&BenchRecord>, gpu: &BenchRecord, f: fn(&BenchRecord) -> Option<&Summary>| {
            let (c, g) = (median_of(cpu?, f)?, median_of(gpu, f)?);
            (g > 0.0).then_some(c / g)
        };
    records
        .iter()
        .filter(|r| r.config.backend == BackendKind::Gpu && r.validated_ok())
        .map(|g| {
            let (kind, size) = (g.config.workload, size_of(g));
            let (scalar, rayon) = (
                find(kind, size, BackendKind::Scalar),
                find(kind, size, BackendKind::Rayon),
            );
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
pub fn run_sweep(
    spec: &SweepSpec,
    progress: &mut dyn FnMut(&str),
) -> Result<BenchReport, BenchError> {
    spec.base
        .validate()
        .map_err(|e| BenchError::Config(e.to_string()))?;
    if spec.base.repeats < 1 {
        return Err(BenchError::Config("repeats must be >= 1".into()));
    }
    if spec.base.steps < 1 {
        return Err(BenchError::Config(
            "steps must be >= 1 for a benchmark".into(),
        ));
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

    let default_size = if spec.base.workload.is_particles() {
        spec.base.count
    } else {
        spec.base.grid
    };
    let sizes = if spec.sizes.is_empty() {
        vec![default_size]
    } else {
        spec.sizes.clone()
    };
    let workgroups = if spec.workgroups.is_empty() {
        vec![spec.base.workgroup_size]
    } else {
        spec.workgroups.clone()
    };

    let machine = MachineInfo::detect();
    let ram_budget = (machine.ram_bytes as f64 * RAM_BUDGET_FRACTION) as u64;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(spec.order_seed);
    let mut run_order = Vec::new();
    let mut records: Vec<(usize, BenchRecord)> = Vec::new();
    for &size in &sizes {
        let cfg = with_size(&spec.base, size);
        cfg.validate()
            .map_err(|e| BenchError::Config(e.to_string()))?;
        cfg.check_memory(ram_budget)
            .map_err(|e| BenchError::Config(e.to_string()))?;
        progress(&format!(
            "[{}] {}: computing scalar reference",
            cfg.workload.as_str(),
            cfg.size_label()
        ));
        let reference = reference_run(&cfg)?;

        // Canonical order is the report order; execution order may differ.
        let mut jobs: Vec<(BackendKind, u32)> = Vec::new();
        for &backend in &spec.backends {
            let wgs: &[u32] = if backend == BackendKind::Gpu {
                &workgroups
            } else {
                &workgroups[..1]
            };
            jobs.extend(wgs.iter().map(|&wg| (backend, wg)));
        }
        let canonical: Vec<(BackendKind, u32)> = jobs.clone();
        if spec.shuffle {
            jobs.shuffle(&mut rng);
        }
        for (backend, wg) in jobs {
            let mut c = cfg.clone();
            c.backend = backend;
            c.workgroup_size = wg;
            let label = if backend == BackendKind::Gpu {
                format!("{} wg={wg}", backend.as_str())
            } else {
                backend.as_str().to_string()
            };
            progress(&format!(
                "[{}] {}: {label}",
                c.workload.as_str(),
                c.size_label()
            ));
            if spec.cooldown_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(spec.cooldown_ms));
            }
            let mut rec = if backend == BackendKind::Gpu && gpu.is_none() {
                BenchRecord::unavailable(
                    c,
                    "gpu (unavailable)".into(),
                    gpu_error.clone().unwrap_or_else(|| "no GPU".into()),
                )
            } else {
                bench_one(&c, gpu.as_ref(), &reference, spec.mode)
            };
            rec.run_index = run_order.len();
            run_order.push(format!(
                "{} {} {label}",
                rec.config.workload.as_str(),
                rec.config.size_label()
            ));
            if let Some(m) = rec.median_step_ms {
                let drift = rec.compute.as_ref().map_or(1.0, |s| s.order_drift);
                progress(&format!(
                    "    median {m:.4} ms/step, {:.1} M updates/s, validation {}{}",
                    rec.median_mupdates_per_s.unwrap_or(0.0),
                    match &rec.validation {
                        Some(v) if v.passed => "PASS".to_string(),
                        Some(v) => format!("FAIL (max_abs {:.3e})", v.max_abs),
                        None => "n/a".into(),
                    },
                    if (drift - 1.0).abs() > 0.1 {
                        format!(", WARNING sample drift x{drift:.2}")
                    } else {
                        String::new()
                    }
                ));
            } else if let Some(e) = &rec.error {
                progress(&format!("    skipped: {e}"));
            }
            let pos = canonical
                .iter()
                .position(|&j| j == (backend, wg))
                .unwrap_or(0);
            records.push((pos, rec));
        }
    }
    // Restore canonical (size, backend, workgroup) order for stable reports.
    // The sort is stable and sizes were appended in order, so sorting by the
    // per-size canonical position needs the size index too.
    let size_index = |r: &BenchRecord| {
        let sz = if r.config.workload.is_particles() {
            r.config.count
        } else {
            r.config.grid
        };
        sizes.iter().position(|&x| x == sz).unwrap_or(0)
    };
    records.sort_by_key(|(pos, r)| (size_index(r), *pos));
    let records: Vec<BenchRecord> = records.into_iter().map(|(_, r)| r).collect();

    let (git_sha, git_dirty) = git_state();
    let speedups = speedups(&records);
    Ok(BenchReport {
        schema_version: SCHEMA_VERSION,
        git_sha,
        git_dirty,
        timestamp_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        machine,
        adapter: gpu.as_ref().map(|g| g.report.clone()),
        timing_scope: TimingScope::default(),
        command: spec.command.clone(),
        bench_mode: spec.mode,
        shuffled: spec.shuffle,
        order_seed: spec.order_seed,
        cooldown_ms: spec.cooldown_ms,
        run_order,
        records,
        speedups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick(workload: WorkloadKind) -> ExperimentConfig {
        ExperimentConfig {
            workload,
            count: 200,
            grid: 32,
            steps: 5,
            warmup: 1,
            repeats: 3,
            ..Default::default()
        }
    }

    fn spec(workload: WorkloadKind, backends: &[BackendKind]) -> SweepSpec {
        SweepSpec {
            base: quick(workload),
            backends: backends.to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn cpu_sweep_validates_and_records_samples() {
        let r = run_sweep(
            &spec(
                WorkloadKind::ParticlesLocal,
                &[BackendKind::Scalar, BackendKind::Rayon],
            ),
            &mut |_| {},
        )
        .unwrap();
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
        let r = run_sweep(
            &spec(WorkloadKind::Heat, &[BackendKind::Scalar]),
            &mut |_| {},
        )
        .unwrap();
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
        let mut s = spec(WorkloadKind::Heat, &[BackendKind::Scalar]);
        s.base.steps = 0;
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
        assert_eq!(
            r.records
                .iter()
                .filter(|x| x.config.backend == BackendKind::Gpu)
                .count(),
            3
        );
        assert_eq!(
            r.records
                .iter()
                .filter(|x| x.config.backend == BackendKind::Scalar)
                .count(),
            1
        );
    }

    #[test]
    fn failures_flag_bad_validation() {
        let mut r = run_sweep(
            &spec(WorkloadKind::Heat, &[BackendKind::Scalar]),
            &mut |_| {},
        )
        .unwrap();
        r.records[0].validation.as_mut().unwrap().passed = false;
        assert_eq!(r.failures(false).len(), 1);
    }

    #[test]
    fn modes_both_validate_and_record_provenance() {
        for mode in [BenchMode::ResetEachSample, BenchMode::Continuous] {
            let mut s = spec(WorkloadKind::ParticlesLocal, &[BackendKind::Rayon]);
            s.mode = mode;
            s.command = "aether bench test".into();
            let r = run_sweep(&s, &mut |_| {}).unwrap();
            assert_eq!(r.bench_mode, mode);
            assert_eq!(r.command, "aether bench test");
            assert!(r.records[0].validated_ok());
            assert_eq!(r.records[0].final_nonfinite, Some(0));
            assert_eq!(
                r.records[0].final_validation.is_some(),
                mode == BenchMode::ResetEachSample
            );
        }
    }

    #[test]
    fn reset_mode_final_state_matches_reference_exactly_on_cpu() {
        let r = run_sweep(
            &spec(
                WorkloadKind::Heat,
                &[BackendKind::Scalar, BackendKind::Rayon],
            ),
            &mut |_| {},
        )
        .unwrap();
        for rec in &r.records {
            let fv = rec.final_validation.as_ref().unwrap();
            assert!(fv.passed);
            assert_eq!(fv.max_abs, 0.0);
        }
    }

    #[test]
    fn shuffle_is_seeded_recorded_and_keeps_report_order() {
        let mut s = spec(WorkloadKind::Heat, &BackendKind::ALL);
        s.sizes = vec![16, 24];
        s.order_seed = 7;
        let a = run_sweep(&s, &mut |_| {}).unwrap();
        let b = run_sweep(&s, &mut |_| {}).unwrap();
        assert_eq!(a.run_order, b.run_order);
        assert_eq!(a.run_order.len(), 6);
        // Report order stays (size, backend) regardless of execution order.
        let order: Vec<_> = a
            .records
            .iter()
            .map(|r| (r.config.grid, r.config.backend))
            .collect();
        assert_eq!(order[0], (16, BackendKind::Scalar));
        assert_eq!(order[3], (24, BackendKind::Scalar));
        let mut idx: Vec<_> = a.records.iter().map(|r| r.run_index).collect();
        idx.sort();
        assert_eq!(idx, vec![0, 1, 2, 3, 4, 5]);
        s.shuffle = false;
        let c = run_sweep(&s, &mut |_| {}).unwrap();
        assert!(!c.shuffled);
        assert!(c.run_order[0].ends_with("scalar"));
    }

    #[test]
    fn status_distinguishes_unavailable_from_failed() {
        let r = run_sweep(
            &spec(WorkloadKind::Heat, &[BackendKind::Scalar]),
            &mut |_| {},
        )
        .unwrap();
        let mut rep = r.clone();
        let cfg = rep.records[0].config.clone();
        let mut gpu_cfg = cfg.clone();
        gpu_cfg.backend = BackendKind::Gpu;
        rep.records = vec![
            BenchRecord::unavailable(gpu_cfg.clone(), "gpu".into(), "no adapter".into()),
            BenchRecord::failed(gpu_cfg, "gpu".into(), "shader compile error".into()),
        ];
        // The missing adapter is optional; a real GPU failure never is.
        assert_eq!(rep.failures(false).len(), 1);
        assert!(rep.failures(false)[0].contains("shader compile"));
        assert_eq!(rep.failures(true).len(), 2);
    }

    #[test]
    fn invalid_records_do_not_produce_speedups() {
        let mut recs = run_sweep(
            &spec(WorkloadKind::Heat, &[BackendKind::Scalar]),
            &mut |_| {},
        )
        .unwrap()
        .records;
        let mut gpu = recs[0].clone();
        gpu.config.backend = BackendKind::Gpu;
        recs.push(gpu.clone());
        assert_eq!(speedups(&recs).len(), 1);
        gpu.validation.as_mut().unwrap().passed = false;
        recs[1] = gpu;
        assert!(speedups(&recs).is_empty());
        // An invalid CPU baseline is not used either.
        let mut recs2 = recs.clone();
        recs2[1].validation.as_mut().unwrap().passed = true;
        recs2[0].validation.as_mut().unwrap().passed = false;
        assert_eq!(speedups(&recs2)[0].vs_scalar_compute, None);
    }

    #[test]
    fn create_workload_requires_context_for_gpu() {
        let mut c = quick(WorkloadKind::Heat);
        c.backend = BackendKind::Gpu;
        assert!(create_workload(&c, None).is_err());
    }
}
