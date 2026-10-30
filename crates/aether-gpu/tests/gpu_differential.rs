//! GPU-vs-CPU differential tests. They skip (with a message) when the machine
//! has no usable GPU adapter, so CPU-only CI stays green; on an Apple Silicon
//! Mac they run on the Metal backend.

use std::sync::{Arc, OnceLock};

use aether_core::{
    validate::Tolerance, BackendKind, BoundaryMode, ExperimentConfig, PassMode, Snapshot, Workload,
    WorkloadKind,
};
use aether_cpu::CpuWorkload;
use aether_gpu::{GpuContext, GpuWorkload};

fn ctx() -> Option<Arc<GpuContext>> {
    static CTX: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();
    CTX.get_or_init(|| match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP: no GPU available: {e}");
            None
        }
    })
    .clone()
}

macro_rules! gpu_or_skip {
    () => {
        match ctx() {
            Some(c) => c,
            None => return,
        }
    };
}

fn cfg(workload: WorkloadKind, size: u32, steps: u32) -> ExperimentConfig {
    ExperimentConfig {
        workload,
        count: size,
        grid: size,
        steps,
        ..Default::default()
    }
}

fn gpu_run(ctx: &Arc<GpuContext>, c: &ExperimentConfig, steps: u32) -> Snapshot {
    let mut c = c.clone();
    c.backend = BackendKind::Gpu;
    let mut w = GpuWorkload::new(ctx.clone(), c).unwrap();
    w.step(steps).unwrap();
    w.sync().unwrap();
    w.snapshot().unwrap()
}

fn cpu_run(c: &ExperimentConfig, steps: u32) -> Snapshot {
    let mut c = c.clone();
    c.backend = BackendKind::Scalar;
    let mut w = CpuWorkload::new(c).unwrap();
    w.step(steps).unwrap();
    w.snapshot().unwrap()
}

fn check(ctx: &Arc<GpuContext>, c: ExperimentConfig, steps: u32) {
    let gpu = gpu_run(ctx, &c, steps);
    let cpu = cpu_run(&c, steps);
    let tol = Tolerance::gpu_vs_cpu(c.workload, steps);
    let r = gpu.compare(&cpu, tol).unwrap();
    assert!(
        r.passed,
        "{:?} size={} wg={} steps={steps}: {r:?}",
        c.workload,
        c.count.max(c.grid),
        c.workgroup_size
    );
}

#[test]
fn probe_reports_adapter() {
    let ctx = gpu_or_skip!();
    let r = &ctx.report;
    assert!(!r.name.is_empty());
    assert!(r.limits.max_buffer_size > 0);
    assert!(r.limits.max_compute_invocations_per_workgroup >= 256);
    eprintln!(
        "adapter: {} / {} / {:?}",
        r.name, r.backend, r.optional_features
    );
    #[cfg(target_os = "macos")]
    assert_eq!(r.backend, "Metal");
}

#[test]
fn local_particles_match_cpu_across_sizes() {
    let ctx = gpu_or_skip!();
    for n in [1, 2, 31, 63, 64, 65, 127, 1000, 1024, 10_000, 100_000] {
        check(&ctx, cfg(WorkloadKind::ParticlesLocal, n, 50), 50);
    }
}

#[test]
fn local_particles_match_cpu_across_workgroup_sizes() {
    let ctx = gpu_or_skip!();
    for wg in [1, 32, 64, 128, 256] {
        let mut c = cfg(WorkloadKind::ParticlesLocal, 1000, 30);
        c.workgroup_size = wg;
        check(&ctx, c, 30);
    }
}

#[test]
fn local_particles_long_run_stays_close() {
    let ctx = gpu_or_skip!();
    check(&ctx, cfg(WorkloadKind::ParticlesLocal, 5000, 1000), 1000);
}

#[test]
fn nbody_matches_cpu_across_sizes() {
    let ctx = gpu_or_skip!();
    for n in [1, 2, 31, 64, 65, 256, 1024] {
        check(&ctx, cfg(WorkloadKind::NbodyAllpairs, n, 20), 20);
    }
}

#[test]
fn nbody_matches_cpu_across_workgroup_sizes() {
    let ctx = gpu_or_skip!();
    for wg in [32, 64, 128, 256] {
        let mut c = cfg(WorkloadKind::NbodyAllpairs, 300, 10);
        c.workgroup_size = wg;
        check(&ctx, c, 10);
    }
}

#[test]
fn heat_matches_cpu_dirichlet() {
    let ctx = gpu_or_skip!();
    for n in [3, 4, 17, 64, 65, 255, 256, 512] {
        check(&ctx, cfg(WorkloadKind::Heat, n, 100), 100);
    }
}

#[test]
fn heat_matches_cpu_neumann() {
    let ctx = gpu_or_skip!();
    for n in [3, 31, 128, 300] {
        let mut c = cfg(WorkloadKind::Heat, n, 100);
        c.boundary = BoundaryMode::Neumann;
        check(&ctx, c, 100);
    }
}

#[test]
fn heat_matches_cpu_across_workgroup_sizes() {
    let ctx = gpu_or_skip!();
    for wg in [32, 64, 128, 256] {
        let mut c = cfg(WorkloadKind::Heat, 200, 50);
        c.workgroup_size = wg;
        check(&ctx, c, 50);
    }
}

#[test]
fn heat_large_grid_1024() {
    let ctx = gpu_or_skip!();
    check(&ctx, cfg(WorkloadKind::Heat, 1024, 20), 20);
}

#[test]
fn heat_conserves_energy_on_gpu_neumann() {
    let ctx = gpu_or_skip!();
    let mut c = cfg(WorkloadKind::Heat, 128, 400);
    c.boundary = BoundaryMode::Neumann;
    let before = match cpu_run(&c, 0) {
        Snapshot::Heat(g) => aether_core::heat::total_heat(&g),
        _ => unreachable!(),
    };
    let after = match gpu_run(&ctx, &c, 400) {
        Snapshot::Heat(g) => aether_core::heat::total_heat(&g),
        _ => unreachable!(),
    };
    assert!(
        (after - before).abs() / before < 1e-3,
        "{before} vs {after}"
    );
}

#[test]
fn zero_steps_returns_initial_state() {
    let ctx = gpu_or_skip!();
    let c = cfg(WorkloadKind::ParticlesLocal, 100, 0);
    assert_eq!(gpu_run(&ctx, &c, 0), cpu_run(&c, 0));
}

#[test]
fn step_batches_compose_on_gpu() {
    let ctx = gpu_or_skip!();
    let c = cfg(WorkloadKind::ParticlesLocal, 500, 7);
    let mut w = GpuWorkload::new(
        ctx.clone(),
        ExperimentConfig {
            backend: BackendKind::Gpu,
            ..c.clone()
        },
    )
    .unwrap();
    w.step(3).unwrap();
    w.step(4).unwrap();
    assert_eq!(w.snapshot().unwrap(), gpu_run(&ctx, &c, 7));
}

#[test]
fn pass_modes_are_bit_identical() {
    let ctx = gpu_or_skip!();
    for kind in [
        WorkloadKind::ParticlesLocal,
        WorkloadKind::Heat,
        WorkloadKind::NbodyAllpairs,
    ] {
        let mut a = cfg(kind, 300, 25);
        a.pass_mode = PassMode::PerStep;
        let mut b = a.clone();
        b.pass_mode = PassMode::Batched;
        assert_eq!(gpu_run(&ctx, &a, 25), gpu_run(&ctx, &b, 25), "{kind:?}");
        check(&ctx, a, 25);
    }
}

#[test]
fn gpu_is_repeatable() {
    let ctx = gpu_or_skip!();
    let c = cfg(WorkloadKind::NbodyAllpairs, 200, 10);
    assert_eq!(gpu_run(&ctx, &c, 10), gpu_run(&ctx, &c, 10));
}

#[test]
fn initialize_resets_gpu_state() {
    let ctx = gpu_or_skip!();
    let c = ExperimentConfig {
        backend: BackendKind::Gpu,
        ..cfg(WorkloadKind::Heat, 64, 10)
    };
    let mut w = GpuWorkload::new(ctx, c).unwrap();
    let s0 = w.snapshot().unwrap();
    w.step(10).unwrap();
    assert_ne!(w.snapshot().unwrap(), s0);
    w.initialize().unwrap();
    assert_eq!(w.snapshot().unwrap(), s0);
}

#[test]
fn validate_via_trait() {
    let ctx = gpu_or_skip!();
    let c = ExperimentConfig {
        backend: BackendKind::Gpu,
        ..cfg(WorkloadKind::Heat, 64, 20)
    };
    let reference = cpu_run(&c, 20);
    let mut w = GpuWorkload::new(ctx, c.clone()).unwrap();
    w.step(20).unwrap();
    let r = w
        .validate(&reference, Tolerance::gpu_vs_cpu(c.workload, 20))
        .unwrap();
    assert!(r.passed, "{r:?}");
}

#[test]
fn oversized_workgroup_is_a_clean_error() {
    let ctx = gpu_or_skip!();
    let c = ExperimentConfig {
        backend: BackendKind::Gpu,
        workgroup_size: 1024,
        ..cfg(WorkloadKind::ParticlesLocal, 10, 1)
    };
    // 1024 may be legal on some adapters; only assert that we never panic.
    let _ = GpuWorkload::new(ctx.clone(), c);
    let mut c = cfg(WorkloadKind::ParticlesLocal, 10, 1);
    c.workgroup_size = 512;
    c.backend = BackendKind::Gpu;
    let limit = ctx.report.limits.max_compute_invocations_per_workgroup;
    assert_eq!(GpuWorkload::new(ctx, c).is_err(), limit < 512);
}

#[test]
fn oversized_state_is_a_clean_error() {
    let ctx = gpu_or_skip!();
    let max = ctx
        .report
        .limits
        .max_storage_buffer_binding_size
        .min(ctx.report.limits.max_buffer_size);
    let too_many = (max / 32 + 1).min(50_000_000);
    if too_many * 32 <= max {
        return; // device limit above our config cap; nothing to test
    }
    let c = ExperimentConfig {
        backend: BackendKind::Gpu,
        count: too_many as u32,
        ..cfg(WorkloadKind::ParticlesLocal, 1, 1)
    };
    assert!(GpuWorkload::new(ctx, c).is_err());
}

#[test]
fn gpu_particle_energy_and_bounds() {
    let ctx = gpu_or_skip!();
    let c = cfg(WorkloadKind::ParticlesLocal, 20_000, 500);
    match gpu_run(&ctx, &c, 500) {
        Snapshot::Particles(ps) => {
            for p in ps {
                assert!(p.pos.iter().all(|x| x.is_finite() && x.abs() <= 1.0 + 1e-5));
                assert!(p.vel.iter().all(|x| x.is_finite()));
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn offscreen_render_produces_pixels() {
    let ctx = gpu_or_skip!();
    for kind in [WorkloadKind::ParticlesLocal, WorkloadKind::Heat] {
        let mut c = cfg(
            kind,
            if kind == WorkloadKind::Heat {
                128
            } else {
                5000
            },
            10,
        );
        c.backend = BackendKind::Gpu;
        let mut w = GpuWorkload::new(ctx.clone(), c.clone()).unwrap();
        w.step(10).unwrap();
        let view = aether_gpu::View::for_workload(kind, c.grid, c.count, 1.0);
        let px = aether_gpu::render_to_rgba(&ctx, &w, 128, 128, &view).unwrap();
        assert_eq!(px.len(), 128 * 128 * 4);
        let lit = px
            .chunks(4)
            .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 120)
            .count();
        assert!(
            lit > 50,
            "{kind:?}: expected visible content, got {lit} bright pixels"
        );
    }
}

fn timed(ctx: &Arc<GpuContext>, c: &ExperimentConfig, steps: u32) -> GpuWorkload {
    let mut c = c.clone();
    c.backend = BackendKind::Gpu;
    let mut w = GpuWorkload::new(ctx.clone(), c).unwrap();
    w.step(steps).unwrap();
    w.sync().unwrap();
    w
}

#[test]
fn device_timestamps_are_positive_finite_in_both_pass_modes() {
    let ctx = gpu_or_skip!();
    let info = GpuWorkload::new(ctx.clone(), cfg(WorkloadKind::Heat, 64, 1))
        .unwrap()
        .device_timing_info();
    assert_eq!(info.available, ctx.timestamps);
    if !ctx.timestamps {
        assert!(info.period_ns.is_none() && info.reason.contains("TIMESTAMP_QUERY"));
        return;
    }
    assert!(info.period_ns.unwrap() > 0.0);
    for mode in [PassMode::Batched, PassMode::PerStep] {
        for steps in [1, 7] {
            let mut c = cfg(WorkloadKind::Heat, 128, steps);
            c.pass_mode = mode;
            let mut w = timed(&ctx, &c, steps);
            let ms = w.take_device_time_ms().unwrap().expect("timestamp");
            assert!(ms.is_finite() && ms > 0.0, "{mode:?} steps={steps}: {ms}");
            // One reading per batch: a second take has nothing pending.
            assert_eq!(w.take_device_time_ms().unwrap(), None);
        }
    }
}

#[test]
fn device_time_scales_with_batch_size() {
    let ctx = gpu_or_skip!();
    if !ctx.timestamps {
        return;
    }
    let c = cfg(WorkloadKind::Heat, 512, 1);
    let mut short = timed(&ctx, &c, 5);
    let mut long = timed(&ctx, &c, 200);
    let (a, b) = (
        short.take_device_time_ms().unwrap().unwrap(),
        long.take_device_time_ms().unwrap().unwrap(),
    );
    assert!(b > a * 4.0, "200 steps ({b} ms) vs 5 steps ({a} ms)");
}

#[test]
fn zero_steps_leaves_no_device_time_and_cpu_has_none() {
    let ctx = gpu_or_skip!();
    let mut w = GpuWorkload::new(ctx, cfg(WorkloadKind::Heat, 32, 0)).unwrap();
    w.step(0).unwrap();
    assert_eq!(w.take_device_time_ms().unwrap(), None);
    let mut cpu = CpuWorkload::new(cfg(WorkloadKind::Heat, 32, 3)).unwrap();
    cpu.step(3).unwrap();
    assert!(!cpu.device_timing_info().available);
    assert_eq!(cpu.take_device_time_ms().unwrap(), None);
}
