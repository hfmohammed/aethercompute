//! CPU microbenchmarks (criterion). End-to-end and GPU timings live in
//! `aether bench`; these only characterise the reference kernels.

use aether_core::{BackendKind, ExperimentConfig, Workload, WorkloadKind};
use aether_cpu::CpuWorkload;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_workload(c: &mut Criterion, kind: WorkloadKind, sizes: &[u32]) {
    let mut group = c.benchmark_group(kind.as_str());
    for &size in sizes {
        for backend in [BackendKind::Scalar, BackendKind::Rayon] {
            let cfg = ExperimentConfig {
                workload: kind,
                backend,
                count: size,
                grid: size,
                ..Default::default()
            };
            group.throughput(Throughput::Elements(cfg.updates_per_step()));
            group.bench_with_input(BenchmarkId::new(backend.as_str(), size), &cfg, |b, cfg| {
                let mut w = CpuWorkload::new(cfg.clone()).unwrap();
                b.iter(|| w.step(1).unwrap());
            });
        }
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_workload(c, WorkloadKind::ParticlesLocal, &[1_000, 100_000]);
    bench_workload(c, WorkloadKind::NbodyAllpairs, &[256, 1024]);
    bench_workload(c, WorkloadKind::Heat, &[256, 512]);
}

criterion_group!(cpu, benches);
criterion_main!(cpu);
