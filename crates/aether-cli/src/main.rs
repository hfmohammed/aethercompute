//! `aether` command-line interface.

use std::path::PathBuf;
use std::process::ExitCode;

use aether_bench::{
    create_workload, export, machine::MachineInfo, plot, run_sweep, BenchMode, SweepSpec,
};
use aether_core::{
    validate::Tolerance, BackendKind, BoundaryMode, ExperimentConfig, Workload, WorkloadKind,
};
use aether_gpu::{enumerate_adapters, GpuContext};
use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "aether",
    version,
    about = "GPU-accelerated scientific computing engine (wgpu/Metal vs scalar and Rayon CPU)"
)]
struct Cli {
    /// Log level for diagnostics (error, warn, info, debug).
    #[arg(long, global = true, default_value = "warn")]
    log: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print machine and GPU adapter information (limits, optional features).
    Info {
        /// Emit JSON instead of text.
        #[arg(long)]
        json: bool,
        /// Exit non-zero when no GPU adapter is available.
        #[arg(long)]
        require_gpu: bool,
    },
    /// Check CPU Rayon and GPU output against the scalar CPU reference.
    Validate {
        #[command(flatten)]
        common: Common,
        /// Backends to validate: any of rayon,gpu (default: both).
        #[arg(long, value_delimiter = ',', default_value = "rayon,gpu")]
        backend: Vec<BackendKind>,
        /// Fail (exit 1) instead of skipping when no GPU is available.
        #[arg(long)]
        require_gpu: bool,
    },
    /// Benchmark scalar CPU, Rayon CPU and GPU with validation and export.
    Bench {
        #[command(flatten)]
        common: Common,
        /// Backends: all, or a comma list of scalar,rayon,gpu.
        #[arg(long, value_delimiter = ',', default_value = "all")]
        backend: Vec<String>,
        /// Unmeasured warm-up runs (each `--steps` steps).
        #[arg(long)]
        warmup: Option<u32>,
        /// Measured repeats (each `--steps` steps).
        #[arg(long)]
        repeats: Option<u32>,
        /// GPU workgroup sizes to sweep, e.g. 64,128,256.
        #[arg(long, value_delimiter = ',')]
        workgroup_sweep: Vec<u32>,
        /// Write the full report (raw samples included) as JSON.
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Also write a summary CSV.
        #[arg(long)]
        csv: Option<PathBuf>,
        /// Also write SVG charts into this directory.
        #[arg(long)]
        plot_dir: Option<PathBuf>,
        /// Treat a missing GPU as a failure.
        #[arg(long)]
        require_gpu: bool,
        /// `reset` (default): re-initialise before every sample so each does
        /// identical work; `continuous`: keep evolving one simulation.
        #[arg(long, default_value = "reset")]
        bench_mode: BenchMode,
        /// Run backends/workgroups in a fixed order instead of shuffling.
        #[arg(long)]
        no_shuffle: bool,
        /// Seed for the execution-order shuffle. Default: random per run, so
        /// repeated runs do not always put the same backend in the same
        /// thermal position. Always recorded in the report.
        #[arg(long)]
        order_seed: Option<u64>,
        /// Idle milliseconds before each run (thermal recovery).
        #[arg(long, default_value_t = 0)]
        cooldown_ms: u64,
    },
    /// Open the interactive viewer (GPU).
    Viewer {
        #[command(flatten)]
        common: Common,
    },
    /// Render a frame offscreen to a PNG (no window needed).
    Render {
        #[command(flatten)]
        common: Common,
        #[arg(long, short, default_value = "aether.png")]
        output: PathBuf,
        #[arg(long, default_value_t = 1280)]
        width: u32,
        #[arg(long, default_value_t = 720)]
        height: u32,
        /// Steps to simulate before rendering.
        #[arg(long, default_value_t = 200)]
        warm_steps: u32,
    },
    /// Draw SVG charts from a saved benchmark JSON report.
    Plot {
        report: PathBuf,
        #[arg(long, short, default_value = "docs/img")]
        output_dir: PathBuf,
    },
}

/// Options shared by every workload-running subcommand. A JSON `--config`
/// provides the base; explicit flags override it.
#[derive(Args, Clone)]
struct Common {
    /// JSON file with an ExperimentConfig (missing fields use defaults).
    #[arg(long)]
    config: Option<PathBuf>,
    /// particles-local | nbody-allpairs | heat
    #[arg(long, short)]
    workload: Option<WorkloadKind>,
    /// Particle counts (comma list for sweeps).
    #[arg(long, value_delimiter = ',')]
    count: Vec<u32>,
    /// Heat grid edge lengths (comma list for sweeps).
    #[arg(long, value_delimiter = ',')]
    grid: Vec<u32>,
    /// Problem size for whichever workload is selected (count or grid).
    #[arg(long, value_delimiter = ',')]
    size: Vec<u32>,
    /// Simulation steps (per sample for `bench`).
    #[arg(long)]
    steps: Option<u32>,
    #[arg(long)]
    seed: Option<u64>,
    /// Particle timestep (default depends on the workload).
    #[arg(long)]
    dt: Option<f32>,
    /// Force scale / gravitational constant.
    #[arg(long)]
    force_scale: Option<f32>,
    #[arg(long)]
    softening: Option<f32>,
    #[arg(long)]
    damping: Option<f32>,
    /// GPU workgroup size.
    #[arg(long)]
    workgroup: Option<u32>,
    /// GPU step encoding: per-step | batched
    #[arg(long)]
    pass_mode: Option<aether_core::PassMode>,
    /// dirichlet | neumann
    #[arg(long)]
    boundary: Option<BoundaryMode>,
    /// Heat diffusion number alpha*dt/dx^2 (stable up to 0.25 per axis pair).
    #[arg(long)]
    diffusion_number: Option<f32>,
}

impl Common {
    fn base(&self) -> Result<ExperimentConfig> {
        let mut c = match &self.config {
            Some(p) => serde_json::from_slice(
                &std::fs::read(p).with_context(|| format!("reading {}", p.display()))?,
            )
            .with_context(|| format!("parsing {}", p.display()))?,
            None => ExperimentConfig::default(),
        };
        if let Some(w) = self.workload {
            c.workload = w;
        }
        if let Some(v) = self.steps {
            c.steps = v;
        }
        if let Some(v) = self.seed {
            c.seed = v;
        }
        if self.dt.is_some() {
            c.dt = self.dt;
        }
        if self.force_scale.is_some() {
            c.force_scale = self.force_scale;
        }
        if let Some(v) = self.softening {
            c.softening = v;
        }
        if let Some(v) = self.damping {
            c.damping = v;
        }
        if let Some(v) = self.workgroup {
            c.workgroup_size = v;
        }
        if let Some(v) = self.pass_mode {
            c.pass_mode = v;
        }
        if let Some(v) = self.boundary {
            c.boundary = v;
        }
        if let Some(v) = self.diffusion_number {
            c.diffusion_number = v;
        }
        Ok(c)
    }

    /// Sizes to sweep for the (already resolved) workload; empty = config size.
    fn sizes(&self, workload: WorkloadKind) -> Vec<u32> {
        let specific = if workload.is_particles() {
            &self.count
        } else {
            &self.grid
        };
        if !specific.is_empty() {
            specific.clone()
        } else {
            self.size.clone()
        }
    }

    /// Single resolved config (first size), for non-sweeping commands.
    fn single(&self) -> Result<ExperimentConfig> {
        let mut c = self.base()?;
        if let Some(&s) = self.sizes(c.workload).first() {
            if c.workload.is_particles() {
                c.count = s;
            } else {
                c.grid = s;
            }
        }
        c.validate()?;
        Ok(c)
    }
}

fn init_logging(level: &str) {
    let filter = tracing_subscriber::EnvFilter::try_new(level)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn cmd_info(json: bool, require_gpu: bool) -> Result<ExitCode> {
    let machine = MachineInfo::detect();
    let ctx = GpuContext::new();
    let all = enumerate_adapters();
    if json {
        let v = serde_json::json!({
            "machine": machine,
            "selected_adapter": ctx.as_ref().ok().map(|c| &c.report),
            "gpu_error": ctx.as_ref().err().map(|e| e.to_string()),
            "all_adapters": all,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        println!("AetherCompute {}", machine.aether_version);
        println!(
            "  machine : {} ({}) | {} {} | {:.0} GiB RAM | {} logical CPUs | {}",
            machine.chip,
            machine.model,
            machine.os,
            machine.os_version,
            gib(machine.ram_bytes),
            machine.logical_cpus,
            machine.power_source
        );
        println!(
            "  toolchain: {} | {} build",
            machine.rustc, machine.build_profile
        );
        match &ctx {
            Ok(c) => {
                let r = &c.report;
                println!("  GPU     : {} [{} / {}]", r.name, r.backend, r.device_type);
                println!("    driver: {} {}", r.driver, r.driver_info);
                println!(
                    "    max buffer size           : {:.2} GiB",
                    gib(r.limits.max_buffer_size)
                );
                println!(
                    "    max storage binding size  : {:.2} GiB",
                    gib(r.limits.max_storage_buffer_binding_size)
                );
                println!(
                    "    max workgroup size (x)    : {}",
                    r.limits.max_compute_workgroup_size_x
                );
                println!(
                    "    max invocations/workgroup : {}",
                    r.limits.max_compute_invocations_per_workgroup
                );
                println!(
                    "    max workgroups/dimension  : {}",
                    r.limits.max_compute_workgroups_per_dimension
                );
                println!(
                    "    workgroup storage         : {} bytes",
                    r.limits.max_compute_workgroup_storage_size
                );
                println!(
                    "    optional features (not required): {}",
                    r.optional_features.join(", ")
                );
            }
            Err(e) => {
                println!("  GPU     : unavailable - {e}\n    (CPU backends remain fully usable)")
            }
        }
        if all.len() > 1 {
            println!(
                "  all adapters: {}",
                all.iter()
                    .map(|a| format!("{} [{}]", a.name, a.backend))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    Ok(if ctx.is_err() && require_gpu {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}

fn cmd_validate(common: Common, backends: Vec<BackendKind>, require_gpu: bool) -> Result<ExitCode> {
    let cfg = common.single()?;
    let mut reference = {
        let mut c = cfg.clone();
        c.backend = BackendKind::Scalar;
        create_workload(&c, None)?
    };
    reference.step(cfg.steps)?;
    let ref_snap = reference.snapshot()?;
    println!(
        "validate {} {} steps={} seed={} (reference: scalar CPU)",
        cfg.workload.as_str(),
        cfg.size_label(),
        cfg.steps,
        cfg.seed
    );

    let gpu = if backends.contains(&BackendKind::Gpu) {
        Some(GpuContext::new())
    } else {
        None
    };
    let mut failed = false;
    for b in backends
        .iter()
        .copied()
        .filter(|b| *b != BackendKind::Scalar)
    {
        let mut c = cfg.clone();
        c.backend = b;
        let ctx = match (&b, &gpu) {
            (BackendKind::Gpu, Some(Ok(ctx))) => Some(ctx.clone()),
            (BackendKind::Gpu, Some(Err(e))) => {
                if require_gpu {
                    eprintln!("error: GPU required but unavailable: {e}");
                    failed = true;
                } else {
                    println!("  gpu   : SKIPPED (no GPU adapter: {e})");
                }
                continue;
            }
            _ => None,
        };
        let mut w = create_workload(&c, ctx.as_ref())?;
        w.step(cfg.steps)?;
        w.sync()?;
        let tol = if b == BackendKind::Gpu {
            Tolerance::gpu_vs_cpu(cfg.workload, cfg.steps)
        } else {
            Tolerance::EXACT
        };
        let r = w.validate(&ref_snap, tol)?;
        println!(
            "  {:<6}: {}  max_abs={:.3e} rms={:.3e} max_rel={:.3e} nonfinite={} (tol abs={:.1e} rel={:.1e} rms={:.1e})  [{}]",
            b.as_str(), if r.passed { "PASS" } else { "FAIL" },
            r.max_abs, r.rms, r.max_rel, r.nonfinite,
            r.tolerance.abs, r.tolerance.rel, r.tolerance.rms, w.describe()
        );
        failed |= !r.passed;
    }
    Ok(if failed {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    })
}

fn parse_backends(list: &[String]) -> Result<Vec<BackendKind>> {
    let mut out = Vec::new();
    for s in list {
        if s == "all" {
            return Ok(BackendKind::ALL.to_vec());
        }
        let b: BackendKind = s.parse().map_err(|e: String| anyhow::anyhow!(e))?;
        if !out.contains(&b) {
            out.push(b);
        }
    }
    if out.is_empty() {
        bail!("no backends selected");
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn cmd_bench(
    common: Common,
    backend: Vec<String>,
    warmup: Option<u32>,
    repeats: Option<u32>,
    workgroups: Vec<u32>,
    output: Option<PathBuf>,
    csv: Option<PathBuf>,
    plot_dir: Option<PathBuf>,
    require_gpu_flag: bool,
    bench_mode: BenchMode,
    no_shuffle: bool,
    order_seed: Option<u64>,
    cooldown_ms: u64,
) -> Result<ExitCode> {
    let mut base = common.base()?;
    if let Some(w) = warmup {
        base.warmup = w;
    }
    if let Some(r) = repeats {
        base.repeats = r;
    }
    let backends = parse_backends(&backend)?;
    // An explicit `--backend gpu` makes the GPU mandatory.
    let require_gpu = require_gpu_flag || (backends == [BackendKind::Gpu]);
    if base.repeats < 20 || base.warmup < 3 {
        eprintln!(
            "note: methodology targets >= 3 warm-up and >= 20 measured runs (got {} / {})",
            base.warmup, base.repeats
        );
    }
    let sizes = common.sizes(base.workload);
    let spec = SweepSpec {
        base,
        backends,
        sizes,
        workgroups,
        mode: bench_mode,
        shuffle: !no_shuffle,
        order_seed: order_seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1)
        }),
        cooldown_ms,
        command: std::env::args().collect::<Vec<_>>().join(" "),
    };
    let report = run_sweep(&spec, &mut |m| eprintln!("{m}"))?;

    if let Some(p) = &output {
        export::write_json(&report, p).with_context(|| format!("writing {}", p.display()))?;
        eprintln!("wrote {}", p.display());
    }
    if let Some(p) = &csv {
        export::write_csv(&report, p)
            .map_err(|e| anyhow::anyhow!("writing {}: {e}", p.display()))?;
        eprintln!("wrote {}", p.display());
    }
    if let Some(d) = &plot_dir {
        write_plots(&report, d)?;
    }
    print_summary(&report);

    let failures = report.failures(require_gpu);
    for f in &failures {
        eprintln!("FAILED: {f}");
    }
    Ok(if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

fn write_plots(report: &aether_bench::BenchReport, dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let name = report
        .records
        .first()
        .map(|r| r.config.workload.as_str())
        .unwrap_or("bench");
    let t = dir.join(format!("{name}-throughput.svg"));
    std::fs::write(&t, plot::throughput_chart(report))?;
    eprintln!("wrote {}", t.display());
    if !report.speedups.is_empty() {
        let s = dir.join(format!("{name}-speedup.svg"));
        std::fs::write(&s, plot::speedup_chart(report))?;
        eprintln!("wrote {}", s.display());
    }
    Ok(())
}

fn print_summary(r: &aether_bench::BenchReport) {
    println!(
        "\n{} | {} | commit {}{}",
        r.machine.chip,
        r.adapter
            .as_ref()
            .map(|a| format!("{} ({})", a.name, a.backend))
            .unwrap_or_else(|| "no GPU".into()),
        &r.git_sha[..r.git_sha.len().min(7)],
        if r.git_dirty { " (dirty)" } else { "" }
    );
    println!(
        "{:<16} {:>10} {:<8} {:>5} {:>12} {:>12} {:>12} {:>12} {:>10}  valid",
        "workload",
        "size",
        "backend",
        "wg",
        "compute ms",
        "p95 ms",
        "e2e ms",
        "Mupdates/s",
        "ms/step"
    );
    for rec in &r.records {
        let size = if rec.config.workload.is_particles() {
            rec.config.count
        } else {
            rec.config.grid
        };
        match (&rec.compute, &rec.end_to_end) {
            (Some(c), Some(e)) => println!(
                "{:<16} {:>10} {:<8} {:>5} {:>12.3} {:>12.3} {:>12.3} {:>12.1} {:>10.4}  {}",
                rec.config.workload.as_str(),
                size,
                rec.config.backend.as_str(),
                if rec.config.backend == BackendKind::Gpu {
                    rec.config.workgroup_size.to_string()
                } else {
                    "-".into()
                },
                c.median,
                c.p95,
                e.median,
                rec.median_mupdates_per_s.unwrap_or(0.0),
                rec.median_step_ms.unwrap_or(0.0),
                if rec.validated_ok() { "PASS" } else { "FAIL" }
            ),
            _ => println!(
                "{:<16} {:>10} {:<8} {:>5} skipped: {}",
                rec.config.workload.as_str(),
                size,
                rec.config.backend.as_str(),
                "-",
                rec.error.as_deref().unwrap_or("?")
            ),
        }
    }
    for s in &r.speedups {
        let f = |v: Option<f64>| v.map(|x| format!("{x:.1}x")).unwrap_or_else(|| "-".into());
        println!(
            "speedup {} size={} wg={}: vs scalar {} / vs rayon {} (compute); vs scalar {} / vs rayon {} (end-to-end)",
            s.workload, s.size, s.workgroup_size,
            f(s.vs_scalar_compute), f(s.vs_rayon_compute), f(s.vs_scalar_end_to_end), f(s.vs_rayon_end_to_end)
        );
    }
}

fn cmd_render(
    common: Common,
    output: PathBuf,
    width: u32,
    height: u32,
    warm_steps: u32,
) -> Result<ExitCode> {
    let mut cfg = common.single()?;
    cfg.backend = BackendKind::Gpu;
    let ctx = GpuContext::new().context("rendering needs a GPU adapter")?;
    let mut sim = aether_gpu::GpuWorkload::new(ctx.clone(), cfg.clone())?;
    sim.step(warm_steps)?;
    sim.sync()?;
    let view = aether_gpu::View::for_workload(
        cfg.workload,
        cfg.grid,
        cfg.count,
        width as f32 / height as f32,
    );
    let px = aether_gpu::render_to_rgba(&ctx, &sim, width, height, &view)?;
    aether_viewer::screenshot::save_png(&output, width, height, &px)?;
    println!(
        "wrote {} ({width}x{height}, {} after {warm_steps} steps, {})",
        output.display(),
        cfg.workload.as_str(),
        ctx.report.name
    );
    Ok(ExitCode::SUCCESS)
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    init_logging(&cli.log);
    match cli.command {
        Command::Info { json, require_gpu } => cmd_info(json, require_gpu),
        Command::Validate {
            common,
            backend,
            require_gpu,
        } => cmd_validate(common, backend, require_gpu),
        Command::Bench {
            common,
            backend,
            warmup,
            repeats,
            workgroup_sweep,
            output,
            csv,
            plot_dir,
            require_gpu,
            bench_mode,
            no_shuffle,
            order_seed,
            cooldown_ms,
        } => cmd_bench(
            common,
            backend,
            warmup,
            repeats,
            workgroup_sweep,
            output,
            csv,
            plot_dir,
            require_gpu,
            bench_mode,
            no_shuffle,
            order_seed,
            cooldown_ms,
        ),
        Command::Viewer { common } => {
            aether_viewer::run(common.single()?)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Render {
            common,
            output,
            width,
            height,
            warm_steps,
        } => cmd_render(common, output, width, height, warm_steps),
        Command::Plot { report, output_dir } => {
            let r = export::read_json(&report)
                .with_context(|| format!("reading {}", report.display()))?;
            write_plots(&r, &output_dir)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}
