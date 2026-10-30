//! Multi-report aggregation and baseline-vs-candidate comparison.
//!
//! Each report file is one *run* and runs are the top-level replicates: a
//! metric's value for a configuration is the median of the per-run medians,
//! with the min/max across runs as the spread. Raw samples are never pooled,
//! because samples inside a run are autocorrelated.

use std::collections::BTreeMap;

use aether_core::BackendKind;
use serde::Serialize;
use thiserror::Error;

use crate::record::{BenchMode, BenchRecord, BenchReport, RecordStatus};

#[derive(Debug, Error)]
pub enum CompareError {
    #[error("{0}")]
    Incompatible(String),
    #[error("no reports given")]
    Empty,
}

/// Timing series that are always kept apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Metric {
    /// Host wall-clock compute time.
    Compute,
    /// GPU execution time from timestamp queries.
    Device,
    /// Host compute minus device compute.
    Gap,
    Readback,
    EndToEnd,
}

impl Metric {
    pub const ALL: [Metric; 5] = [
        Self::Compute,
        Self::Device,
        Self::Gap,
        Self::Readback,
        Self::EndToEnd,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Device => "device",
            Self::Gap => "gap",
            Self::Readback => "readback",
            Self::EndToEnd => "end-to-end",
        }
    }

    fn median_of(self, r: &BenchRecord) -> Option<f64> {
        match self {
            Self::Compute => r.compute.as_ref(),
            Self::Device => r.device_compute.as_ref(),
            Self::Gap => r.host_device_gap.as_ref(),
            Self::Readback => r.readback.as_ref(),
            Self::EndToEnd => r.end_to_end.as_ref(),
        }
        .map(|s| s.median)
    }
}

impl std::str::FromStr for Metric {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "compute" => Ok(Self::Compute),
            "device" => Ok(Self::Device),
            "gap" => Ok(Self::Gap),
            "readback" => Ok(Self::Readback),
            "end-to-end" | "e2e" => Ok(Self::EndToEnd),
            other => Err(format!(
                "unknown metric '{other}' (compute, device, gap, readback, end-to-end)"
            )),
        }
    }
}

/// What must match for two records to be comparable: workload, size, backend,
/// workgroup and pass mode (GPU only), steps, and every numerical parameter.
/// Warm-up and repeat counts do not change the work, so they are excluded.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Key {
    pub workload: String,
    pub size: u32,
    pub backend: String,
    pub workgroup: u32,
    pub pass_mode: String,
    pub steps: u32,
    pub params: String,
}

impl Key {
    pub fn of(r: &BenchRecord) -> Self {
        let c = &r.config;
        let gpu = c.backend == BackendKind::Gpu;
        Self {
            workload: c.workload.as_str().into(),
            size: if c.workload.is_particles() {
                c.count
            } else {
                c.grid
            },
            backend: c.backend.as_str().into(),
            workgroup: if gpu { c.workgroup_size } else { 0 },
            pass_mode: if gpu {
                format!("{:?}", c.pass_mode).to_lowercase()
            } else {
                "-".into()
            },
            steps: c.steps,
            params: format!(
                "seed={} dt={:?} soft={} force={:?} damp={} bc={:?} r={} alpha={}",
                c.seed,
                c.dt,
                c.softening,
                c.force_scale,
                c.damping,
                c.boundary,
                c.diffusion_number,
                c.alpha
            ),
        }
    }

    /// Everything except the backend axes: identifies one problem.
    fn problem(&self) -> (String, u32, u32, String) {
        (
            self.workload.clone(),
            self.size,
            self.steps,
            self.params.clone(),
        )
    }

    pub fn label(&self) -> String {
        let mut s = format!("{} {} {}", self.workload, self.size, self.backend);
        if self.workgroup > 0 {
            s += &format!(" wg={} {}", self.workgroup, self.pass_mode);
        }
        s
    }
}

/// Per-run medians of one metric and their cross-run summary.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Agg {
    /// One median per run, in input order (runs lacking the record are absent).
    pub runs: Vec<f64>,
    /// Median of the per-run medians.
    pub median: f64,
    pub min: f64,
    pub max: f64,
    /// `(max - min) / median` in percent.
    pub spread_pct: f64,
}

impl Agg {
    fn new(runs: Vec<f64>) -> Option<Self> {
        if runs.is_empty() {
            return None;
        }
        let mut s = runs.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = aether_core::stats::percentile_sorted(&s, 0.5);
        let (min, max) = (s[0], s[s.len() - 1]);
        Some(Self {
            runs,
            median,
            min,
            max,
            spread_pct: if median > 0.0 {
                (max - min) / median * 100.0
            } else {
                0.0
            },
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub key: Key,
    pub metrics: BTreeMap<Metric, Agg>,
    /// Runs that contributed a validated record for this key.
    pub n_runs: usize,
    /// Largest GPU-vs-CPU max abs error over the contributing runs.
    pub max_abs_err: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Aggregate {
    pub n_runs: usize,
    pub bench_mode: BenchMode,
    pub order_seeds: Vec<u64>,
    pub git_shas: Vec<String>,
    pub rows: Vec<Row>,
    pub warnings: Vec<String>,
    #[serde(skip)]
    runs_by_key: BTreeMap<Key, BTreeMap<Metric, Vec<(usize, f64)>>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Fractional slowdown that counts as a regression (0.10 = 10%).
    pub threshold: f64,
    pub metric: Metric,
    /// Allow mixing reset and continuous (or legacy v1) reports.
    pub allow_mode_mismatch: bool,
    /// Treat the GPU pass mode as a value to compare rather than part of the
    /// key (for per-step vs batched experiments).
    pub ignore_pass_mode: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            threshold: 0.10,
            metric: Metric::Compute,
            allow_mode_mismatch: false,
            ignore_pass_mode: false,
        }
    }
}

fn mode_label(r: &BenchReport) -> String {
    if r.schema_version < 2 {
        format!(
            "{} (legacy schema v{})",
            r.bench_mode.as_str(),
            r.schema_version
        )
    } else {
        r.bench_mode.as_str().to_string()
    }
}

/// Environment warnings for one group of reports. `name` prefixes each line.
fn environment_warnings(name: &str, reports: &[BenchReport]) -> Vec<String> {
    let mut w = Vec::new();
    let n_dirty = reports.iter().filter(|r| r.git_dirty).count();
    if n_dirty > 0 {
        w.push(format!(
            "{name}: {n_dirty} report(s) from a dirty working tree"
        ));
    }
    for r in reports {
        if r.machine.build_profile != "release" {
            w.push(format!(
                "{name}: {} build, not release",
                r.machine.build_profile
            ));
            break;
        }
    }
    if reports
        .iter()
        .any(|r| r.machine.power_source.to_lowercase().contains("battery"))
    {
        w.push(format!("{name}: some runs were on battery power"));
    }
    let uniq = |f: &dyn Fn(&BenchReport) -> String| {
        let mut v: Vec<String> = reports.iter().map(f).collect();
        v.sort();
        v.dedup();
        v
    };
    for (what, vals) in [
        ("machine chip", uniq(&|r| r.machine.chip.clone())),
        ("OS version", uniq(&|r| r.machine.os_version.clone())),
        ("rustc", uniq(&|r| r.machine.rustc.clone())),
        (
            "GPU adapter",
            uniq(&|r| {
                r.adapter
                    .as_ref()
                    .map(|a| a.name.clone())
                    .unwrap_or_default()
            }),
        ),
        ("git commit", uniq(&|r| r.git_sha.chars().take(7).collect())),
    ] {
        if vals.len() > 1 {
            w.push(format!(
                "{name}: runs differ in {what}: {}",
                vals.join(", ")
            ));
        }
    }
    let mut seeds: Vec<u64> = reports.iter().map(|r| r.order_seed).collect();
    seeds.sort();
    seeds.dedup();
    if reports.len() > 1 && seeds.len() < reports.len() && reports.iter().any(|r| r.shuffled) {
        w.push(format!(
            "{name}: runs reuse an order seed; use a different seed per run"
        ));
    }
    w
}

/// Aggregate several reports (runs) of the same experiment.
pub fn aggregate(
    name: &str,
    reports: &[BenchReport],
    opts: &Options,
) -> Result<Aggregate, CompareError> {
    let first = reports.first().ok_or(CompareError::Empty)?;
    if !opts.allow_mode_mismatch {
        if let Some(r) = reports.iter().find(|r| r.bench_mode != first.bench_mode) {
            return Err(CompareError::Incompatible(format!(
                "{name}: mixes benchmark modes ({} vs {}); pass --allow-mode-mismatch to override",
                mode_label(first),
                mode_label(r)
            )));
        }
    }

    let mut by_key: BTreeMap<Key, BTreeMap<Metric, BTreeMap<usize, f64>>> = BTreeMap::new();
    let mut err: BTreeMap<Key, f64> = BTreeMap::new();
    let mut excluded = 0usize;
    for (i, rep) in reports.iter().enumerate() {
        for r in &rep.records {
            if r.status != RecordStatus::Success || !r.validated_ok() {
                excluded += 1;
                continue;
            }
            let mut key = Key::of(r);
            if opts.ignore_pass_mode && key.pass_mode != "-" {
                key.pass_mode = "*".into();
            }
            let e = [&r.validation, &r.final_validation]
                .into_iter()
                .flatten()
                .map(|v| v.max_abs)
                .fold(0.0, f64::max);
            let slot = err.entry(key.clone()).or_insert(0.0);
            *slot = slot.max(e);
            for m in Metric::ALL {
                if let Some(v) = m.median_of(r) {
                    by_key
                        .entry(key.clone())
                        .or_default()
                        .entry(m)
                        .or_default()
                        .insert(i, v);
                }
            }
        }
    }

    let mut warnings = environment_warnings(name, reports);
    if reports.len() == 1 {
        warnings.push(format!(
            "{name}: a single run; spread and regression significance need >= 2 runs (3 recommended)"
        ));
    }
    if reports.iter().any(|r| r.schema_version < 2) {
        warnings.push(format!(
            "{name}: contains legacy schema-v1 reports (continuous mode, no final-state validation)"
        ));
    }
    if excluded > 0 {
        warnings.push(format!(
            "{name}: {excluded} failed, skipped or unvalidated record(s) excluded"
        ));
    }

    let mut rows = Vec::new();
    let mut runs_by_key = BTreeMap::new();
    for (key, metrics) in by_key {
        let n_runs = metrics.values().map(|m| m.len()).max().unwrap_or(0);
        let mut agg = BTreeMap::new();
        let mut raw = BTreeMap::new();
        for (m, per_run) in metrics {
            raw.insert(m, per_run.iter().map(|(i, v)| (*i, *v)).collect::<Vec<_>>());
            if let Some(a) = Agg::new(per_run.values().copied().collect()) {
                agg.insert(m, a);
            }
        }
        if n_runs < reports.len() {
            warnings.push(format!(
                "{name}: {} present in only {n_runs}/{} runs",
                key.label(),
                reports.len()
            ));
        }
        rows.push(Row {
            max_abs_err: err.get(&key).copied().unwrap_or(0.0),
            key: key.clone(),
            metrics: agg,
            n_runs,
        });
        runs_by_key.insert(key, raw);
    }

    let mut seeds: Vec<u64> = reports.iter().map(|r| r.order_seed).collect();
    seeds.dedup();
    Ok(Aggregate {
        n_runs: reports.len(),
        bench_mode: first.bench_mode,
        order_seeds: seeds,
        git_shas: reports
            .iter()
            .map(|r| r.git_sha.chars().take(7).collect())
            .collect(),
        rows,
        warnings,
        runs_by_key,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    /// Slower than the threshold and the run ranges do not overlap.
    Regression,
    Improvement,
    Unchanged,
    /// Beyond the threshold, but run-to-run spread overlaps (or too few runs).
    Inconclusive,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Regression => "REGRESSION",
            Self::Improvement => "improvement",
            Self::Unchanged => "unchanged",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompareRow {
    pub key: Key,
    pub baseline: Agg,
    pub candidate: Agg,
    /// `candidate / baseline` of the median-of-medians; > 1 means slower.
    pub ratio: f64,
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    pub metric: Metric,
    pub threshold: f64,
    pub baseline: Aggregate,
    pub candidate: Aggregate,
    pub rows: Vec<CompareRow>,
    pub warnings: Vec<String>,
}

impl Comparison {
    pub fn regressions(&self) -> Vec<&CompareRow> {
        self.rows
            .iter()
            .filter(|r| r.verdict == Verdict::Regression)
            .collect()
    }

    pub fn inconclusive(&self) -> Vec<&CompareRow> {
        self.rows
            .iter()
            .filter(|r| r.verdict == Verdict::Inconclusive)
            .collect()
    }
}

fn verdict(base: &Agg, cand: &Agg, ratio: f64, threshold: f64) -> Verdict {
    let multi = base.runs.len() >= 2 && cand.runs.len() >= 2;
    if ratio > 1.0 + threshold {
        // A regression needs every candidate run slower than every baseline run.
        if multi && cand.min > base.max {
            Verdict::Regression
        } else {
            Verdict::Inconclusive
        }
    } else if ratio < 1.0 / (1.0 + threshold) {
        if multi && cand.max < base.min {
            Verdict::Improvement
        } else {
            Verdict::Inconclusive
        }
    } else {
        Verdict::Unchanged
    }
}

/// Compare candidate runs with baseline runs on one metric.
pub fn compare(
    baseline: &[BenchReport],
    candidate: &[BenchReport],
    opts: &Options,
) -> Result<Comparison, CompareError> {
    let b = aggregate("baseline", baseline, opts)?;
    let c = aggregate("candidate", candidate, opts)?;
    if b.bench_mode != c.bench_mode && !opts.allow_mode_mismatch {
        return Err(CompareError::Incompatible(format!(
            "baseline is {} but candidate is {}; reset and continuous results measure different work \
             (pass --allow-mode-mismatch to override)",
            mode_label(&baseline[0]),
            mode_label(&candidate[0])
        )));
    }

    let mut warnings: Vec<String> = b.warnings.iter().chain(&c.warnings).cloned().collect();
    let both: Vec<BenchReport> = baseline.iter().chain(candidate).cloned().collect();
    for w in environment_warnings("baseline+candidate", &both) {
        // Within-group diffs are already reported; keep only cross-group ones.
        if w.contains("differ in") && !warnings.contains(&w) {
            warnings.push(w);
        }
    }

    let base_rows: BTreeMap<&Key, &Row> = b.rows.iter().map(|r| (&r.key, r)).collect();
    let mut rows = Vec::new();
    let mut unmatched_cand = 0;
    for cr in &c.rows {
        let Some(br) = base_rows.get(&cr.key) else {
            unmatched_cand += 1;
            continue;
        };
        let (Some(ba), Some(ca)) = (br.metrics.get(&opts.metric), cr.metrics.get(&opts.metric))
        else {
            continue;
        };
        let ratio = ca.median / ba.median;
        rows.push(CompareRow {
            key: cr.key.clone(),
            verdict: verdict(ba, ca, ratio, opts.threshold),
            baseline: ba.clone(),
            candidate: ca.clone(),
            ratio,
        });
    }
    if rows.is_empty() {
        return Err(CompareError::Incompatible(format!(
            "no comparable records: baseline and candidate share no configuration with a '{}' \
             metric (workload, size, backend, workgroup, pass mode, steps and numerical \
             parameters must all match)",
            opts.metric.as_str()
        )));
    }
    let unmatched_base = b.rows.len().saturating_sub(rows.len());
    if unmatched_cand > 0 || unmatched_base > 0 {
        warnings.push(format!(
            "{unmatched_base} baseline and {unmatched_cand} candidate configuration(s) had no \
             counterpart and were not compared"
        ));
    }
    Ok(Comparison {
        metric: opts.metric,
        threshold: opts.threshold,
        baseline: b,
        candidate: c,
        rows,
        warnings,
    })
}

// ---------------------------------------------------------------- speedups

/// GPU-vs-CPU speed-up for one problem, aggregated across runs.
#[derive(Debug, Clone, Serialize)]
pub struct SpeedRow {
    pub workload: String,
    pub size: u32,
    pub steps: u32,
    pub scalar_ms: Option<f64>,
    pub rayon_ms: Option<f64>,
    pub gpu_ms: f64,
    pub gpu_workgroup: u32,
    pub gpu_device_ms: Option<f64>,
    pub vs_scalar: Option<Agg>,
    pub vs_rayon: Option<Agg>,
    pub vs_rayon_end_to_end: Option<Agg>,
    pub gpu_mupdates_per_s: Option<f64>,
    pub max_abs_err: f64,
}

fn run_vals(a: &Aggregate, k: &Key, m: Metric) -> BTreeMap<usize, f64> {
    a.runs_by_key
        .get(k)
        .and_then(|x| x.get(&m))
        .map(|v| v.iter().copied().collect())
        .unwrap_or_default()
}

fn ratios(num: &BTreeMap<usize, f64>, den: &BTreeMap<usize, f64>) -> Option<Agg> {
    Agg::new(
        den.iter()
            .filter_map(|(i, d)| num.get(i).map(|n| n / d))
            .collect(),
    )
}

/// Per problem: the best GPU configuration (lowest median compute time over
/// workgroup size and pass mode) and its per-run speed-ups over scalar and
/// Rayon. Speed-ups are computed run by run, then summarised across runs.
pub fn speedups(a: &Aggregate) -> Vec<SpeedRow> {
    let mut problems: BTreeMap<_, Vec<&Row>> = BTreeMap::new();
    for r in &a.rows {
        problems.entry(r.key.problem()).or_default().push(r);
    }
    let mut out = Vec::new();
    for ((workload, size, steps, _), rows) in problems {
        let med = |r: &&Row| r.metrics.get(&Metric::Compute).map(|m| m.median);
        let cpu = |backend: &str| rows.iter().find(|r| r.key.backend == backend).copied();
        let Some(best) = rows
            .iter()
            .filter(|r| r.key.backend == "gpu" && med(r).is_some())
            .min_by(|x, y| med(x).partial_cmp(&med(y)).unwrap())
        else {
            continue;
        };
        let gpu_c = run_vals(a, &best.key, Metric::Compute);
        let gpu_e = run_vals(a, &best.key, Metric::EndToEnd);
        let (scalar, rayon) = (cpu("scalar"), cpu("rayon"));
        let rel = |r: Option<&Row>, m: Metric, gpu: &BTreeMap<usize, f64>| {
            r.and_then(|r| ratios(&run_vals(a, &r.key, m), gpu))
        };
        let gpu_ms = best.metrics[&Metric::Compute].median;
        out.push(SpeedRow {
            workload: workload.clone(),
            size,
            steps,
            scalar_ms: scalar
                .and_then(|r| r.metrics.get(&Metric::Compute))
                .map(|m| m.median),
            rayon_ms: rayon
                .and_then(|r| r.metrics.get(&Metric::Compute))
                .map(|m| m.median),
            gpu_ms,
            gpu_workgroup: best.key.workgroup,
            gpu_device_ms: best.metrics.get(&Metric::Device).map(|m| m.median),
            vs_scalar: rel(scalar, Metric::Compute, &gpu_c),
            vs_rayon: rel(rayon, Metric::Compute, &gpu_c),
            vs_rayon_end_to_end: rel(rayon, Metric::EndToEnd, &gpu_e),
            gpu_mupdates_per_s: {
                let units = if workload.starts_with("heat") {
                    size as f64 * size as f64
                } else {
                    size as f64
                };
                Some(units * steps as f64 / (gpu_ms / 1e3) / 1e6)
            },
            max_abs_err: best.max_abs_err,
        });
    }
    out
}

// ------------------------------------------------------------------ output

fn fmt_runs(a: &Agg) -> String {
    a.runs
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(" / ")
}

fn warn_block(warnings: &[String], md: bool) -> String {
    if warnings.is_empty() {
        return String::new();
    }
    let mut s = String::new();
    if md {
        s += "**Warnings**\n\n";
    } else {
        s += "warnings:\n";
    }
    for w in warnings {
        s += &if md {
            format!("- {w}\n")
        } else {
            format!("  ! {w}\n")
        };
    }
    s + "\n"
}

pub fn aggregate_text(a: &Aggregate, metric: Metric) -> String {
    let mut s = format!(
        "{} run(s), mode {}, order seeds {:?}, commits {}\n{}",
        a.n_runs,
        a.bench_mode.as_str(),
        a.order_seeds,
        a.git_shas.join(","),
        warn_block(&a.warnings, false)
    );
    s += &format!(
        "{:<48} {:>5} {:>12} {:>12} {:>12} {:>8}  per-run medians ({})\n",
        "configuration",
        "runs",
        "median ms",
        "min ms",
        "max ms",
        "spread",
        metric.as_str()
    );
    for r in &a.rows {
        if let Some(m) = r.metrics.get(&metric) {
            s += &format!(
                "{:<48} {:>5} {:>12.4} {:>12.4} {:>12.4} {:>7.1}%  {}\n",
                r.key.label(),
                r.n_runs,
                m.median,
                m.min,
                m.max,
                m.spread_pct,
                fmt_runs(m)
            );
        }
    }
    s
}

pub fn comparison_text(c: &Comparison) -> String {
    let mut s = format!(
        "metric: {} (median of per-run medians), threshold {:.0}%, baseline {} run(s) vs candidate {} run(s)\n{}",
        c.metric.as_str(),
        c.threshold * 100.0,
        c.baseline.n_runs,
        c.candidate.n_runs,
        warn_block(&c.warnings, false)
    );
    s += &format!(
        "{:<48} {:>11} {:>11} {:>8}  verdict\n",
        "configuration", "baseline ms", "candidate ms", "ratio"
    );
    for r in &c.rows {
        s += &format!(
            "{:<48} {:>11.4} {:>11.4} {:>7.3}x  {}\n",
            r.key.label(),
            r.baseline.median,
            r.candidate.median,
            r.ratio,
            r.verdict.as_str()
        );
    }
    let (reg, inc) = (c.regressions().len(), c.inconclusive().len());
    s += &format!(
        "{} regression(s), {} inconclusive, {} compared\n",
        reg,
        inc,
        c.rows.len()
    );
    s
}

pub fn comparison_markdown(c: &Comparison) -> String {
    let mut s = format!(
        "## Benchmark comparison ({} metric)\n\nBaseline: {} run(s), candidate: {} run(s); values are the median of per-run medians; regression threshold {:.0}%. \
         A regression needs every candidate run slower than every baseline run.\n\n{}",
        c.metric.as_str(),
        c.baseline.n_runs,
        c.candidate.n_runs,
        c.threshold * 100.0,
        warn_block(&c.warnings, true)
    );
    s += "| configuration | baseline (ms) | candidate (ms) | ratio | baseline runs | candidate runs | verdict |\n|---|---|---|---|---|---|---|\n";
    for r in &c.rows {
        s += &format!(
            "| {} | {:.4} | {:.4} | {:.3}x | {} | {} | {} |\n",
            r.key.label(),
            r.baseline.median,
            r.candidate.median,
            r.ratio,
            fmt_runs(&r.baseline),
            fmt_runs(&r.candidate),
            r.verdict.as_str()
        );
    }
    s += &format!(
        "\n**{} regression(s)**, {} inconclusive, {} compared.\n",
        c.regressions().len(),
        c.inconclusive().len(),
        c.rows.len()
    );
    s
}

fn range(a: &Option<Agg>) -> String {
    match a {
        Some(a) if a.runs.len() > 1 => format!("{:.1}x ({:.1}-{:.1})", a.median, a.min, a.max),
        Some(a) => format!("{:.1}x", a.median),
        None => "-".into(),
    }
}

/// README-style tables per workload from an aggregate. Speed-ups are medians
/// of per-run ratios with the min-max across runs in parentheses.
pub fn aggregate_markdown(a: &Aggregate) -> String {
    let mut s = format!(
        "{} run(s), `{}` mode, order seeds {:?}, commits {}. Times are the median of per-run medians (host wall-clock, {} steps/sample where stated); \
         speed-ups are the median of per-run ratios with the min-max across runs in parentheses; GPU is the best workgroup size / pass mode per size.\n\n{}",
        a.n_runs,
        a.bench_mode.as_str(),
        a.order_seeds,
        a.git_shas.join(","),
        a.rows.first().map(|r| r.key.steps).unwrap_or(0),
        warn_block(&a.warnings, true)
    );
    let table = speedups(a);
    let mut names: Vec<&str> = table.iter().map(|r| r.workload.as_str()).collect();
    names.dedup();
    for w in names {
        s += &format!(
            "### {w}\n\n| size | scalar (ms) | Rayon (ms) | GPU (ms, best wg) | GPU device (ms) | GPU vs scalar | GPU vs Rayon | GPU vs Rayon end-to-end | GPU M updates/s | max abs err (GPU vs CPU) |\n|---|---|---|---|---|---|---|---|---|---|\n"
        );
        for r in table.iter().filter(|r| r.workload == w) {
            let ms = |v: Option<f64>| v.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
            s += &format!(
                "| {} | {} | {} | {:.2} (wg {}) | {} | {} | {} | {} | {} | {:.1e} |\n",
                r.size,
                ms(r.scalar_ms),
                ms(r.rayon_ms),
                r.gpu_ms,
                r.gpu_workgroup,
                ms(r.gpu_device_ms),
                range(&r.vs_scalar),
                range(&r.vs_rayon),
                range(&r.vs_rayon_end_to_end),
                r.gpu_mupdates_per_s
                    .map(|v| format!("{v:.0}"))
                    .unwrap_or_default(),
                r.max_abs_err
            );
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{run_sweep, SweepSpec};
    use aether_core::{ExperimentConfig, Summary, WorkloadKind};

    fn base_report() -> BenchReport {
        let base = ExperimentConfig {
            workload: WorkloadKind::Heat,
            grid: 16,
            steps: 3,
            warmup: 1,
            repeats: 3,
            ..Default::default()
        };
        let mut r = run_sweep(
            &SweepSpec {
                base,
                backends: vec![BackendKind::Scalar, BackendKind::Rayon],
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
        // Make the fixture independent of the machine running the tests.
        r.git_dirty = false;
        r.machine.build_profile = "release".into();
        r.machine.power_source = "AC Power".into();
        for (i, rec) in r.records.iter_mut().enumerate() {
            set_compute(rec, 10.0 * (i + 1) as f64);
        }
        r
    }

    fn set_compute(rec: &mut BenchRecord, median: f64) {
        rec.compute = Summary::from_samples(&[median; 5]);
        rec.end_to_end = Summary::from_samples(&[median + 1.0; 5]);
    }

    fn scaled(r: &BenchReport, f: f64, seed: u64) -> BenchReport {
        let mut r = r.clone();
        r.order_seed = seed;
        for rec in &mut r.records {
            let m = rec.compute.as_ref().unwrap().median;
            set_compute(rec, m * f);
        }
        r
    }

    fn runs(r: &BenchReport, factors: &[f64]) -> Vec<BenchReport> {
        factors
            .iter()
            .enumerate()
            .map(|(i, f)| scaled(r, *f, i as u64 + 1))
            .collect()
    }

    #[test]
    fn aggregates_median_of_run_medians_with_spread() {
        let r = base_report();
        let a = aggregate("x", &runs(&r, &[1.0, 1.2, 1.1]), &Options::default()).unwrap();
        assert_eq!(a.n_runs, 3);
        let row = a.rows.iter().find(|r| r.key.backend == "scalar").unwrap();
        let m = &row.metrics[&Metric::Compute];
        assert_eq!(m.runs.len(), 3);
        assert!((m.median - 11.0).abs() < 1e-9, "{m:?}");
        assert!((m.min - 10.0).abs() < 1e-9 && (m.max - 12.0).abs() < 1e-9);
        assert!((m.spread_pct - 200.0 / 11.0).abs() < 1e-6);
        // CPU rows have no device series; metrics stay separate.
        assert!(!row.metrics.contains_key(&Metric::Device));
        assert!(row.metrics.contains_key(&Metric::EndToEnd));
    }

    #[test]
    fn rejects_mixed_and_cross_mode_comparisons() {
        let r = base_report();
        let mut cont = r.clone();
        cont.bench_mode = BenchMode::Continuous;
        let opts = Options::default();
        assert!(aggregate("x", &[r.clone(), cont.clone()], &opts).is_err());
        let e = compare(std::slice::from_ref(&r), &[cont.clone()], &opts).unwrap_err();
        assert!(e.to_string().contains("--allow-mode-mismatch"));
        let ok = Options {
            allow_mode_mismatch: true,
            ..opts
        };
        assert!(compare(std::slice::from_ref(&r), &[cont], &ok).is_ok());
        // A legacy schema-v1 report is continuous and is called out as legacy.
        let mut legacy = r.clone();
        legacy.schema_version = 1;
        legacy.bench_mode = BenchMode::Continuous;
        let e = compare(&[r], &[legacy], &opts).unwrap_err().to_string();
        assert!(e.contains("legacy schema v1"), "{e}");
    }

    #[test]
    fn flags_definite_regression_and_exits_nonzero_material() {
        let r = base_report();
        let c = compare(
            &runs(&r, &[1.0, 1.02, 0.99]),
            &runs(&r, &[1.3, 1.32, 1.28]),
            &Options::default(),
        )
        .unwrap();
        assert!(c.rows.iter().all(|x| x.verdict == Verdict::Regression));
        assert_eq!(c.regressions().len(), c.rows.len());
        assert!(c.rows[0].ratio > 1.25);
    }

    #[test]
    fn overlapping_runs_are_inconclusive_not_regressions() {
        let r = base_report();
        // Median is 1.15x slower but a candidate run is faster than a baseline run.
        let c = compare(
            &runs(&r, &[1.0, 1.0, 1.3]),
            &runs(&r, &[1.15, 1.2, 0.95]),
            &Options::default(),
        )
        .unwrap();
        assert!(c.regressions().is_empty());
        assert!(!c.inconclusive().is_empty());
    }

    #[test]
    fn unchanged_and_improvement() {
        let r = base_report();
        let same = compare(
            &runs(&r, &[1.0, 1.01, 0.99]),
            &runs(&r, &[1.02, 1.0, 1.03]),
            &Options::default(),
        )
        .unwrap();
        assert!(same.rows.iter().all(|x| x.verdict == Verdict::Unchanged));
        let better = compare(
            &runs(&r, &[1.0, 1.01, 0.99]),
            &runs(&r, &[0.6, 0.62, 0.61]),
            &Options::default(),
        )
        .unwrap();
        assert!(better
            .rows
            .iter()
            .all(|x| x.verdict == Verdict::Improvement));
    }

    #[test]
    fn single_runs_cannot_prove_a_regression() {
        let r = base_report();
        let c = compare(&runs(&r, &[1.0]), &runs(&r, &[2.0]), &Options::default()).unwrap();
        assert!(c.regressions().is_empty());
        assert!(c.rows.iter().all(|x| x.verdict == Verdict::Inconclusive));
        assert!(c.warnings.iter().any(|w| w.contains("single run")));
    }

    #[test]
    fn pass_modes_align_only_when_ignored() {
        let r = base_report();
        let with_gpu = |mode: aether_core::PassMode, ms: f64| {
            let mut rep = r.clone();
            let mut gpu = rep.records[0].clone();
            gpu.config.backend = BackendKind::Gpu;
            gpu.config.pass_mode = mode;
            set_compute(&mut gpu, ms);
            rep.records = vec![gpu];
            rep
        };
        let per_step = with_gpu(aether_core::PassMode::PerStep, 8.0);
        let batched = with_gpu(aether_core::PassMode::Batched, 2.0);
        let strict = Options::default();
        assert!(compare(
            std::slice::from_ref(&per_step),
            std::slice::from_ref(&batched),
            &strict
        )
        .is_err());
        let loose = Options {
            ignore_pass_mode: true,
            ..strict
        };
        let c = compare(
            &[per_step.clone(), per_step],
            &[batched.clone(), batched],
            &loose,
        )
        .unwrap();
        assert_eq!(c.rows.len(), 1);
        assert!((c.rows[0].ratio - 0.25).abs() < 1e-9);
        assert_eq!(c.rows[0].verdict, Verdict::Improvement);
    }

    #[test]
    fn unvalidated_and_failed_records_are_excluded() {
        let r = base_report();
        let mut bad = r.clone();
        bad.records[0].validation.as_mut().unwrap().passed = false;
        let a = aggregate("x", &[bad], &Options::default()).unwrap();
        assert_eq!(a.rows.len(), r.records.len() - 1);
        assert!(a.warnings.iter().any(|w| w.contains("excluded")));
    }

    #[test]
    fn mismatched_configurations_do_not_compare() {
        let r = base_report();
        let mut other = r.clone();
        for rec in &mut other.records {
            rec.config.steps += 1;
        }
        let e = compare(std::slice::from_ref(&r), &[other], &Options::default()).unwrap_err();
        assert!(e.to_string().contains("no comparable records"));
        // Partial overlap compares the overlap and warns about the rest.
        let mut partial = r.clone();
        partial.records[1].config.seed += 1;
        let c = compare(&[r], &[partial], &Options::default()).unwrap();
        assert_eq!(c.rows.len(), 1);
        assert!(c.warnings.iter().any(|w| w.contains("no counterpart")));
    }

    #[test]
    fn warns_about_environment_problems() {
        let r = base_report();
        let mut dirty = r.clone();
        dirty.git_dirty = true;
        dirty.machine.build_profile = "debug".into();
        dirty.machine.power_source = "Battery Power".into();
        let mut other_chip = r.clone();
        other_chip.machine.chip = "Apple M3".into();
        let c = compare(&[r], &[dirty, other_chip], &Options::default()).unwrap();
        let all = c.warnings.join("\n");
        for needle in ["dirty", "debug build", "battery", "machine chip"] {
            assert!(all.contains(needle), "missing '{needle}' in:\n{all}");
        }
    }

    #[test]
    fn device_and_gap_metrics_are_aggregated_separately() {
        let r = base_report();
        let mut rep = r.clone();
        let rec = &mut rep.records[0];
        rec.device_compute = Summary::from_samples(&[2.0; 5]);
        rec.host_device_gap = Summary::from_samples(&[0.5; 5]);
        let a = aggregate("x", &[rep], &Options::default()).unwrap();
        let m = &a
            .rows
            .iter()
            .find(|r| r.key.backend == "scalar")
            .unwrap()
            .metrics;
        assert_eq!(m[&Metric::Device].median, 2.0);
        assert_eq!(m[&Metric::Gap].median, 0.5);
        assert_ne!(m[&Metric::Compute].median, m[&Metric::Device].median);
    }

    #[test]
    fn speedup_tables_use_per_run_ratios() {
        let r = base_report();
        // Fake a GPU row from the scalar row: 5x faster than scalar.
        let mut with_gpu = r.clone();
        let mut gpu = with_gpu.records[0].clone();
        gpu.config.backend = BackendKind::Gpu;
        set_compute(&mut gpu, 2.0);
        with_gpu.records.push(gpu);
        let a = aggregate("x", &[with_gpu.clone(), with_gpu], &Options::default()).unwrap();
        let t = speedups(&a);
        assert_eq!(t.len(), 1);
        assert!((t[0].vs_scalar.as_ref().unwrap().median - 5.0).abs() < 1e-9);
        assert!((t[0].vs_rayon.as_ref().unwrap().median - 10.0).abs() < 1e-9);
        let md = aggregate_markdown(&a);
        assert!(md.contains("### heat") && md.contains("5.0x"));
    }

    #[test]
    fn renderers_produce_tables_and_json() {
        let r = base_report();
        let c = compare(
            &runs(&r, &[1.0, 1.0]),
            &runs(&r, &[1.0, 1.0]),
            &Options::default(),
        )
        .unwrap();
        assert!(comparison_markdown(&c).contains("| configuration |"));
        assert!(comparison_text(&c).contains("0 regression(s)"));
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("\"verdict\":\"unchanged\""));
        let a = aggregate("x", &runs(&r, &[1.0, 1.1]), &Options::default()).unwrap();
        assert!(aggregate_text(&a, Metric::Compute).contains("per-run medians"));
    }
}
