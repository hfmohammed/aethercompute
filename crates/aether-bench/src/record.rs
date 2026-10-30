use aether_core::{validate::ErrorReport, DeviceTimingInfo, ExperimentConfig, Summary};
use aether_gpu::AdapterReport;
use serde::{Deserialize, Serialize};

use crate::machine::MachineInfo;

pub const SCHEMA_VERSION: u32 = 3;

/// What state each measured sample starts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchMode {
    /// Re-initialise (untimed) before every sample, so every sample performs
    /// identical work from the identical state. Reproducible; the default.
    #[default]
    ResetEachSample,
    /// Keep evolving one simulation across warm-up and all samples. Models a
    /// sustained run, but later samples start from a different state.
    Continuous,
}

fn legacy_mode() -> BenchMode {
    BenchMode::Continuous
}

impl BenchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ResetEachSample => "reset-each-sample",
            Self::Continuous => "continuous",
        }
    }
}

impl std::str::FromStr for BenchMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "reset" | "reset-each-sample" => Ok(Self::ResetEachSample),
            "continuous" => Ok(Self::Continuous),
            other => Err(format!(
                "unknown bench mode '{other}' (expected reset or continuous)"
            )),
        }
    }
}

/// Documents exactly what each timing column covers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimingScope {
    pub compute: String,
    pub readback: String,
    pub end_to_end: String,
    pub cold_start: String,
}

impl Default for TimingScope {
    fn default() -> Self {
        Self {
            compute: "wall time for `steps` simulation steps, from enqueue until the device is idle (GPU: submit + wait for completion; CPU: the loop itself). Excludes setup, upload and readback.".into(),
            readback: "wall time to copy the final state to host memory (GPU: copy to staging + map + memcpy; CPU: buffer clone).".into(),
            end_to_end: "compute + readback per sample (state is already resident, so no per-sample upload).".into(),
            cold_start: "setup = workload construction incl. buffer upload and pipeline compilation; first_run = the first `steps` steps after construction, before warm-up.".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColdStart {
    pub setup_ms: f64,
    pub first_run_ms: f64,
}

/// Raw samples (milliseconds), one per measured repeat.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Samples {
    pub compute_ms: Vec<f64>,
    pub readback_ms: Vec<f64>,
    pub end_to_end_ms: Vec<f64>,
    /// GPU execution time per sample from timestamp queries (empty when the
    /// backend has none). Kept separate from the host wall-clock `compute_ms`.
    #[serde(default)]
    pub device_compute_ms: Vec<f64>,
    /// `compute_ms - device_compute_ms`, sample by sample.
    #[serde(default)]
    pub host_device_gap_ms: Vec<f64>,
}

fn no_device_timing() -> DeviceTimingInfo {
    DeviceTimingInfo::unavailable("not measured")
}

/// Outcome of one run. Only `SkippedUnavailable` is ever optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecordStatus {
    /// Ran to completion (it may still have failed validation).
    #[default]
    Success,
    /// The backend does not exist on this machine (e.g. no GPU adapter).
    SkippedUnavailable,
    /// The run was attempted and errored (shader compile, device loss, ...).
    Failed,
}

impl RecordStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::SkippedUnavailable => "skipped-unavailable",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchRecord {
    #[serde(default)]
    pub status: RecordStatus,
    pub config: ExperimentConfig,
    /// Backend description, e.g. `gpu (Apple M2, Metal)`.
    pub backend_description: String,
    pub complexity: String,
    pub cold_start: Option<ColdStart>,
    pub samples: Samples,
    pub compute: Option<Summary>,
    pub readback: Option<Summary>,
    pub end_to_end: Option<Summary>,
    /// Device-side (timestamp) execution time summary.
    #[serde(default)]
    pub device_compute: Option<Summary>,
    /// Per-sample `host compute - device compute`: the host/device timing gap.
    /// It contains submission, queueing and synchronisation effects, not just
    /// submission overhead.
    #[serde(default)]
    pub host_device_gap: Option<Summary>,
    /// Whether and how device timing was obtained for this record.
    #[serde(default = "no_device_timing")]
    pub device_timing: DeviceTimingInfo,
    /// Median device time divided by `steps`.
    #[serde(default)]
    pub median_device_step_ms: Option<f64>,
    /// Median compute time divided by `steps`.
    pub median_step_ms: Option<f64>,
    /// Median throughput in millions of updates (particles or cells) per second.
    pub median_mupdates_per_s: Option<f64>,
    /// Output compared with the scalar CPU reference for the same config.
    pub validation: Option<ErrorReport>,
    /// Reset mode: the state after the *last measured sample* compared with
    /// the reference, i.e. validation of what was actually timed.
    #[serde(default)]
    pub final_validation: Option<ErrorReport>,
    /// Number of non-finite values in the final state (all modes).
    #[serde(default)]
    pub final_nonfinite: Option<usize>,
    /// Position of this run in the executed (possibly shuffled) order.
    #[serde(default)]
    pub run_index: usize,
    /// Set when the run could not be performed (e.g. no GPU adapter).
    pub error: Option<String>,
}

impl BenchRecord {
    pub fn failed(config: ExperimentConfig, description: String, error: String) -> Self {
        Self::errored(RecordStatus::Failed, config, description, error)
    }

    /// The backend is not present on this machine.
    pub fn unavailable(config: ExperimentConfig, description: String, error: String) -> Self {
        Self::errored(RecordStatus::SkippedUnavailable, config, description, error)
    }

    fn errored(
        status: RecordStatus,
        config: ExperimentConfig,
        description: String,
        error: String,
    ) -> Self {
        Self {
            status,
            complexity: config.workload.complexity().to_string(),
            config,
            backend_description: description,
            cold_start: None,
            samples: Samples::default(),
            compute: None,
            readback: None,
            end_to_end: None,
            device_compute: None,
            host_device_gap: None,
            device_timing: no_device_timing(),
            median_device_step_ms: None,
            median_step_ms: None,
            median_mupdates_per_s: None,
            validation: None,
            final_validation: None,
            final_nonfinite: None,
            run_index: 0,
            error: Some(error),
        }
    }

    pub fn validated_ok(&self) -> bool {
        self.error.is_none()
            && self.validation.as_ref().is_some_and(|v| v.passed)
            && self.final_validation.as_ref().is_none_or(|v| v.passed)
            && self.final_nonfinite.unwrap_or(0) == 0
    }
}

/// GPU speed-up over each CPU baseline for one (workload, size, workgroup).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Speedup {
    pub workload: String,
    pub size: u32,
    pub workgroup_size: u32,
    pub vs_scalar_compute: Option<f64>,
    pub vs_rayon_compute: Option<f64>,
    pub vs_scalar_end_to_end: Option<f64>,
    pub vs_rayon_end_to_end: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchReport {
    pub schema_version: u32,
    pub git_sha: String,
    pub git_dirty: bool,
    pub timestamp_unix: u64,
    pub machine: MachineInfo,
    pub adapter: Option<AdapterReport>,
    pub timing_scope: TimingScope,
    /// Exact command line that produced the report.
    #[serde(default)]
    pub command: String,
    /// Schema-v1 reports predate the modes and always evolved one simulation.
    #[serde(default = "legacy_mode")]
    pub bench_mode: BenchMode,
    /// Whether backend/workgroup order was randomised within each size.
    #[serde(default)]
    pub shuffled: bool,
    #[serde(default)]
    pub order_seed: u64,
    #[serde(default)]
    pub cooldown_ms: u64,
    /// Device timing was requested to be mandatory (`--require-gpu-timestamps`).
    #[serde(default)]
    pub require_gpu_timestamps: bool,
    /// Labels in the order the runs were executed.
    #[serde(default)]
    pub run_order: Vec<String>,
    pub records: Vec<BenchRecord>,
    pub speedups: Vec<Speedup>,
}

impl BenchReport {
    /// GPU records that ran but have no device timestamps. These only count as
    /// failures when device timing was made mandatory.
    pub fn timestamp_failures(&self) -> Vec<String> {
        self.records
            .iter()
            .filter(|r| {
                r.config.backend == aether_core::BackendKind::Gpu
                    && r.status == RecordStatus::Success
            })
            .filter_map(|r| {
                let label = format!(
                    "{} gpu {}",
                    r.config.workload.as_str(),
                    r.config.size_label()
                );
                if !r.device_timing.available {
                    Some(format!(
                        "{label}: no device timestamps ({})",
                        r.device_timing.reason
                    ))
                } else if r.samples.device_compute_ms.len() != r.samples.compute_ms.len() {
                    Some(format!(
                        "{label}: {} device samples for {} timed samples",
                        r.samples.device_compute_ms.len(),
                        r.samples.compute_ms.len()
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Records that make a run a failure: failed numerical validation, or a
    /// run error (a missing GPU only counts when `require_gpu` is set).
    pub fn failures(&self, require_gpu: bool) -> Vec<String> {
        self.records
            .iter()
            .filter_map(|r| {
                let label = format!(
                    "{} {} {}",
                    r.config.workload.as_str(),
                    r.config.backend.as_str(),
                    r.config.size_label()
                );
                match (&r.error, &r.validation) {
                    (Some(_), _)
                        if r.status == RecordStatus::SkippedUnavailable && !require_gpu =>
                    {
                        None
                    }
                    (Some(e), _) => Some(format!("{label}: {e}")),
                    (None, Some(v)) if !v.passed => {
                        Some(format!("{label}: validation FAILED ({v:?})"))
                    }
                    (None, Some(_))
                        if r.final_validation.as_ref().is_some_and(|f| !f.passed)
                            || r.final_nonfinite.unwrap_or(0) > 0 =>
                    {
                        Some(format!(
                            "{label}: final state invalid ({:?}, {} non-finite)",
                            r.final_validation,
                            r.final_nonfinite.unwrap_or(0)
                        ))
                    }
                    (None, None) => Some(format!("{label}: not validated")),
                    _ => None,
                }
            })
            .collect()
    }
}
