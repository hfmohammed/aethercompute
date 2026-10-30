use std::fs;
use std::path::Path;

use crate::record::BenchReport;

/// Write the full report (including raw samples) as pretty JSON.
pub fn write_json(report: &BenchReport, path: &Path) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, serde_json::to_string_pretty(report).map_err(std::io::Error::other)? + "\n")
}

pub fn read_json(path: &Path) -> std::io::Result<BenchReport> {
    serde_json::from_slice(&fs::read(path)?).map_err(std::io::Error::other)
}

/// One row per record with the summary statistics (raw samples stay in JSON).
pub fn to_csv(report: &BenchReport) -> Result<String, csv::Error> {
    let mut w = csv::Writer::from_writer(vec![]);
    w.write_record([
        "workload", "backend", "size", "steps", "workgroup_size", "seed", "samples",
        "compute_median_ms", "compute_p95_ms", "compute_p99_ms", "compute_min_ms", "compute_stddev_ms",
        "compute_ci95_half_ms", "readback_median_ms", "end_to_end_median_ms", "median_step_ms",
        "median_mupdates_per_s", "setup_ms", "first_run_ms", "max_abs_err", "rms_err", "validation",
        "git_sha", "adapter", "error",
    ])?;
    let f = |v: Option<f64>| v.map(|x| format!("{x:.6}")).unwrap_or_default();
    for r in &report.records {
        let size = if r.config.workload.is_particles() { r.config.count } else { r.config.grid };
        let c = r.compute.as_ref();
        w.write_record([
            r.config.workload.as_str().to_string(),
            r.config.backend.as_str().to_string(),
            size.to_string(),
            r.config.steps.to_string(),
            r.config.workgroup_size.to_string(),
            r.config.seed.to_string(),
            c.map(|s| s.n.to_string()).unwrap_or_default(),
            f(c.map(|s| s.median)),
            f(c.map(|s| s.p95)),
            f(c.map(|s| s.p99)),
            f(c.map(|s| s.min)),
            f(c.map(|s| s.stddev)),
            f(c.map(|s| s.ci95_half)),
            f(r.readback.as_ref().map(|s| s.median)),
            f(r.end_to_end.as_ref().map(|s| s.median)),
            f(r.median_step_ms),
            f(r.median_mupdates_per_s),
            f(r.cold_start.as_ref().map(|c| c.setup_ms)),
            f(r.cold_start.as_ref().map(|c| c.first_run_ms)),
            r.validation.as_ref().map(|v| format!("{:.6e}", v.max_abs)).unwrap_or_default(),
            r.validation.as_ref().map(|v| format!("{:.6e}", v.rms)).unwrap_or_default(),
            r.validation.as_ref().map(|v| if v.passed { "pass" } else { "FAIL" }.to_string()).unwrap_or_default(),
            report.git_sha.clone(),
            report.adapter.as_ref().map(|a| a.name.clone()).unwrap_or_default(),
            r.error.clone().unwrap_or_default(),
        ])?;
    }
    Ok(String::from_utf8(w.into_inner().map_err(|e| e.into_error())?).unwrap())
}

pub fn write_csv(report: &BenchReport, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, to_csv(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{run_sweep, SweepSpec};
    use aether_core::{BackendKind, ExperimentConfig, WorkloadKind};

    fn report() -> BenchReport {
        let base = ExperimentConfig { workload: WorkloadKind::Heat, grid: 16, steps: 3, warmup: 1, repeats: 3, ..Default::default() };
        run_sweep(&SweepSpec { base, backends: vec![BackendKind::Scalar, BackendKind::Rayon], sizes: vec![], workgroups: vec![] }, &mut |_| {}).unwrap()
    }

    #[test]
    fn json_roundtrip_preserves_samples() {
        let r = report();
        let dir = std::env::temp_dir().join(format!("aether-export-{}", std::process::id()));
        let p = dir.join("nested/out.json");
        write_json(&r, &p).unwrap();
        let back = read_json(&p).unwrap();
        assert_eq!(back.records[0].samples, r.records[0].samples);
        assert_eq!(back.git_sha, r.git_sha);
        assert_eq!(back.records[0].config, r.records[0].config);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn csv_has_header_and_row_per_record() {
        let r = report();
        let csv = to_csv(&r).unwrap();
        let lines: Vec<_> = csv.lines().collect();
        assert_eq!(lines.len(), 1 + r.records.len());
        assert!(lines[0].starts_with("workload,backend,size"));
        assert!(lines[1].starts_with("heat,scalar,16"));
        assert!(lines[1].contains("pass"));
    }
}
