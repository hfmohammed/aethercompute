//! Dependency-free SVG charts from benchmark reports.

use std::collections::BTreeMap;
use std::fmt::Write;

use aether_core::BackendKind;

use crate::record::BenchReport;

const W: f64 = 860.0;
const H: f64 = 460.0;
const COLORS: [&str; 5] = ["#6b7280", "#2563eb", "#ea580c", "#16a34a", "#9333ea"];

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Grouped bars: x = problem size, one bar per series, log10 y axis.
fn grouped_log_bars(title: &str, subtitle: &str, y_label: &str, groups: &[(String, Vec<(String, f64)>)]) -> String {
    let series: Vec<String> = {
        let mut s: Vec<String> = vec![];
        for (_, bars) in groups {
            for (name, _) in bars {
                if !s.contains(name) {
                    s.push(name.clone());
                }
            }
        }
        s
    };
    let vals: Vec<f64> = groups.iter().flat_map(|(_, b)| b.iter().map(|x| x.1)).filter(|v| *v > 0.0).collect();
    let (lo, hi) = vals.iter().fold((f64::MAX, f64::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
    let (lo, hi) = if vals.is_empty() { (0.1, 10.0) } else { (10f64.powf(lo.log10().floor()), 10f64.powf(hi.log10().ceil().max(lo.log10().floor() + 1.0))) };
    let (ml, mr, mt, mb) = (78.0, 20.0, 70.0, 70.0);
    let (pw, ph) = (W - ml - mr, H - mt - mb);
    let y = |v: f64| mt + ph * (1.0 - (v.max(lo).log10() - lo.log10()) / (hi.log10() - lo.log10()));

    let mut s = String::new();
    let _ = write!(s, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="-apple-system,Helvetica,Arial,sans-serif">"#);
    let _ = write!(s, r##"<rect width="{W}" height="{H}" fill="#ffffff"/>"##);
    let _ = write!(s, r##"<text x="{ml}" y="28" font-size="18" font-weight="600" fill="#111827">{}</text>"##, esc(title));
    let _ = write!(s, r##"<text x="{ml}" y="48" font-size="12" fill="#6b7280">{}</text>"##, esc(subtitle));
    let mut e = lo.log10().round() as i32;
    while e <= hi.log10().round() as i32 {
        let v = 10f64.powi(e);
        let yy = y(v);
        let _ = write!(s, r##"<line x1="{ml}" x2="{}" y1="{yy}" y2="{yy}" stroke="#e5e7eb"/>"##, W - mr);
        let _ = write!(s, r##"<text x="{}" y="{}" font-size="11" text-anchor="end" fill="#6b7280">{}</text>"##, ml - 8.0, yy + 4.0, fmt_tick(v));
        e += 1;
    }
    let _ = write!(s, r##"<text x="16" y="{}" font-size="12" fill="#374151" transform="rotate(-90 16 {})" text-anchor="middle">{}</text>"##, mt + ph / 2.0, mt + ph / 2.0, esc(y_label));
    let gw = pw / groups.len().max(1) as f64;
    let bw = (gw * 0.8 / series.len().max(1) as f64).min(46.0);
    for (gi, (label, bars)) in groups.iter().enumerate() {
        let gx = ml + gw * gi as f64 + gw / 2.0;
        let x0 = gx - bw * series.len() as f64 / 2.0;
        for (name, v) in bars {
            let si = series.iter().position(|n| n == name).unwrap();
            let top = y(*v);
            let _ = write!(s, r#"<rect x="{}" y="{top}" width="{}" height="{}" rx="2" fill="{}"/>"#, x0 + bw * si as f64, bw - 2.0, (mt + ph - top).max(0.0), COLORS[si % COLORS.len()]);
            let _ = write!(s, r##"<text x="{}" y="{}" font-size="10" text-anchor="middle" fill="#111827">{}</text>"##, x0 + bw * si as f64 + (bw - 2.0) / 2.0, top - 4.0, fmt_val(*v));
        }
        let _ = write!(s, r##"<text x="{gx}" y="{}" font-size="12" text-anchor="middle" fill="#374151">{}</text>"##, mt + ph + 20.0, esc(label));
    }
    for (i, name) in series.iter().enumerate() {
        let lx = ml + 130.0 * i as f64;
        let _ = write!(s, r#"<rect x="{lx}" y="{}" width="12" height="12" rx="2" fill="{}"/>"#, H - 28.0, COLORS[i % COLORS.len()]);
        let _ = write!(s, r##"<text x="{}" y="{}" font-size="12" fill="#374151">{}</text>"##, lx + 18.0, H - 18.0, esc(name));
    }
    s.push_str("</svg>\n");
    s
}

fn fmt_tick(v: f64) -> String {
    if v >= 1000.0 { format!("{:.0}k", v / 1000.0) } else if v >= 1.0 { format!("{v:.0}") } else { format!("{v}") }
}

fn fmt_val(v: f64) -> String {
    if v >= 100.0 { format!("{v:.0}") } else if v >= 10.0 { format!("{v:.1}") } else { format!("{v:.2}") }
}

fn hardware_line(r: &BenchReport) -> String {
    format!(
        "{} · {} GB · macOS {} · {} · {} · commit {}",
        r.machine.chip,
        r.machine.ram_bytes / (1 << 30),
        r.machine.os_version,
        r.adapter.as_ref().map(|a| format!("{} ({})", a.name, a.backend)).unwrap_or_else(|| "no GPU".into()),
        r.machine.build_profile,
        &r.git_sha[..r.git_sha.len().min(7)],
    )
}

/// Throughput (millions of updates/s, log scale) per size and backend.
/// Only the best GPU workgroup size per size is shown.
pub fn throughput_chart(report: &BenchReport) -> String {
    let mut groups: BTreeMap<u32, BTreeMap<String, f64>> = BTreeMap::new();
    let mut workload = String::new();
    for r in report.records.iter().filter(|r| r.validated_ok()) {
        workload = r.config.workload.as_str().to_string();
        let size = if r.config.workload.is_particles() { r.config.count } else { r.config.grid };
        let name = r.config.backend.as_str().to_string();
        let v = r.median_mupdates_per_s.unwrap_or(0.0);
        let e = groups.entry(size).or_default().entry(name).or_insert(0.0);
        *e = e.max(v);
    }
    let order = |n: &str| [BackendKind::Scalar, BackendKind::Rayon, BackendKind::Gpu].iter().position(|b| b.as_str() == n).unwrap_or(9);
    let groups: Vec<_> = groups
        .into_iter()
        .map(|(size, m)| {
            let mut bars: Vec<_> = m.into_iter().collect();
            bars.sort_by_key(|(n, _)| order(n));
            let label = if workload == "heat" { format!("{size}²") } else { format!("{size}") };
            (label, bars)
        })
        .collect();
    let unit = if workload == "heat" { "M cell-updates/s" } else { "M particle-updates/s" };
    grouped_log_bars(
        &format!("{workload}: throughput by backend (median, higher is better)"),
        &hardware_line(report),
        unit,
        &groups,
    )
}

/// GPU speed-up over the scalar and Rayon CPU baselines (compute only and
/// end-to-end including readback), best workgroup size per size.
pub fn speedup_chart(report: &BenchReport) -> String {
    let mut groups: BTreeMap<u32, BTreeMap<&'static str, f64>> = BTreeMap::new();
    let mut workload = String::new();
    for s in &report.speedups {
        workload = s.workload.clone();
        let g = groups.entry(s.size).or_default();
        for (name, v) in [
            ("vs scalar (compute)", s.vs_scalar_compute),
            ("vs rayon (compute)", s.vs_rayon_compute),
            ("vs rayon (end-to-end)", s.vs_rayon_end_to_end),
        ] {
            if let Some(v) = v {
                let e = g.entry(name).or_insert(0.0);
                *e = e.max(v);
            }
        }
    }
    let groups: Vec<_> = groups
        .into_iter()
        .map(|(size, m)| {
            let mut bars: Vec<(String, f64)> = m.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
            bars.sort_by_key(|(n, _)| ["vs scalar (compute)", "vs rayon (compute)", "vs rayon (end-to-end)"].iter().position(|x| x == n));
            let label = if workload == "heat" { format!("{size}²") } else { format!("{size}") };
            (label, bars)
        })
        .collect();
    grouped_log_bars(
        &format!("{workload}: GPU speed-up (median time ratio, >1 = GPU faster)"),
        &hardware_line(report),
        "speed-up (x)",
        &groups,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{run_sweep, SweepSpec};
    use aether_core::{ExperimentConfig, WorkloadKind};

    fn report() -> BenchReport {
        let base = ExperimentConfig { workload: WorkloadKind::Heat, grid: 16, steps: 3, warmup: 1, repeats: 3, ..Default::default() };
        run_sweep(&SweepSpec { base, backends: BackendKind::ALL.to_vec(), sizes: vec![16, 24], workgroups: vec![] }, &mut |_| {}).unwrap()
    }

    #[test]
    fn charts_are_wellformed_svg() {
        let r = report();
        for svg in [throughput_chart(&r), speedup_chart(&r)] {
            assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
            assert!(svg.contains("<rect"));
        }
    }

    #[test]
    fn titles_are_escaped() {
        assert_eq!(esc("a<b&c"), "a&lt;b&amp;c");
    }

    #[test]
    fn empty_report_does_not_panic() {
        let mut r = report();
        r.records.clear();
        r.speedups.clear();
        let _ = throughput_chart(&r);
        let _ = speedup_chart(&r);
    }
}
