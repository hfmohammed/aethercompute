//! Numerical comparison of a candidate result against a reference.

use serde::{Deserialize, Serialize};

use crate::config::WorkloadKind;

/// Pass criteria. An element passes if `|a-b| <= abs + rel*|b|`; the run also
/// needs RMS error below `rms` and no NaN/Inf.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tolerance {
    pub abs: f64,
    pub rel: f64,
    pub rms: f64,
}

impl Tolerance {
    /// Exact equality (scalar vs Rayon run the identical per-element code).
    pub const EXACT: Tolerance = Tolerance {
        abs: 0.0,
        rel: 0.0,
        rms: 0.0,
    };

    /// Default CPU-vs-GPU tolerance, scaled by the number of steps because
    /// rounding differences accumulate (documented in docs/numerics.md).
    pub fn gpu_vs_cpu(workload: WorkloadKind, steps: u32) -> Tolerance {
        let s = (steps.max(1) as f64).sqrt();
        match workload {
            WorkloadKind::ParticlesLocal => Tolerance {
                abs: 1e-4 * s.max(1.0),
                rel: 1e-4,
                rms: 1e-5 * s.max(1.0),
            },
            WorkloadKind::NbodyAllpairs => Tolerance {
                abs: 5e-4 * s.max(1.0),
                rel: 5e-4,
                rms: 1e-4 * s.max(1.0),
            },
            WorkloadKind::Heat => Tolerance {
                abs: 1e-5 * s.max(1.0),
                rel: 1e-4,
                rms: 1e-6 * s.max(1.0),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorReport {
    pub n: usize,
    pub max_abs: f64,
    pub rms: f64,
    /// Max relative error with the denominator floored at `REL_FLOOR`.
    pub max_rel: f64,
    pub nonfinite: usize,
    pub tolerance: Tolerance,
    pub passed: bool,
}

/// Denominator floor so values near zero do not blow up relative error.
pub const REL_FLOOR: f64 = 1e-3;

/// Compare flat `f32` slices.
pub fn compare(candidate: &[f32], reference: &[f32], tol: Tolerance) -> ErrorReport {
    assert_eq!(candidate.len(), reference.len(), "length mismatch");
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut sq = 0.0f64;
    let mut nonfinite = 0usize;
    let mut violations = 0usize;
    for (&a, &b) in candidate.iter().zip(reference) {
        if !a.is_finite() || !b.is_finite() {
            nonfinite += 1;
            continue;
        }
        let (a, b) = (a as f64, b as f64);
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / b.abs().max(REL_FLOOR));
        sq += d * d;
        if d > tol.abs + tol.rel * b.abs() {
            violations += 1;
        }
    }
    let n = candidate.len();
    let rms = if n == 0 { 0.0 } else { (sq / n as f64).sqrt() };
    let passed = nonfinite == 0 && violations == 0 && rms <= tol.rms;
    ErrorReport {
        n,
        max_abs,
        rms,
        max_rel,
        nonfinite,
        tolerance: tol,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Tolerance = Tolerance {
        abs: 1e-3,
        rel: 1e-3,
        rms: 1e-3,
    };

    #[test]
    fn identical_passes_even_exact() {
        let a = [1.0, 2.0, 3.0];
        let r = compare(&a, &a, Tolerance::EXACT);
        assert!(r.passed);
        assert_eq!(r.max_abs, 0.0);
    }

    #[test]
    fn small_error_passes() {
        let r = compare(&[1.0005, 2.0], &[1.0, 2.0], TOL);
        assert!(r.passed);
        assert!((r.max_abs - 0.0005).abs() < 1e-6);
    }

    #[test]
    fn large_error_fails() {
        let r = compare(&[1.1, 2.0], &[1.0, 2.0], TOL);
        assert!(!r.passed);
    }

    #[test]
    fn nan_fails() {
        let r = compare(&[f32::NAN, 2.0], &[1.0, 2.0], TOL);
        assert!(!r.passed);
        assert_eq!(r.nonfinite, 1);
        let r = compare(&[f32::INFINITY], &[f32::INFINITY], TOL);
        assert!(!r.passed);
    }

    #[test]
    fn near_zero_relative_uses_floor() {
        let r = compare(&[1e-9], &[0.0], TOL);
        assert!(r.passed);
        assert!(r.max_rel < 1e-5);
    }

    #[test]
    fn exact_rejects_any_difference() {
        let r = compare(&[1.0 + f32::EPSILON], &[1.0], Tolerance::EXACT);
        assert!(!r.passed);
    }

    #[test]
    fn rms_is_computed() {
        let r = compare(
            &[1.0, 1.0],
            &[0.0, 0.0],
            Tolerance {
                abs: 2.0,
                rel: 0.0,
                rms: 2.0,
            },
        );
        assert!((r.rms - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_is_trivially_ok() {
        assert!(compare(&[], &[], TOL).passed);
    }

    #[test]
    fn tolerances_grow_with_steps() {
        let a = Tolerance::gpu_vs_cpu(WorkloadKind::Heat, 1);
        let b = Tolerance::gpu_vs_cpu(WorkloadKind::Heat, 400);
        assert!(b.abs > a.abs);
    }
}
