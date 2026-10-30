use serde::{Deserialize, Serialize};

/// Distribution summary of timing samples (any unit).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub n: usize,
    pub min: f64,
    pub median: f64,
    pub mean: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub stddev: f64,
    /// Half-width of the 95% confidence interval of the mean (Student t).
    pub ci95_half: f64,
}

/// Linear-interpolated percentile of an already sorted slice, `q` in [0, 1].
pub fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty());
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Two-sided 95% Student-t critical value for `df` degrees of freedom.
pub fn t_crit_95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    match df {
        0 => f64::NAN,
        1..=30 => T[df - 1],
        31..=60 => 2.0,
        _ => 1.96,
    }
}

impl Summary {
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut s = samples.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        let mean = s.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let stddev = var.sqrt();
        let ci95_half = if n > 1 {
            t_crit_95(n - 1) * stddev / (n as f64).sqrt()
        } else {
            0.0
        };
        Some(Self {
            n,
            min: s[0],
            median: percentile_sorted(&s, 0.5),
            mean,
            p95: percentile_sorted(&s, 0.95),
            p99: percentile_sorted(&s, 0.99),
            max: s[n - 1],
            stddev,
            ci95_half,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_summary() {
        let s = Summary::from_samples(&[5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
        assert_eq!(s.n, 5);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        assert_eq!(s.median, 3.0);
        assert_eq!(s.mean, 3.0);
        assert!((s.stddev - 2.5f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn even_median_interpolates() {
        let s = Summary::from_samples(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(s.median, 2.5);
    }

    #[test]
    fn percentiles() {
        let v: Vec<f64> = (0..=100).map(|x| x as f64).collect();
        let s = Summary::from_samples(&v).unwrap();
        assert!((s.p95 - 95.0).abs() < 1e-9);
        assert!((s.p99 - 99.0).abs() < 1e-9);
    }

    #[test]
    fn single_sample_has_zero_spread() {
        let s = Summary::from_samples(&[7.0]).unwrap();
        assert_eq!((s.stddev, s.ci95_half, s.median), (0.0, 0.0, 7.0));
    }

    #[test]
    fn rejects_empty_and_nonfinite() {
        assert!(Summary::from_samples(&[]).is_none());
        assert!(Summary::from_samples(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn ci_shrinks_with_more_samples() {
        let a = Summary::from_samples(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let v: Vec<f64> = (0..50).map(|i| 1.0 + (i % 5) as f64).collect();
        let b = Summary::from_samples(&v).unwrap();
        assert!(b.ci95_half < a.ci95_half);
    }

    #[test]
    fn t_table() {
        assert_eq!(t_crit_95(19), 2.093);
        assert_eq!(t_crit_95(1000), 1.96);
    }
}
