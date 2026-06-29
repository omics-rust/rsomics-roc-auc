//! ROC curve and ROC AUC.
//!
//! `roc_curve` collapses the classification curve with `drop_intermediate`
//! (scikit-learn's default), prepends the `(0, 0)` origin with a `+inf`
//! threshold, then normalises `fps`/`tps` by their totals to get `fpr`/`tpr`.
//! `roc_auc` is the trapezoidal area under that curve, reduced through NumPy's
//! pairwise sum so it is bit-identical to `sklearn.metrics.auc(fpr, tpr)`.

use crate::clf_curve::{ClfCurve, binary_clf_curve};
use crate::io::Samples;
use crate::npsum::npsum;

/// ROC curve points; `fpr[i]` / `tpr[i]` are the false / true positive rates at
/// `score >= thresholds[i]`. The first threshold is `+inf` (the `(0,0)` origin).
pub struct RocCurve {
    pub fpr: Vec<f64>,
    pub tpr: Vec<f64>,
    pub thresholds: Vec<f64>,
}

/// Second difference `x[i+2] - 2*x[i+1] + x[i]`, as `np.diff(x, 2)`.
fn diff2_is_zero(x: &[f64], i: usize) -> bool {
    x[i + 2] - 2.0 * x[i + 1] + x[i] == 0.0
}

/// Apply scikit-learn's `drop_intermediate` collinear-point removal to the raw
/// `fps`/`tps`/`thresholds`. Keeps the endpoints and every point that is a
/// corner in either fps or tps.
fn drop_intermediate(c: &ClfCurve) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = c.fps.len();
    if n <= 2 {
        return (c.fps.clone(), c.tps.clone(), c.thresholds.clone());
    }
    let mut keep = Vec::with_capacity(n);
    keep.push(0);
    for i in 1..n - 1 {
        if !diff2_is_zero(&c.fps, i - 1) || !diff2_is_zero(&c.tps, i - 1) {
            keep.push(i);
        }
    }
    keep.push(n - 1);
    let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
    (pick(&c.fps), pick(&c.tps), pick(&c.thresholds))
}

#[must_use]
pub fn roc_curve(s: &Samples, drop_inter: bool) -> RocCurve {
    let c = binary_clf_curve(s);
    let (mut fps, mut tps, mut thr) = if drop_inter {
        drop_intermediate(&c)
    } else {
        (c.fps, c.tps, c.thresholds)
    };

    // Prepend the (0,0) origin and a +inf threshold.
    fps.insert(0, 0.0);
    tps.insert(0, 0.0);
    thr.insert(0, f64::INFINITY);

    let fps_total = *fps.last().expect("at least the origin point");
    let tps_total = *tps.last().expect("at least the origin point");
    let fpr = if fps_total <= 0.0 {
        vec![f64::NAN; fps.len()]
    } else {
        fps.iter().map(|&v| v / fps_total).collect()
    };
    let tpr = if tps_total <= 0.0 {
        vec![f64::NAN; tps.len()]
    } else {
        tps.iter().map(|&v| v / tps_total).collect()
    };

    RocCurve {
        fpr,
        tpr,
        thresholds: thr,
    }
}

/// Trapezoidal area under `(x, y)`, matching `numpy.trapezoid` followed by
/// scikit-learn's monotonic-direction sign: `(diff(x) * (y[1:]+y[:-1]) / 2).sum()`.
fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    let m = x.len() - 1;
    let mut terms = Vec::with_capacity(m);
    for i in 0..m {
        terms.push((x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.0);
    }
    let area = npsum(&terms);
    // fpr is non-decreasing, so the direction is +1; an all-non-increasing x
    // would flip it, matching sklearn.metrics.auc.
    let mut any_neg = false;
    let mut all_nonpos = true;
    for i in 0..m {
        let dx = x[i + 1] - x[i];
        if dx < 0.0 {
            any_neg = true;
        }
        if dx > 0.0 {
            all_nonpos = false;
        }
    }
    if any_neg && all_nonpos { -area } else { area }
}

/// ROC AUC for a binary problem; `f64::NAN` when only one class is present
/// (scikit-learn emits a warning and returns NaN there).
#[must_use]
pub fn roc_auc(s: &Samples) -> f64 {
    let pos = s.y_true.iter().filter(|&&t| t == 1).count();
    if pos == 0 || pos == s.len() {
        return f64::NAN;
    }
    let curve = roc_curve(s, true);
    trapezoid(&curve.fpr, &curve.tpr)
}

#[cfg(test)]
mod tests {
    use super::{roc_auc, roc_curve};
    use crate::io::Samples;

    #[test]
    fn perfect_separation_auc_is_one() {
        let s = Samples {
            y_true: vec![0, 0, 1, 1],
            y_score: vec![0.1, 0.2, 0.8, 0.9],
        };
        assert_eq!(roc_auc(&s), 1.0);
    }

    #[test]
    fn single_class_is_nan() {
        let s = Samples {
            y_true: vec![1, 1, 1],
            y_score: vec![0.1, 0.5, 0.9],
        };
        assert!(roc_auc(&s).is_nan());
    }

    #[test]
    fn curve_starts_at_origin_with_inf_threshold() {
        let s = Samples {
            y_true: vec![0, 0, 1, 1],
            y_score: vec![0.1, 0.4, 0.35, 0.8],
        };
        let c = roc_curve(&s, true);
        assert_eq!(c.fpr[0], 0.0);
        assert_eq!(c.tpr[0], 0.0);
        assert!(c.thresholds[0].is_infinite());
    }
}
