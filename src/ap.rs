//! Precision-recall curve and average precision.
//!
//! `precision_recall_curve` turns the classification curve into precision and
//! recall per threshold, reverses them so recall decreases, and appends the
//! `(precision=1, recall=0)` endpoint. Average precision is the step-function
//! integral `Σ (R_n − R_{n−1}) · P_n` — scikit-learn computes it as
//! `−Σ diff(recall) · precision[:-1]`, clipped at zero, reduced through NumPy's
//! pairwise sum so it is bit-identical to `average_precision_score`.

use crate::clf_curve::{ClfCurve, binary_clf_curve};
use crate::io::Samples;
use crate::npsum::npsum;

/// Precision-recall curve points; precision/recall align with the threshold at
/// the same index for all but the final appended `(1, 0)` endpoint, which has no
/// threshold. Recall is non-increasing.
pub struct PrCurve {
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub thresholds: Vec<f64>,
}

/// Drop the points where `tps` does not change from a neighbour (the vertical
/// segments of equal recall), matching `precision_recall_curve(drop_intermediate=True)`.
fn drop_intermediate(c: &ClfCurve) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = c.tps.len();
    if n <= 2 {
        return (c.fps.clone(), c.tps.clone(), c.thresholds.clone());
    }
    // optimal_idxs = where(concat([True], logical_or(diff(tps[:-1]), diff(tps[1:])), [True]))
    let mut keep = Vec::with_capacity(n);
    keep.push(0);
    for i in 1..n - 1 {
        let d_prev = c.tps[i] - c.tps[i - 1];
        let d_next = c.tps[i + 1] - c.tps[i];
        if d_prev != 0.0 || d_next != 0.0 {
            keep.push(i);
        }
    }
    keep.push(n - 1);
    let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
    (pick(&c.fps), pick(&c.tps), pick(&c.thresholds))
}

#[must_use]
pub fn pr_curve(s: &Samples, drop_inter: bool) -> PrCurve {
    let c = binary_clf_curve(s);
    let (fps, tps, thr) = if drop_inter {
        drop_intermediate(&c)
    } else {
        (c.fps, c.tps, c.thresholds)
    };

    let n = tps.len();
    let tps_total = tps[n - 1];

    let mut precision = Vec::with_capacity(n + 1);
    let mut recall = Vec::with_capacity(n + 1);
    // Build forward (increasing threshold index), then flip and append endpoint.
    let mut fwd_p = Vec::with_capacity(n);
    let mut fwd_r = Vec::with_capacity(n);
    for i in 0..n {
        let ps = tps[i] + fps[i];
        fwd_p.push(if ps != 0.0 { tps[i] / ps } else { 0.0 });
        fwd_r.push(if tps_total == 0.0 {
            1.0
        } else {
            tps[i] / tps_total
        });
    }
    for i in (0..n).rev() {
        precision.push(fwd_p[i]);
        recall.push(fwd_r[i]);
    }
    precision.push(1.0);
    recall.push(0.0);

    let thresholds = thr.iter().rev().copied().collect();

    PrCurve {
        precision,
        recall,
        thresholds,
    }
}

/// Average precision for a binary problem. Single-class inputs are handled by
/// the curve itself, matching scikit-learn: all-positive integrates to `1.0`
/// (precision is everywhere 1), all-negative to `0.0` (recall is constant so
/// every step contributes nothing).
#[must_use]
pub fn average_precision(s: &Samples) -> f64 {
    let curve = pr_curve(s, false);
    // -sum(diff(recall) * precision[:-1]), clipped to >= 0.
    let m = curve.recall.len() - 1;
    let mut terms = Vec::with_capacity(m);
    for i in 0..m {
        terms.push((curve.recall[i + 1] - curve.recall[i]) * curve.precision[i]);
    }
    let ap = -npsum(&terms);
    ap.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::{average_precision, pr_curve};
    use crate::io::Samples;

    #[test]
    fn perfect_separation_ap_is_one() {
        let s = Samples {
            y_true: vec![0, 0, 1, 1],
            y_score: vec![0.1, 0.2, 0.8, 0.9],
        };
        assert_eq!(average_precision(&s), 1.0);
    }

    #[test]
    fn curve_ends_at_recall_zero_precision_one() {
        let s = Samples {
            y_true: vec![0, 0, 1, 1],
            y_score: vec![0.1, 0.4, 0.35, 0.8],
        };
        let c = pr_curve(&s, false);
        assert_eq!(*c.precision.last().unwrap(), 1.0);
        assert_eq!(*c.recall.last().unwrap(), 0.0);
        assert_eq!(c.thresholds.len(), c.precision.len() - 1);
    }

    #[test]
    fn all_negative_ap_is_zero() {
        let s = Samples {
            y_true: vec![0, 0, 0],
            y_score: vec![0.1, 0.5, 0.9],
        };
        assert_eq!(average_precision(&s), 0.0);
    }

    #[test]
    fn all_positive_ap_is_one() {
        let s = Samples {
            y_true: vec![1, 1, 1],
            y_score: vec![0.1, 0.5, 0.9],
        };
        assert_eq!(average_precision(&s), 1.0);
    }
}
