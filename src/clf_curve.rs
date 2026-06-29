//! The binary classification curve scikit-learn builds before any ranking
//! metric: `confusion_matrix_at_thresholds` (the function formerly called
//! `_binary_clf_curve`).
//!
//! Samples are sorted by descending score with a *stable* sort; tied scores
//! collapse to one threshold at the last index of each run. At each distinct
//! threshold we report the cumulative count of true positives (`tps`) and false
//! positives (`fps`) for predictions with `score >= threshold`. Without sample
//! weights these counts are exact integers — `tps` is a running count of the
//! positive labels and `fps = 1 + index - tps` — so the curve carries no
//! rounding error into the downstream AUC / AP reductions.

use crate::io::Samples;

/// One distinct-threshold point of the classification curve.
pub struct ClfCurve {
    /// True positives at `score >= threshold`, monotonically increasing.
    pub tps: Vec<f64>,
    /// False positives at `score >= threshold`, monotonically increasing.
    pub fps: Vec<f64>,
    /// Decreasing score thresholds, one per distinct score value.
    pub thresholds: Vec<f64>,
}

/// Build the descending-score classification curve, matching scikit-learn's
/// stable argsort, tie collapse, and threshold endpoint exactly.
#[must_use]
pub fn binary_clf_curve(s: &Samples) -> ClfCurve {
    let n = s.len();

    // Stable argsort by descending score: ties keep input order, like
    // `np.argsort(..., stable=True)` then read in reverse-score order. A stable
    // ascending sort on the negated index-paired key reproduces numpy's
    // stable-descending result.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        s.y_score[j]
            .partial_cmp(&s.y_score[i])
            .expect("scores are finite")
            .then(i.cmp(&j))
    });

    let mut cum_tps = Vec::with_capacity(n);
    let mut running = 0.0f64;
    for &idx in &order {
        running += f64::from(s.y_true[idx]);
        cum_tps.push(running);
    }

    // distinct_value_indices = nonzero(diff(sorted_score)); always append n-1.
    let mut tps = Vec::new();
    let mut fps = Vec::new();
    let mut thresholds = Vec::new();
    for k in 0..n {
        let last = k + 1 == n;
        let boundary = last || s.y_score[order[k]] != s.y_score[order[k + 1]];
        if boundary {
            let t = cum_tps[k];
            tps.push(t);
            #[allow(clippy::cast_precision_loss)]
            let f = 1.0 + k as f64 - t;
            fps.push(f);
            thresholds.push(s.y_score[order[k]]);
        }
    }

    ClfCurve {
        tps,
        fps,
        thresholds,
    }
}

#[cfg(test)]
mod tests {
    use super::binary_clf_curve;
    use crate::io::Samples;

    /// The docstring example from `confusion_matrix_at_thresholds`.
    #[test]
    fn matches_sklearn_docstring_example() {
        let s = Samples {
            y_true: vec![0, 0, 1, 1],
            y_score: vec![0.1, 0.4, 0.35, 0.8],
        };
        let c = binary_clf_curve(&s);
        assert_eq!(c.fps, vec![0.0, 1.0, 1.0, 2.0]);
        assert_eq!(c.tps, vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(c.thresholds, vec![0.8, 0.4, 0.35, 0.1]);
    }

    #[test]
    fn ties_collapse_to_last_index() {
        let s = Samples {
            y_true: vec![1, 0, 1, 0],
            y_score: vec![0.5, 0.5, 0.2, 0.2],
        };
        let c = binary_clf_curve(&s);
        assert_eq!(c.thresholds, vec![0.5, 0.2]);
        assert_eq!(c.tps, vec![1.0, 2.0]);
        assert_eq!(c.fps, vec![1.0, 2.0]);
    }
}
