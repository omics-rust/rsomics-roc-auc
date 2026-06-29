//! Binary ranking / threshold classification metrics — value-exact, faster
//! ports of `sklearn.metrics`: `roc_auc_score`, `average_precision_score`, and
//! the `roc_curve` / `precision_recall_curve` points.
//!
//! Input is two columns `y_true y_score` (binary 0/1 labels, continuous
//! scores). Everything is built on scikit-learn's `confusion_matrix_at_thresholds`
//! (the descending-score, tie-collapsing classification curve), so the sort,
//! tie aggregation, curve endpoints, `drop_intermediate` removal, trapezoid
//! direction, and the AP step-sum all match scikit-learn exactly. The sum
//! reductions follow NumPy's pairwise scheme to stay bit-identical on large
//! inputs.

mod ap;
mod clf_curve;
mod io;
mod npsum;
mod roc;

pub use ap::{PrCurve, average_precision, pr_curve};
pub use clf_curve::{ClfCurve, binary_clf_curve};
pub use io::{Samples, parse_samples, read_samples};
pub use roc::{RocCurve, roc_auc, roc_curve};
