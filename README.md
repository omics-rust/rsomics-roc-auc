# rsomics-roc-auc

Binary ranking and threshold classification metrics — ROC AUC, average
precision, and the underlying ROC / precision-recall curve points. A
value-exact, faster port of `scikit-learn`'s `roc_auc_score`,
`average_precision_score`, `roc_curve`, and `precision_recall_curve`.

```sh
cargo install rsomics-roc-auc
```

## Usage

Input is two whitespace/tab-separated columns `y_true y_score`, one sample per
line, from a file argument or stdin (`-`): `y_true` is a binary 0/1 label,
`y_score` a continuous decision value (a probability or a margin).

```sh
rsomics-roc-auc scores.tsv                              # ROC AUC (default)
rsomics-roc-auc scores.tsv --metric average-precision   # average precision
rsomics-roc-auc scores.tsv --metric roc-curve           # ROC curve points
rsomics-roc-auc scores.tsv --metric pr-curve            # PR curve points
rsomics-roc-auc scores.tsv --json
```

`--metric roc-auc` and `--metric average-precision` print a single number.
`--metric roc-curve` prints `fpr<TAB>tpr<TAB>threshold` per line, with the first
threshold `inf` (the `(0,0)` origin) — matching scikit-learn's default
`drop_intermediate=True`, which removes collinear interior points without
changing the curve shape or AUC. `--metric pr-curve` prints
`precision<TAB>recall<TAB>threshold` per line with recall decreasing and a final
`(precision=1, recall=0)` endpoint that carries no threshold — matching
scikit-learn's `precision_recall_curve` default of keeping every point.

Single-class inputs follow scikit-learn: ROC AUC is `nan` (undefined),
average precision is `1` for all-positive and `0` for all-negative.

## Accuracy

Verified **bit-identical (0 ULP)** against `scikit-learn` 1.9.0 (numpy 2.4.6) on
datasets spanning n = 10 … 1,000,000, balanced and imbalanced (5% positive),
with and without score ties:

- **ROC AUC** and **average precision** match scikit-learn's `float64` result
  to the last bit. Both are exact-integer count ratios fed through a trapezoid
  (`auc`) or a step-function sum (`average_precision_score`); there is no
  transcendental step, so the only thing that can drift is the order of the
  final `ndarray.sum()`. That sum follows NumPy's pairwise reduction (128-block
  recursion with an 8-accumulator base case), so the rounding matches `np.sum`
  element for element rather than diverging through a naive fold — which matters
  on million-element curves where a sequential sum loses ULPs.
- **ROC curve** and **PR curve** points (`fpr`/`tpr`/`thresholds`,
  `precision`/`recall`/`thresholds`) are bit-identical too, including the stable
  descending-score sort, the tie collapse (equal scores share one threshold at
  the last index of the run), the `(0,0)` ROC origin with its `+inf` threshold,
  the `drop_intermediate` second-difference point removal, and the reversed PR
  endpoint — checked against curves with tens of thousands of points.

The compatibility test compares committed IEEE-754 hex goldens and needs no
Python at test time.

## Performance

Single-threaded, versus scikit-learn on the same machine (macOS, Apple M2),
n = 1,000,000 with score ties and `OPENBLAS/OMP/MKL/VECLIB/NUMEXPR_NUM_THREADS=1`:

- **ROC AUC** — ours ≈ 97 ms end-to-end (`-t1`, including the full TSV parse)
  versus scikit-learn ≈ 1.71 s end-to-end → **~17.6× faster**. Compute-only
  (parse excluded both sides), ours ≈ 63 ms versus scikit-learn ≈ 139 ms →
  **~2.2× faster**.
- **Average precision** — ours ≈ 98 ms end-to-end versus scikit-learn ≈ 1.80 s →
  **~18.4× faster**. Compute-only, ours ≈ 68 ms versus scikit-learn ≈ 99 ms →
  **~1.4× faster**.

The win holds on both the end-to-end and compute-only axes, single-threaded. The
work is dominated by the descending-score sort; ours uses a pattern-defeating
quicksort and a fast float parser where scikit-learn pays `np.argsort` plus the
Python call and validation overhead on each metric.

## Origin

This crate is an independent Rust reimplementation of `scikit-learn`'s
`sklearn.metrics` ranking metrics, based on the scikit-learn source (1.9.0,
BSD-3-Clause) and black-box differential testing against it. The classification
curve follows `confusion_matrix_at_thresholds` (the function formerly named
`_binary_clf_curve`): a stable descending-score sort, tie collapse at distinct
score boundaries, and cumulative true/false positive counts. `roc_curve`
reproduces the `drop_intermediate` second-difference removal, the `(0,0)` origin
with a `+inf` threshold, and the `fps`/`tps` normalisation; `roc_auc_score` is
`numpy.trapezoid` over `(fpr, tpr)` with scikit-learn's monotonic-direction
sign. `precision_recall_curve` reproduces the precision/recall construction, the
reversal, and the appended endpoint; `average_precision_score` is the step
integral `−Σ diff(recall)·precision[:-1]` clipped at zero. All final reductions
go through NumPy's pairwise summation so the result is bit-identical.

No code was copied; the implementation reads the BSD-licensed source as a spec.
Golden expectations in `tests/golden/` were generated once from scikit-learn and
are checked into the repo as IEEE-754 hex.

License: MIT OR Apache-2.0.
Upstream credit: scikit-learn (https://scikit-learn.org, BSD-3-Clause); NumPy (https://numpy.org, BSD-3-Clause).
