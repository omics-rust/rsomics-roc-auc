//! NumPy's pairwise summation over a C-contiguous buffer.
//!
//! `np.sum` does not fold left-to-right: it splits blocks larger than 128
//! recursively (the split point rounded down to a multiple of 8) and reduces
//! each ≤128 block with an 8-accumulator unrolled loop. This rounds differently
//! from a naive sequential sum. The trapezoid area (`auc`) and the
//! average-precision step integral both end in an `ndarray.sum()` over arrays
//! that can be a million elements long, so matching this reduction order is
//! what makes the result bit-identical to scikit-learn rather than drifting by
//! ULPs that grow with the input size.

/// Sum a slice exactly as `numpy.sum` reduces a flat C-order array.
#[must_use]
pub fn npsum(a: &[f64]) -> f64 {
    pairwise(a)
}

fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0, |s, &x| s + x);
    }
    if n <= 128 {
        let (mut r0, mut r1, mut r2, mut r3) = (a[0], a[1], a[2], a[3]);
        let (mut r4, mut r5, mut r6, mut r7) = (a[4], a[5], a[6], a[7]);
        let mut i = 8;
        while i + 8 <= n {
            r0 += a[i];
            r1 += a[i + 1];
            r2 += a[i + 2];
            r3 += a[i + 3];
            r4 += a[i + 4];
            r5 += a[i + 5];
            r6 += a[i + 6];
            r7 += a[i + 7];
            i += 8;
        }
        let mut res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise(&a[..n2]) + pairwise(&a[n2..])
}
