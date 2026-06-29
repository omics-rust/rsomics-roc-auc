use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_roc_auc::{Samples, average_precision, roc_auc};
use std::hint::black_box;

fn xorshift() -> impl FnMut() -> u64 {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

/// `n` samples, roughly balanced, with score ties (scores quantised to 1000
/// distinct values so the tie path is exercised at scale).
fn dataset(n: usize) -> Samples {
    let mut next = xorshift();
    let mut y_true = Vec::with_capacity(n);
    let mut y_score = Vec::with_capacity(n);
    for _ in 0..n {
        let r = next();
        y_true.push((r & 1) as u8);
        #[allow(clippy::cast_precision_loss)]
        let q = (next() % 1000) as f64 / 1000.0;
        y_score.push(q);
    }
    Samples { y_true, y_score }
}

fn bench_roc_auc(c: &mut Criterion) {
    let s = dataset(1_000_000);
    c.bench_function("roc_auc_1m", |b| {
        b.iter(|| black_box(roc_auc(black_box(&s))));
    });
}

fn bench_ap(c: &mut Criterion) {
    let s = dataset(1_000_000);
    c.bench_function("average_precision_1m", |b| {
        b.iter(|| black_box(average_precision(black_box(&s))));
    });
}

criterion_group!(benches, bench_roc_auc, bench_ap);
criterion_main!(benches);
