//! Compat against frozen scikit-learn goldens — no scikit-learn at test time.
//!
//! `tests/golden/<name>.tsv` are `y_true y_score` inputs; `expected_scalars.tsv`
//! holds the `roc_auc_score` / `average_precision_score` values and the
//! `<name>.{roc,pr}.expected` files the full `roc_curve` /
//! `precision_recall_curve` points, all as IEEE-754 hex (`float.hex()`) under
//! scikit-learn 1.9.0 (numpy 2.4.6). Every metric is a ratio / trapezoid / step
//! sum of exact integer counts with no transcendental step, so all are asserted
//! **bit-identical** by comparing the hex bit pattern, not a tolerance.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rsomics-roc-auc"))
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn run(args: &[&str]) -> String {
    let out = Command::new(bin()).args(args).output().expect("run binary");
    assert!(
        out.status.success(),
        "binary failed for {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// IEEE-754 hex of a token, matching Python's `float.hex()`. Non-finite tokens
/// are passed through verbatim (the goldens spell them `inf`/`-inf`/`nan`).
fn to_hex(tok: &str) -> String {
    match tok {
        "inf" | "-inf" | "nan" => tok.to_string(),
        _ => tok.parse::<f64>().unwrap().to_bits_hex(),
    }
}

trait HexFloat {
    fn to_bits_hex(self) -> String;
}

impl HexFloat for f64 {
    /// Format as Python's `float.hex()`: `[-]0x1.<13 mantissa hex>p<exp>`,
    /// `0x0.0p+0` for zero. Only finite values reach here.
    fn to_bits_hex(self) -> String {
        if self == 0.0 {
            return if self.is_sign_negative() {
                "-0x0.0p+0".into()
            } else {
                "0x0.0p+0".into()
            };
        }
        let neg = self < 0.0;
        let bits = self.abs().to_bits();
        let exp_field = ((bits >> 52) & 0x7ff) as i64;
        let mantissa = bits & 0x000f_ffff_ffff_ffff;
        let (lead, unbiased) = if exp_field == 0 {
            (0u64, -1022)
        } else {
            (1u64, exp_field - 1023)
        };
        let sign = if neg { "-" } else { "" };
        format!("{sign}0x{lead}.{mantissa:013x}p{unbiased:+}")
    }
}

#[test]
fn scalars_bit_identical() {
    let expected = std::fs::read_to_string(golden_dir().join("expected_scalars.tsv")).unwrap();
    let mut checked = 0;
    for line in expected.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (name, metric, want_hex) = (c[0], c[1], c[2]);
        let path = golden_dir().join(format!("{name}.tsv"));
        let got = run(&["--metric", metric, path.to_str().unwrap()]);
        let got_hex = to_hex(got.trim());
        assert_eq!(
            got_hex,
            want_hex,
            "{name} {metric}: got {} ({got_hex}), want {want_hex}",
            got.trim()
        );
        checked += 1;
    }
    assert!(checked >= 10, "expected >= 10 scalar rows, ran {checked}");
}

fn check_curve(name: &str, metric: &str, ext: &str) {
    let exp = std::fs::read_to_string(golden_dir().join(format!("{name}.{ext}"))).unwrap();
    let path = golden_dir().join(format!("{name}.tsv"));
    let got = run(&["--metric", metric, path.to_str().unwrap()]);
    let got_lines: Vec<&str> = got.lines().collect();
    let exp_lines: Vec<&str> = exp.lines().collect();
    assert_eq!(
        got_lines.len(),
        exp_lines.len(),
        "{name} {metric}: line count {} vs {}",
        got_lines.len(),
        exp_lines.len()
    );
    for (i, (g, e)) in got_lines.iter().zip(exp_lines.iter()).enumerate() {
        let gh: Vec<String> = g.split('\t').map(to_hex).collect();
        let ef: Vec<&str> = e.split('\t').collect();
        assert_eq!(
            gh.len(),
            ef.len(),
            "{name} {metric} line {i}: field count {} vs {}",
            gh.len(),
            ef.len()
        );
        for (col, (a, b)) in gh.iter().zip(ef.iter()).enumerate() {
            assert_eq!(a, b, "{name} {metric} line {i} col {col}: {a} != {b}");
        }
    }
}

#[test]
fn roc_curves_bit_identical() {
    for name in [
        "bal_small",
        "bal_mid",
        "imb_mid",
        "bal_noties",
        "imb_big",
        "allneg_small",
        "allpos_small",
    ] {
        check_curve(name, "roc-curve", "roc.expected");
    }
}

#[test]
fn pr_curves_bit_identical() {
    for name in [
        "bal_small",
        "bal_mid",
        "imb_mid",
        "bal_noties",
        "imb_big",
        "allneg_small",
        "allpos_small",
    ] {
        check_curve(name, "pr-curve", "pr.expected");
    }
}

#[test]
fn stdin_matches_file() {
    let path = golden_dir().join("bal_mid.tsv");
    let from_file = run(&["--metric", "roc-auc", path.to_str().unwrap()]);
    let input = std::fs::read(&path).unwrap();
    let out = Command::new(bin())
        .args(["--metric", "roc-auc", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("run stdin");
    let from_stdin = String::from_utf8(out.stdout).unwrap();
    assert_eq!(from_file.trim(), from_stdin.trim(), "stdin != file");
}

#[test]
fn json_envelope_scalar() {
    let path = golden_dir().join("bal_mid.tsv");
    let out = Command::new(bin())
        .args(["--metric", "roc-auc", path.to_str().unwrap(), "--json"])
        .output()
        .expect("run --json");
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("one json envelope");
    assert_eq!(v["status"], "ok");
    assert!(v["result"]["value"].is_number(), "missing value: {s}");
}

#[test]
fn json_envelope_curve() {
    let path = golden_dir().join("bal_small.tsv");
    let out = Command::new(bin())
        .args(["--metric", "roc-curve", path.to_str().unwrap(), "--json"])
        .output()
        .expect("run --json");
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("one json envelope");
    assert_eq!(v["status"], "ok");
    assert!(v["result"]["fpr"].is_array(), "missing fpr: {s}");
    assert!(v["result"]["tpr"].is_array());
    assert!(v["result"]["thresholds"].is_array());
}

#[test]
fn empty_input_fails_loud() {
    let out = Command::new(bin())
        .args(["--metric", "roc-auc", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(b"").unwrap();
            child.wait_with_output()
        })
        .expect("run");
    assert!(!out.status.success(), "empty input must fail loud");
}

#[test]
fn non_binary_label_fails_loud() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.tsv");
    std::fs::write(&path, "2\t0.5\n0\t0.1\n").unwrap();
    let out = Command::new(bin())
        .args(["--metric", "roc-auc", path.to_str().unwrap()])
        .output()
        .expect("run");
    assert!(!out.status.success(), "non-binary label must fail loud");
}

#[test]
fn help_exits_zero() {
    let out = Command::new(bin())
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(out.status.success(), "--help did not exit 0");
}
