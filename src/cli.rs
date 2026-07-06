use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use serde::Serialize;

use rsomics_common::{CommonFlags, Result, RsomicsError, ToolMeta, run};

use rsomics_roc_auc::{
    PrCurve, RocCurve, Samples, average_precision, parse_samples, pr_curve, read_samples, roc_auc,
    roc_curve,
};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Metric {
    /// ROC AUC: area under the ROC curve (`roc_auc_score`).
    RocAuc,
    /// Average precision: the PR-curve step integral (`average_precision_score`).
    AveragePrecision,
    /// The ROC curve points: `fpr<TAB>tpr<TAB>threshold` per line (`roc_curve`).
    RocCurve,
    /// The PR curve points: `precision<TAB>recall<TAB>threshold` per line (`precision_recall_curve`).
    PrCurve,
}

/// Binary ranking metrics — value-exact `scikit-learn` `roc_auc_score`,
/// `average_precision_score`, `roc_curve`, `precision_recall_curve`.
///
/// Input is two columns `y_true<TAB>y_score`, one sample per line, from a file
/// argument or stdin (`-`): `y_true` is a binary 0/1 label, `y_score` a
/// continuous decision value (probability or margin). `--metric roc-auc`
/// (default) and `--metric average-precision` print a single number;
/// `--metric roc-curve` and `--metric pr-curve` print the curve points, one per
/// line. `roc-curve` applies scikit-learn's `drop_intermediate` default
/// (collinear points removed); `pr-curve` matches its default of keeping every
/// point.
#[derive(Parser, Debug)]
#[command(name = "rsomics-roc-auc", version, about, long_about = None)]
pub struct Cli {
    /// Which metric or curve to compute.
    #[arg(long = "metric", value_enum, default_value_t = Metric::RocAuc)]
    pub metric: Metric,

    /// Input file of `y_true y_score` rows (`-` or omitted reads stdin).
    #[arg(value_name = "INPUT")]
    pub input: Option<PathBuf>,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Report {
    Scalar {
        value: f64,
    },
    Roc {
        fpr: Vec<f64>,
        tpr: Vec<f64>,
        thresholds: Vec<f64>,
    },
    Pr {
        precision: Vec<f64>,
        recall: Vec<f64>,
        thresholds: Vec<f64>,
    },
}

impl Cli {
    fn compute(&self, s: &Samples) -> Report {
        match self.metric {
            Metric::RocAuc => Report::Scalar { value: roc_auc(s) },
            Metric::AveragePrecision => Report::Scalar {
                value: average_precision(s),
            },
            Metric::RocCurve => {
                let RocCurve {
                    fpr,
                    tpr,
                    thresholds,
                } = roc_curve(s, true);
                Report::Roc {
                    fpr,
                    tpr,
                    thresholds,
                }
            }
            Metric::PrCurve => {
                let PrCurve {
                    precision,
                    recall,
                    thresholds,
                } = pr_curve(s, false);
                Report::Pr {
                    precision,
                    recall,
                    thresholds,
                }
            }
        }
    }

    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let samples = match &self.input {
                Some(p) => read_samples(Some(p))?,
                None => {
                    use std::io::Read;
                    let mut buf = String::new();
                    std::io::stdin()
                        .lock()
                        .read_to_string(&mut buf)
                        .map_err(RsomicsError::Io)?;
                    parse_samples(&buf)?
                }
            };
            let report = self.compute(&samples);
            if !common.json {
                let stdout = std::io::stdout().lock();
                let mut w = BufWriter::new(stdout);
                write_report(&mut w, &report)?;
                w.flush().map_err(RsomicsError::Io)?;
            }
            Ok(report)
        })
    }
}

fn write_report<W: Write>(w: &mut W, report: &Report) -> Result<()> {
    match report {
        Report::Scalar { value } => writeln!(w, "{}", fmt(*value)),
        Report::Roc {
            fpr,
            tpr,
            thresholds,
        } => write_curve(w, fpr, tpr, thresholds),
        Report::Pr {
            precision,
            recall,
            thresholds,
        } => write_curve(w, precision, recall, thresholds),
    }
    .map_err(RsomicsError::Io)
}

/// Three aligned columns; the ROC/PR curve thresholds have one fewer entry than
/// the point columns (PR appends a no-threshold endpoint), so the last line of a
/// PR curve carries no threshold field.
fn write_curve<W: Write>(w: &mut W, a: &[f64], b: &[f64], thr: &[f64]) -> std::io::Result<()> {
    for i in 0..a.len() {
        match thr.get(i) {
            Some(t) => writeln!(w, "{}\t{}\t{}", fmt(a[i]), fmt(b[i]), fmt(*t))?,
            None => writeln!(w, "{}\t{}", fmt(a[i]), fmt(b[i]))?,
        }
    }
    Ok(())
}

/// Format a float exactly as Python's `repr(float)` prints it — which is what
/// scikit-learn writes for every ROC/PR/AUC value. Rust's default `Display`
/// diverges in three ways: it never switches to scientific notation (so `1e300`
/// balloons to 301 digits and `2e-05` prints as `0.00002`), and it drops the
/// trailing `.0` on integer-valued floats (so the `0.0`/`1.0` curve endpoints
/// come out as `0`/`1`). We take Rust's shortest round-trip digits from `{:e}`
/// (value-preserving) and apply CPython's presentation rule: scientific `e`
/// notation with a signed, ≥2-digit exponent when the decimal exponent is
/// `< -4` or `>= 16`, fixed notation otherwise, always keeping one trailing
/// `.0` for integer-valued fixed output. `inf`/`-inf`/`nan` as Python spells
/// them.
fn fmt(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }

    let sci = format!("{x:e}");
    let (neg, rest) = match sci.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, sci.as_str()),
    };
    let (mantissa, exp_str) = rest.split_once('e').expect("{:e} always emits an exponent");
    let exp: i32 = exp_str.parse().expect("{:e} exponent is an integer");
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let sign = if neg { "-" } else { "" };

    if !(-4..16).contains(&exp) {
        let m = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let es = if exp < 0 { '-' } else { '+' };
        format!("{sign}{m}e{es}{:02}", exp.unsigned_abs())
    } else {
        let ndigits = digits.len() as i32;
        let decpt = exp + 1;
        let body = if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt >= ndigits {
            format!("{digits}{}.0", "0".repeat((decpt - ndigits) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..decpt as usize],
                &digits[decpt as usize..]
            )
        };
        format!("{sign}{body}")
    }
}

#[cfg(test)]
mod tests {
    use super::fmt;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }

    #[test]
    fn fmt_matches_python_repr() {
        for (x, want) in [
            (0.0_f64, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (2.0, "2.0"),
            (0.5, "0.5"),
            (1.1, "1.1"),
            (12.34, "12.34"),
            (1234.5, "1234.5"),
            (100.0, "100.0"),
            (0.6666666666666666, "0.6666666666666666"),
            (0.001, "0.001"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e-8, "1e-08"),
            (2.999999999997449e-05, "2.999999999997449e-05"),
            (2.0000000000020002e-05, "2.0000000000020002e-05"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e300, "1e+300"),
            (1e-300, "1e-300"),
            (-1.5e20, "-1.5e+20"),
            (-0.2, "-0.2"),
        ] {
            assert_eq!(fmt(x), want, "fmt({x})");
        }
    }

    #[test]
    fn fmt_round_trips_value() {
        for x in [
            0.0_f64,
            1.0,
            0.6666666666666666,
            2.999999999997449e-05,
            1e300,
            1e-300,
            -1.5e20,
            std::f64::consts::PI,
        ] {
            assert_eq!(fmt(x).parse::<f64>().unwrap().to_bits(), x.to_bits());
        }
    }
}
