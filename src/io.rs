//! Input: two whitespace/tab-separated columns `y_true<TAB>y_score`, one sample
//! per line (file or stdin). `y_true` is binary (0/1); `y_score` is a continuous
//! decision value. Blank lines are skipped; a NaN or non-finite value fails loud
//! because scikit-learn's `assert_all_finite` rejects them.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use rsomics_common::{Result, RsomicsError};

/// Parsed samples in input order: parallel `y_true` / `y_score` columns.
#[derive(Debug, Clone)]
pub struct Samples {
    pub y_true: Vec<u8>,
    pub y_score: Vec<f64>,
}

impl Samples {
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.y_true.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.y_true.is_empty()
    }
}

/// Read a two-column file (`-` or `None` = stdin) into [`Samples`].
pub fn read_samples(path: Option<&Path>) -> Result<Samples> {
    let mut buf = String::new();
    match path {
        Some(p) if p.as_os_str() != "-" => {
            File::open(p)
                .map_err(RsomicsError::Io)?
                .read_to_string(&mut buf)
                .map_err(RsomicsError::Io)?;
        }
        _ => {
            std::io::stdin()
                .lock()
                .read_to_string(&mut buf)
                .map_err(RsomicsError::Io)?;
        }
    }
    parse_samples(&buf)
}

/// Parse whitespace-delimited `y_true y_score` rows into [`Samples`].
pub fn parse_samples(text: &str) -> Result<Samples> {
    let mut y_true = Vec::new();
    let mut y_score = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        let (Some(a), Some(b)) = (it.next(), it.next()) else {
            if line.split_whitespace().next().is_none() {
                continue;
            }
            return Err(RsomicsError::InvalidInput(format!(
                "line {}: expected two columns 'y_true y_score'",
                lineno + 1
            )));
        };
        if it.next().is_some() {
            return Err(RsomicsError::InvalidInput(format!(
                "line {}: expected exactly two columns",
                lineno + 1
            )));
        }
        let label = parse_label(a, lineno + 1)?;
        let score: f64 = fast_float2::parse(b.as_bytes()).map_err(|_| {
            RsomicsError::InvalidInput(format!("line {}: '{b}' is not a number", lineno + 1))
        })?;
        if !score.is_finite() {
            return Err(RsomicsError::InvalidInput(format!(
                "line {}: y_score '{b}' is not finite",
                lineno + 1
            )));
        }
        y_true.push(label);
        y_score.push(score);
    }
    if y_true.is_empty() {
        return Err(RsomicsError::InvalidInput("empty input".into()));
    }
    Ok(Samples { y_true, y_score })
}

fn parse_label(field: &str, lineno: usize) -> Result<u8> {
    match field {
        "0" | "0.0" => Ok(0),
        "1" | "1.0" => Ok(1),
        _ => Err(RsomicsError::InvalidInput(format!(
            "line {lineno}: y_true '{field}' must be binary 0 or 1"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_samples;

    #[test]
    fn parses_two_columns() {
        let s = parse_samples("0\t0.1\n1\t0.8\n0\t0.4\n1\t0.35\n").unwrap();
        assert_eq!(s.y_true, vec![0, 1, 0, 1]);
        assert_eq!(s.y_score, vec![0.1, 0.8, 0.4, 0.35]);
    }

    #[test]
    fn skips_blank_lines() {
        let s = parse_samples("0 0.1\n\n1 0.9\n").unwrap();
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn rejects_non_binary_label() {
        assert!(parse_samples("2 0.5\n").is_err());
    }

    #[test]
    fn rejects_three_columns() {
        assert!(parse_samples("0 0.5 0.5\n").is_err());
    }

    #[test]
    fn rejects_nonfinite_score() {
        assert!(parse_samples("1 inf\n").is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_samples("\n\n").is_err());
    }
}
