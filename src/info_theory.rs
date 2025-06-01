use crate::stats::{max, min};
use num_traits::{Num, NumCast};

/// Computes the normalized Shannon entropy of a numeric data series.
///
/// This estimates the entropy of a continuous data series which tells
/// you the the upper limit on the information content of the data.
///
/// - `0.0` indicates no uncertainty (all data in one bin),
/// - `1.0` indicates maximal uncertainty (uniform distribution).
///
/// The higher the entropy the more information it may carry, and the
/// lower the entropy the less information it may carry.
///
/// # Examples
/// ```
/// use quant_mathema::info_theory::normalized_entropy;
///
/// let data = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 1.0, 4.0, 3.0, 2.0];
/// let entropy = normalized_entropy(&data, 5);
/// assert!(entropy.unwrap() > 0.5 && entropy.unwrap() < 1.0);
/// ```
pub fn normalized_entropy<T>(data: &[T], n_bins: u8) -> Option<f64>
where
    T: Num + NumCast + Copy + PartialOrd,
{
    if data.is_empty() {
        return None;
    }

    // SAFETY: With the invariant check above, we know the data
    // is not empty, so there must be some minimum & maximum.
    let x_min = min(data).and_then(|x| x.to_f64()).unwrap();
    let x_max = max(data).and_then(|x| x.to_f64()).unwrap();

    // SAFETY: We guard against k being out of bounds by
    // subtracting the number of bins by a small epsilon
    // (`n_bins - ε`).
    //
    // We also add the denominator by a tiny epsilon as cheap
    // insurance against dividing by 0 (`(x_max - x_min) + ε`).
    let factor = (n_bins as f64 - 1e-11) / ((x_max - x_min) + 1e-60);
    let mut bin_counts = vec![0u32; n_bins as usize];
    data.iter().for_each(|x| {
        let k = (factor * (x.to_f64().unwrap() - x_min)) as usize;
        bin_counts[k] += 1;
    });

    let entropy_sum = bin_counts
        .iter()
        .copied()
        .fold(0.0, |entropy_sum, bin_count| {
            if bin_count == 0 {
                entropy_sum
            } else {
                let bin_probability = bin_count as f64 / data.len() as f64;
                entropy_sum - (bin_probability * bin_probability.ln())
            }
        });

    let relative_entropy = entropy_sum / (n_bins as f64).ln();

    Some(relative_entropy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_all_same() {
        let data = [1.0; 100];
        let entropy = normalized_entropy(&data, 10);
        assert_eq!(entropy, Some(0.0));
    }

    #[test]
    fn test_entropy_uniform_distribution() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let entropy = normalized_entropy(&data, 10);
        assert!(entropy.unwrap() > 0.95);
    }

    #[test]
    fn test_entropy_random_distribution() {
        let data = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 1.0, 4.0, 3.0, 2.0];
        let entropy = normalized_entropy(&data, 5);
        assert!(entropy.unwrap() > 0.5 && entropy.unwrap() < 1.0);
    }

    #[test]
    fn test_entropy_single_value() {
        let data = [42.0];
        let entropy = normalized_entropy(&data, 5);
        assert_eq!(entropy, Some(0.0));
    }

    #[test]
    fn test_entropy_empty_slice() {
        let data: [f64; 0] = [];
        let entropy = normalized_entropy(&data, 5);
        assert_eq!(entropy, None);
    }

    #[test]
    fn test_entropy_maximum_when_even_bins() {
        let data = [0.0, 1.0, 2.0, 3.0];
        let entropy = normalized_entropy(&data, 4);
        assert!((entropy.unwrap() - 1.0).abs() < 1e-6);
    }
}
