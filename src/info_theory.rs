use crate::stats::{max, min};
use crate::util::bin_series;
use matrix_oxide::Matrix;
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

/// Computes the mutual information (MI) between two numeric data series.
///
/// Mutual information quantifies how much knowing one of the variables
/// reduces uncertainty about the other. It measures the amount of
/// shared information between `a` and `b`.
///
/// - A value of `0.0` indicates complete independence.
/// - A higher value indicates stronger dependency or correlation.
/// - The MI is expressed in **nats** (natural log base).
///
/// This implementation uses equal-width binning to discretize the
/// continuous or numeric inputs before computing joint and marginal
/// distributions.
///
/// # Examples
/// ```
/// use quant_mathema::info_theory::mutual_information;
///
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let b = [5.0, 4.0, 3.0, 2.0, 1.0];
///
/// if let Some(mi) = mutual_information(&a, &b, 5) {
///     assert!(mi > 0.0);
/// }
/// ```
pub fn mutual_information<T>(a: &[T], b: &[T], n_bins: usize) -> Option<f64>
where
    T: Num + NumCast + Copy + PartialOrd,
{
    if a.len() != b.len() || a.is_empty() || b.is_empty() {
        return None;
    }

    let a_bins = bin_series(a, n_bins)?;
    let b_bins = bin_series(b, n_bins)?;

    let mut joint = Matrix::<f64>::new(n_bins, n_bins);
    a_bins.iter().zip(b_bins).for_each(|(i, j)| {
        if let Some(cell) = joint.get_mut(*i, j) {
            *cell += 1.0;
        }
    });

    let total = joint.sum();
    if total == 0.0 {
        return Some(0.0);
    }

    (0..joint.row_size).for_each(|i| {
        (0..joint.col_size).for_each(|j| {
            if let Some(cell) = joint.get_mut(i, j) {
                *cell /= total;
            }
        })
    });

    let mut px = vec![0.0; joint.row_size];
    let mut py = vec![0.0; joint.col_size];

    (0..joint.row_size).for_each(|i| {
        (0..joint.col_size).for_each(|j| {
            if let Some(cell) = joint.get_mut(i, j) {
                px[i] += *cell;
                py[j] += *cell;
            }
        })
    });

    let mut mi = 0.0;
    (0..n_bins).for_each(|i| {
        (0..n_bins).for_each(|j| {
            if let Some(pxy) = joint.get(i, j) {
                if pxy > &0.00 {
                    mi += pxy * (pxy / (px[i] * py[j])).ln()
                }
            }
        })
    });

    Some(mi)
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

    #[test]
    fn test_mi_identical_inputs() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mi = mutual_information(&a, &a, 5).unwrap();
        assert!(mi > 0.0, "MI of identical inputs should be > 0");
    }

    #[test]
    fn test_mi_inverse_inputs() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let b: Vec<_> = a.iter().rev().cloned().collect();
        let mi = mutual_information(&a, &b, 5).unwrap();
        assert!(mi > 0.0, "MI of reversed input should be > 0");
    }

    #[test]
    fn test_mi_constant_input() {
        let a = vec![1; 100];
        let b: Vec<_> = (0..100).collect();
        let mi = mutual_information(&a, &b, 5).unwrap();
        assert!(
            (mi - 0.0).abs() < 1e-10,
            "MI should be ~0 when one series is constant"
        );
    }

    #[test]
    fn test_mi_random_noise() {
        let a = vec![1, 3, 5, 2, 4, 6, 8, 10, 7, 9];
        let b = vec![10, 8, 6, 9, 7, 5, 3, 1, 4, 2];
        let mi = mutual_information(&a, &b, 5).unwrap();
        assert!(mi >= 0.0, "MI must be non-negative");
    }

    #[test]
    fn test_mi_different_lengths() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5];
        assert!(
            mutual_information(&a, &b, 3).is_none(),
            "Mismatched input lengths should return None"
        );
    }

    #[test]
    fn test_mi_symmetry() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![5, 4, 3, 2, 1];
        let mi_ab = mutual_information(&a, &b, 4).unwrap();
        let mi_ba = mutual_information(&b, &a, 4).unwrap();
        assert!((mi_ab - mi_ba).abs() < 1e-10, "MI should be symmetric");
    }
}
