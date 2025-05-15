use num_traits::{Num, NumCast};

#[derive(Debug, Clone, Copy)]
/// The different types of transformations which can be applied to numeric data.
pub enum TransformType {
    /// Applies the square root transformation over the data.
    ///
    /// This will compress large values more than small values,
    /// which can be useful for stabilizing variance or reducing
    /// the distortion effects of outliers.
    Sqrt,

    /// Applies the natural logarithm transformation over the data.
    ///
    /// This will aggressively compress large values and expand small values
    /// which can be useful for stabilizing variance, normalizing skewed
    /// distributions, and linearizing exponential relationships.
    Log,
}

/// Applies a specified transformation to a numeric data series.
///
/// See [`crate::transforms::TransformType`] for information on the
/// supported transformations.
///
/// # Example
/// ```
/// use quant_mathema::transforms::{apply_transform, TransformType};
///
/// let data = vec![0.0, 1.0, 4.0, 9.0, 16.0];
/// let data_sqrt_trans = apply_transform(&data, TransformType::Sqrt);
/// ```
pub fn apply_transform<T>(data: &[T], transform_type: TransformType) -> Vec<f64>
where
    T: Num + NumCast + Copy,
{
    match transform_type {
        TransformType::Sqrt => sqrt_transform(data),
        TransformType::Log => log_transform(data),
    }
}

/// Applies the square root transformation to a numeric data series.
///
/// This will compress large values more than small values, which can
/// be useful for stabilizing variance or reducing the distortion effects
/// of outliers in the data series.
///
/// NOTE: The square root transform is relatively weak at taming tails
/// and is best used when only slight tail dampening is needed.
///
/// # Example
/// ```
/// use quant_mathema::transforms::sqrt_transform;
///
/// let data = vec![0.0, 1.0, 4.0, 9.0, 16.0];
/// let data_sqrt_trans = sqrt_transform(&data);
/// ```
pub fn sqrt_transform<T>(data: &[T]) -> Vec<f64>
where
    T: Num + NumCast + Copy,
{
    if data.is_empty() {
        return Vec::new();
    }

    data.iter()
        .filter_map(|&x| NumCast::from(x).map(f64::sqrt))
        .collect()
}

/// Applies the natural logarithm transformation to a numeric data series.
///
/// This will aggressively compresses large values while expanding small
/// values, which can be useful for stabilizing variance, normalizing
/// right-skewed distributions, and linearizing exponential releationships.
///
/// NOTE: This transformation is much stronger at tail compression than
/// the square root transformation [`crate::transforms::sqrt_transform`],
/// so use that if only light tail dampening is desired.
///
/// NOTE: The log transform is undefined for values less than or equal to zero.
/// Such values will be skipped silently during transformation.
///
/// # Example
/// ```
/// use quant_mathema::transforms::log_transform;
///
/// let data = vec![1.0, 2.0, 10.0, 100.0];
/// let data_log_trans = log_transform(&data);
/// ```
pub fn log_transform<T>(data: &[T]) -> Vec<f64>
where
    T: Num + NumCast + Copy,
{
    if data.is_empty() {
        return Vec::new();
    }

    data.iter()
        .filter_map(|&x| {
            let x_f: f64 = NumCast::from(x)?;
            if x_f > 0.0 { Some(x_f.ln()) } else { None }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "Expected {:.6}, got {:.6}, diff = {:.6}",
            b,
            a,
            (a - b).abs()
        );
    }

    #[test]
    fn test_sqrt_transform_with_f64() {
        let data = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let expected = [0.0, 1.0, 2.0, 3.0, 4.0];
        let result = sqrt_transform(&data);

        for (a, b) in result.into_iter().zip(expected.into_iter()) {
            assert_close(a, b, 1e-6);
        }
    }

    #[test]
    fn test_sqrt_transform_with_u32() {
        let data = vec![0u32, 1, 4, 9, 16, 25];
        let expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sqrt_transform(&data);

        for (a, b) in result.into_iter().zip(expected.into_iter()) {
            assert_close(a, b, 1e-6);
        }
    }

    #[test]
    fn test_sqrt_transform_with_empty_input() {
        let data: Vec<u32> = vec![];
        let result = sqrt_transform(&data);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sqrt_transform_skips_unconvertible_values() {
        let data = vec![1u8, 4u8, 9u8];
        let result = sqrt_transform(&data);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_log_transform_with_f64() {
        let data = vec![1.0, std::f64::consts::E, 10.0, 100.0];
        let expected = [0.0, 1.0, 10.0_f64.ln(), 100.0_f64.ln()];
        let result = log_transform(&data);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_close(*a, *b, 1e-10);
        }
    }

    #[test]
    fn test_log_transform_with_u32() {
        let data = vec![1u32, 2, 10, 100];
        let expected = vec![1.0, 2.0, 10.0, 100.0]
            .into_iter()
            .map(f64::ln)
            .collect::<Vec<_>>();
        let result = log_transform(&data);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_close(*a, *b, 1e-10);
        }
    }

    #[test]
    fn test_log_transform_skips_zeros_and_negatives() {
        let data = vec![0.0, -1.0, -100.0, 1.0];
        let result = log_transform(&data);
        assert_eq!(result.len(), 1);
        assert_close(result[0], 0.0, 1e-10);
    }

    #[test]
    fn test_log_transform_empty_input() {
        let data: Vec<f64> = vec![];
        let result = log_transform(&data);
        assert!(result.is_empty());
    }
}
