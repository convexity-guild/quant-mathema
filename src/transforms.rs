use num_traits::{Num, NumCast};

#[derive(Debug, Clone, Copy)]
/// The different types of transformations which can be applied to numeric data.
pub enum TransformType {
    /// Applies the square root transformation to the data.
    ///
    /// This will compress large values more than small values,
    /// which can be useful for stabilizing variance or reducing
    /// the distortion effects of outliers.
    Sqrt,
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
}
