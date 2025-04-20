use crate::stats;
use num_traits::{Num, NumCast};

/// Compute the Simple Moving Average (SMA) of a data window.
///
/// Optimized for *streaming / online* workflows where you only need the
/// most‑recent window’s average. For full‑series (batch) smoothing, see
/// [`crate::smoothing::sma_series`].
///
/// # Examples
/// ```
/// use quant_mathema::smoothing::sma;
///
/// let window = [2.0, 4.0, 6.0, 8.0];
/// assert_eq!(sma(&window), 5.0);
/// ```
pub fn sma<T>(data_window: &[T]) -> T
where
    T: Num + NumCast + Copy,
{
    stats::mu(data_window)
}

/// Computes the Simple Moving Average (SMA) for
/// every sliding window for a given window length.
///
/// `sma_series` is ideal for *batch* analysis and plotting.
/// For real‑time pipelines, prefer [`crate::smoothing::sma`].
///
/// # Examples
/// ```
/// use quant_mathema::smoothing::sma_series;
/// let data = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(sma_series(&data, 2), vec![1.5, 2.5, 3.5]);
/// ```
pub fn sma_series<T>(data: &[T], window_len: usize) -> Vec<T>
where
    T: Num + NumCast + Copy,
{
    if window_len == 0 || data.len() < window_len {
        return Vec::new();
    }

    data.windows(window_len).map(sma).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "Expected {}, got {}", b, a);
    }

    #[test]
    fn test_sma_basic_integers() {
        let window = [1, 2, 3, 4, 5];
        let result = sma(&window);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_sma_floats() {
        let window: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let result = sma(&window);
        assert_close(result, 2.5, 1e-6);
    }

    #[test]
    fn test_sma_single_element() {
        let window = [42];
        let result = sma(&window);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_sma_empty() {
        let window: [f64; 0] = [];
        let result = sma(&window);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_sma_large_window() {
        let window: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = sma(&window);
        assert!((result - 5.5).abs() < 1e-6);
    }

    #[test]
    fn test_sma_generic_with_i64() {
        let window: Vec<i64> = vec![10, 20, 30, 40, 50];
        let result = sma(&window);
        assert_eq!(result, 30);
    }

    #[test]
    fn test_sma_series_fixed_window() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let window_size = 3;

        let expected = [2.0, 3.0, 4.0, 5.0];
        let result: Vec<f64> = data.windows(window_size).map(sma).collect();

        for (res, exp) in result.iter().zip(expected.iter()) {
            assert!((res - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sma_series_basic_float() {
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = sma_series(&data, 3);
        let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        assert_eq!(result.len(), expected.len());
        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_close(*res, *exp, 1e-6)
        }
    }

    #[test]
    fn test_sma_series_basic_int() {
        let data = vec![10, 20, 30, 40, 50];
        let result = sma_series(&data, 2);
        let expected = vec![15, 25, 35, 45];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sma_series_window_size_one() {
        let data = vec![3.0, 6.0, 9.0];
        let result = sma_series(&data, 1);

        assert_eq!(result, data);
    }

    #[test]
    fn test_sma_series_window_equals_data_len() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = sma_series(&data, data.len());
        assert_eq!(result.len(), 1);
        assert_close(result[0], 2.5, 1e-6)
    }

    #[test]
    fn test_sma_series_zero_window() {
        let data = vec![1.0, 2.0, 3.0];
        let result = sma_series(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sma_series_empty_data() {
        let data: Vec<f64> = vec![];
        let result = sma_series(&data, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sma_series_window_larger_than_data() {
        let data = vec![1.0, 2.0, 3.0];
        let result = sma_series(&data, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sma_series_random_data() {
        let data: Vec<f64> = vec![
            2.5, 8.0, 1.2, 9.5, 4.4, 7.7, 3.3, 6.6, 2.1, 5.5, 9.9, 1.0, 4.4, 8.8, 3.3, 7.7, 5.5,
            6.1, 2.9, 1.1, 9.0, 3.0, 4.2, 8.2, 2.0, 7.3, 6.6, 3.9, 5.2, 1.7,
        ];
        let window = 9;

        let expected = [
            5.0333333333,
            5.3666666667,
            5.5777777778,
            5.5555555556,
            4.9888888889,
            5.4777777778,
            4.9888888889,
            5.4777777778,
            5.3555555556,
            5.8000000000,
            5.5111111111,
            4.5333333333,
            5.4222222222,
            5.2666666667,
            4.7555555556,
            5.3000000000,
            4.6666666667,
            4.8666666667,
            4.9222222222,
            5.0333333333,
            5.4888888889,
            4.6777777778,
        ];

        let result = sma_series(&data, window);

        assert_eq!(result.len(), expected.len());

        for (i, (res, exp)) in result.iter().zip(expected.iter()).enumerate() {
            let diff = (res - exp).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at index {}: expected {:.3}, got {:.3}, diff = {:.6}",
                i,
                exp,
                res,
                diff
            );
        }
    }
}
