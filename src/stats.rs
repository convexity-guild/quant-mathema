use num_traits::{Num, NumCast};

/// Compute the mean (mu) of a data series.
pub fn mu<T>(data: &[T]) -> T
where
    T: Num + NumCast + Copy,
{
    if data.is_empty() {
        return T::zero();
    }

    let sum = data.iter().copied().fold(T::zero(), |sum, x| sum + x);

    // NOTE: This unwrap is safe as `data.len` returns
    // a `usize` which will always be non-negative.
    let n = NumCast::from(data.len()).unwrap();

    sum / n
}

/// Compute the standard deviation of a numeric data series.
///
/// NOTE: This calculates the sample standard deviation using
/// Bessel's correction (dividing by `n - 1` instead of `n`)
/// to reduce bias in estimation from a finite data sample.
///
/// # Example
/// ```
/// use quant_mathema::stats::stdev;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let std_dev = stdev(&data);
/// assert!((std_dev - 1.58).abs() < 0.01);
/// ```
pub fn stdev<T>(data: &[T]) -> f64
where
    T: Num + NumCast + Copy,
{
    if data.is_empty() || data.len() < 2 {
        return 0.0;
    }

    let mu = mu(data).to_f64().unwrap_or_default();

    let dev_sum = data.iter().fold(0.0, |sum, x| {
        let x = x.to_f64().unwrap_or_default();

        sum + (x - mu).powi(2)
    });

    let dev = dev_sum / (data.len() as f64 - 1.0);

    dev.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mu_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mu(&data), 3.0);
    }

    #[test]
    fn test_mu_single_element() {
        let data = vec![42];
        assert_eq!(mu(&data), 42);
    }

    #[test]
    fn test_mu_empty() {
        let data: Vec<f64> = vec![];
        let result = mu(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_mu_negative_numbers() {
        let data = vec![-1.0, -2.0, -3.0];
        assert_eq!(mu(&data), -2.0);
    }

    #[test]
    fn test_mu_mixed_numbers() {
        let data = vec![-2.0, 0.0, 2.0];
        assert_eq!(mu(&data), 0.0);
    }

    #[test]
    fn test_mu_bigger_sample_f64() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = mu(&data);
        let expected = 5.5;
        let epsilon = 1e-10;

        assert!(
            (result - expected).abs() < epsilon,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_mu_bigger_sample_integers() {
        let data = vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
        let result = mu(&data);
        let expected = 11;
        assert_eq!(result, expected);
    }

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
    fn test_stdev_basic_floats() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let expected = 2.138089935;
        let result = stdev(&data);
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_stdev_integers() {
        let data = [1, 2, 3, 4, 5];
        let expected = 1.58113883;
        let result = stdev(&data);
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_stdev_identical_values() {
        let data = [42.0, 42.0, 42.0];
        let result = stdev(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_stdev_single_element() {
        let data = [99.0];
        let result = stdev(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_stdev_empty() {
        let data: [f64; 0] = [];
        let result = stdev(&data);
        assert_eq!(result, 0.0);
    }
}
