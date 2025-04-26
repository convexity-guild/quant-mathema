use num_traits::{Num, NumCast};

/// Compute the mean (μ) of a data series.
///
/// ```rust
/// use quant_mathema::stats::mu;
///
/// let data = [1.0, 2.0, 3.0, 4.0];
/// let mean = mu(&data);
///
/// assert_eq!(mean, 2.5);
/// ```
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

/// Compute the mean (μ) of a data series.
///
/// ```rust
/// use quant_mathema::stats::mean;
///
/// let data = [1.0, 2.0, 3.0, 4.0];
/// let mean = mean(&data);
///
/// assert_eq!(mean, 2.5);
/// ```
pub fn mean<T>(data: &[T]) -> T
where
    T: Num + NumCast + Copy,
{
    mu(data)
}

/// Computes the median of a *sorted* data series.
///
/// NOTE: This assumes your data is already sorted,
/// to compute the median of unsorted data use
/// [`crate::stats::median_unsorted`].
///
/// # Examples
/// ```
/// use quant_mathema::stats::median;
///
/// let sorted_data = [1, 2, 3];
/// let median = median(&sorted_data);
/// assert_eq!(median, 2);
/// ```
pub fn median<T>(data: &[T]) -> T
where
    T: Num + NumCast + Copy,
{
    if data.is_empty() {
        return T::zero();
    }

    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid] + data[mid - 1]) / (T::one() + T::one())
    } else {
        data[mid]
    }
}

/// Computes the median of an *unsorted* data series.
///
/// NOTE: This assumes your data is unsorted, if your
/// data is already sorted use [`crate::stats::median`]
/// as it's faster without having to sort the data.
///
/// # Examples
/// ```
/// use quant_mathema::stats::median;
///
/// let unsorted_data = [2, 4, 3, 5, 1];
/// let median = median(&unsorted_data);
/// assert_eq!(median, 3);
/// ```
pub fn median_unsorted<T>(data: &[T]) -> T
where
    T: Num + NumCast + Copy + PartialOrd,
{
    if data.is_empty() {
        return T::zero();
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    median(&sorted)
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

    #[test]
    fn test_median_sorted_odd_length() {
        let data = [1, 2, 3];
        let result = median(&data);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_median_sorted_even_length() {
        let data = [1, 2, 3, 4];
        let result = median(&data);
        assert_eq!(result, (2 + 3) / 2);
    }

    #[test]
    fn test_median_single_element() {
        let data = [42];
        let result = median(&data);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_median_empty_slice() {
        let data: [i32; 0] = [];
        let result = median(&data);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_median_floats() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let result = median(&data);
        let expected = (2.0 + 3.0) / 2.0;
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_median_unsorted_odd_length() {
        let data = [2, 4, 3, 5, 1];
        let result = median_unsorted(&data);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_median_unsorted_even_length() {
        let data = [7, 1, 5, 3];
        let result = median_unsorted(&data);
        assert_eq!(result, (3 + 5) / 2);
    }

    #[test]
    fn test_median_unsorted_single_element() {
        let data = [42];
        let result = median_unsorted(&data);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_median_unsorted_empty_slice() {
        let data: [i32; 0] = [];
        let result = median_unsorted(&data);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_median_unsorted_floats() {
        let data = [2.5, 3.5, 1.5, 4.5];
        let result = median_unsorted(&data);
        let expected = (2.5 + 3.5) / 2.0;
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_median_unsorted_duplicates() {
        let data = [1, 2, 2, 2, 3];
        let result = median_unsorted(&data);
        assert_eq!(result, 2);
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
