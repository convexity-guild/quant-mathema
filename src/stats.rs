use num_traits::{Num, NumCast, ToPrimitive};
use std::ops::Sub;

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

/// Computes the minimum value in a data series.
///
/// NOTE: Returns `None` if the input slice is empty.
///
/// # Examples
/// ```
/// use quant_mathema::stats::min;
///
/// let data = [4.0, 2.0, 7.0, 1.0, 9.0];
/// let result = min(&data);
/// assert_eq!(result, Some(1.0));
/// ```
pub fn min<T>(data: &[T]) -> Option<T>
where
    T: PartialOrd + Copy,
{
    let mut iter = data.iter().copied();
    let init_min = iter.next()?;

    Some(iter.fold(init_min, |min, x| if x < min { x } else { min }))
}

/// Computes the maximum value in a data series.
///
/// NOTE: Returns `None` if the input slice is empty.
///
/// # Examples
/// ```
/// use quant_mathema::stats::max;
///
/// let data = [4.0, 2.0, 9.0, 1.0, 7.0];
/// let result = max(&data);
/// assert_eq!(result, Some(9.0));
/// ```
pub fn max<T>(data: &[T]) -> Option<T>
where
    T: PartialOrd + Copy,
{
    let mut iter = data.iter().copied();
    let init_max = iter.next()?;

    Some(iter.fold(init_max, |max, x| if x > max { x } else { max }))
}

/// Computes the range  a data series.
///
/// NOTE: Returns `None` if the input slice is empty.
///
/// # Examples
/// ```
/// use quant_mathema::stats::range;
///
/// let data = [4.0, 2.0, 9.0, 1.0, 7.0];
/// let result = range(&data);
/// assert_eq!(result, Some(8.0));
/// ```
pub fn range<T>(data: &[T]) -> Option<T>
where
    T: PartialOrd + Copy + Sub<Output = T>,
{
    if data.is_empty() {
        return None;
    }

    let minimum = min(data).unwrap();
    let maximum = max(data).unwrap();

    Some(maximum - minimum)
}

/// Computes the 1st, 2nd, and 3rd quartiles of a numeric data series.
///
/// This function uses linear interpolation to estimate the 25th, 50th, and 75th
/// percentiles based on the formula `position = p * (n - 1)`. Where `p` is the
/// current percentile (i.e: 0.25 for 25th, 0.50 for 50th, 0.75 for 75th percentiles).
///
/// # Example
/// ```
/// use quant_mathema::stats::quartiles;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// if let Some((q1, q2, q3)) = quartiles(&data) {
///     assert_eq!(q2, 3.5); // Median
/// }
/// ```
pub fn quartiles<T>(data: &[T]) -> Option<(T, T, T)>
where
    T: Num + NumCast + Copy + PartialOrd,
{
    if data.is_empty() {
        return None;
    }

    let mut sorted = data.to_vec();
    if sorted
        .iter()
        .any(|x| x.to_f64().is_some_and(|x| x.is_nan()))
    {
        return None;
    };
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let get_quantile = |p: f64| -> Option<T> {
        let idx = p * (n - 1) as f64;
        let idx_floor = idx.floor();
        let weight = idx - idx_floor;

        let lower = sorted.get(idx_floor as usize)?.to_f64()?;
        let upper = sorted.get(idx.ceil() as usize)?.to_f64()?;
        let interpolated = lower + weight * (upper - lower);

        NumCast::from(interpolated)
    };

    Some((
        get_quantile(0.25)?,
        get_quantile(0.50)?,
        get_quantile(0.75)?,
    ))
}

/// Computes the interquartile range (IQR) of a numeric data series.
///
/// The IQR is the difference between the 75th percentile and the 25th
/// percentile of a data series, and a measure for the natural spread.
///
/// # Example
/// ```
/// use quant_mathema::stats::interquartile_range;
///
/// let data = [10, 20, 30, 40, 50];
/// if let Some(iqr) = interquartile_range(&data) {
///     assert_eq!(iqr, 20);
/// }
///```
pub fn interquartile_range<T>(data: &[T]) -> Option<T>
where
    T: Num + NumCast + Copy + PartialOrd,
{
    if let Some((q1, _, q3)) = quartiles(data) {
        Some(q3 - q1)
    } else {
        None
    }
}

/// Computes the sample variance (σ^2) of a data series.
///
/// NOTE: This calculates the sample standard deviation using
/// Bessel's correction (dividing by `n - 1` instead of `n`)
/// to reduce bias in estimation from a finite data sample.
///
/// # Examples
/// ```
/// use quant_mathema::stats::variance;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let var = variance(&data);
/// assert!((var - 2.5).abs() < 1e-6);
/// ```
pub fn variance<T>(data: &[T]) -> f64
where
    T: Num + NumCast + Copy,
{
    if data.is_empty() || data.len() < 2 {
        return 0.0;
    }

    let mu = mu(data).to_f64().unwrap_or_default();

    let dev_sum: f64 = data
        .iter()
        .map(|x| {
            let x = x.to_f64().unwrap_or_default();

            (x - mu).powi(2)
        })
        .sum();

    dev_sum / (data.len() as f64 - 1.0)
}

/// Computes the standard deviation (σ) of a numeric data series.
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
    variance(data).sqrt()
}

/// Computes the z-score of a numeric datapoint value.
///
/// Optimized for *streaming / online* workflows where you only need the
/// most‑recent window’s average. For full‑series (batch) computation, see
/// [`crate::stats::z_scores`].
///
/// The z-score measures how far a datapoint value is
/// from the mean using a standardized scale.
///
/// # Example
/// ```
/// use quant_mathema::stats::z_score;
///
/// let z = z_score(10.0f64, 5.0f64, 2.0f64);
/// assert_eq!(z, Some(2.5));
/// ```
pub fn z_score<T, F>(datapoint: T, mu: F, sigma: F) -> Option<f64>
where
    T: Num + NumCast + Copy,
    F: Into<f64> + Copy,
{
    let x_f64 = datapoint.to_f64()?;
    Some((x_f64 - mu.into()) / sigma.into())
}

/// Computes the z-scores for a numeric data series.
///
/// `z_scores` is ideal for *batch* computation and analysis.
/// For real‑time pipelines, prefer [`crate::stats::z_score`].
///
/// The z-score measures how far a datapoint value is from
/// the mean using a standardized scale.
///
/// # Example
/// ```
/// use quant_mathema::stats::z_scores;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let stdev = 2.5f64.sqrt();
/// let expected = [
///     (1.0 - 3.0) / stdev,
///     (2.0 - 3.0) / stdev,
///     (3.0 - 3.0) / stdev,
///     (4.0 - 3.0) / stdev,
///     (5.0 - 3.0) / stdev,
/// ];
///
/// if let Some(scores) = z_scores(&data) {
///     for (actual, expected) in scores.into_iter().zip(expected.into_iter()) {
///         assert!((actual - expected).abs() < 1e-6);
///     }
/// }
/// ```
pub fn z_scores<T>(data: &[T]) -> Option<Vec<f64>>
where
    T: Num + NumCast + Copy,
{
    if data.is_empty() {
        return None;
    }

    let mu = mu(data).to_f64()?;
    let sigma = stdev(data).to_f64()?;

    data.iter().map(|x| z_score(*x, mu, sigma)).collect()
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

    #[test]
    fn test_variance_basic_integers() {
        let data = [1, 2, 3, 4, 5];
        let var = variance(&data);
        assert_close(var, 2.5, 1e-6);
    }

    #[test]
    fn test_variance_basic_floats() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let var = variance(&data);
        assert_close(var, 2.5, 1e-6);
    }

    #[test]
    fn test_variance_single_element() {
        let data = [42];
        let var = variance(&data);
        assert_eq!(var, 0.0);
    }

    #[test]
    fn test_variance_empty() {
        let data: [f64; 0] = [];
        let var = variance(&data);
        assert_eq!(var, 0.0);
    }

    #[test]
    fn test_variance_duplicates() {
        let data = [3, 3, 3, 3];
        let var = variance(&data);
        assert_eq!(var, 0.0);
    }

    #[test]
    fn test_variance_negative_numbers() {
        let data = [-1, -2, -3, -4, -5];
        let var = variance(&data);
        assert_close(var, 2.5, 1e-6);
    }

    #[test]
    fn test_min_with_integers() {
        let data = [4, 2, 7, 1, 9];
        assert_eq!(min(&data), Some(1));
    }

    #[test]
    fn test_min_with_floats() {
        let data = [3.5, 2.2, 5.1, 0.1, -4.7];
        assert_eq!(min(&data), Some(-4.7));
    }

    #[test]
    fn test_min_with_one_element() {
        let data = [42];
        assert_eq!(min(&data), Some(42));
    }

    #[test]
    fn test_min_with_empty_slice() {
        let data: [i32; 0] = [];
        assert_eq!(min(&data), None);
    }

    #[test]
    fn test_min_with_duplicates() {
        let data = [5, 5, 5, 5];
        assert_eq!(min(&data), Some(5));
    }

    #[test]
    fn test_min_with_negatives() {
        let data = [-10, -20, -5, -30];
        assert_eq!(min(&data), Some(-30));
    }

    #[test]
    fn test_max_with_integers() {
        let data = [4, 2, 7, 1, 9];
        assert_eq!(max(&data), Some(9));
    }

    #[test]
    fn test_max_with_floats() {
        let data = [3.5, 2.2, 5.1, 0.1, -4.7];
        assert_eq!(max(&data), Some(5.1));
    }

    #[test]
    fn test_max_with_one_element() {
        let data = [42];
        assert_eq!(max(&data), Some(42));
    }

    #[test]
    fn test_max_with_empty_slice() {
        let data: [i32; 0] = [];
        assert_eq!(max(&data), None);
    }

    #[test]
    fn test_max_with_duplicates() {
        let data = [5, 5, 5, 5];
        assert_eq!(max(&data), Some(5));
    }

    #[test]
    fn test_max_with_negatives() {
        let data = [-10, -20, -5, -30];
        assert_eq!(max(&data), Some(-5));
    }

    #[test]
    fn test_range_with_integers() {
        let data = [4, 2, 7, 1, 9];
        assert_eq!(range(&data), Some(8));
    }

    #[test]
    fn test_range_with_floats() {
        let data = [3.5, 2.2, 5.1, 0.1, -4.7];
        let result = range(&data).unwrap();
        assert_close(result, 9.8, 1e-10);
    }

    #[test]
    fn test_range_with_one_element() {
        let data = [42];
        assert_eq!(range(&data), Some(0));
    }

    #[test]
    fn test_range_with_empty_slice() {
        let data: [i32; 0] = [];
        assert_eq!(range(&data), None);
    }

    #[test]
    fn test_range_with_duplicates() {
        let data = [5, 5, 5, 5];
        assert_eq!(range(&data), Some(0));
    }

    #[test]
    fn test_range_with_negatives() {
        let data = [-10, -20, -5, -30];
        assert_eq!(range(&data), Some(25));
    }

    #[test]
    fn test_quartiles_even_sized_data() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (q1, q2, q3) = quartiles(&data).unwrap();

        assert_close(q1, 2.25, 1e-10);
        assert_close(q2, 3.5, 1e-10);
        assert_close(q3, 4.75, 1e-10);
    }

    #[test]
    fn test_quartiles_odd_sized_data() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let (q1, q2, q3) = quartiles(&data).unwrap();

        assert_eq!(q1, 20.0);
        assert_eq!(q2, 30.0);
        assert_eq!(q3, 40.0);
    }

    #[test]
    fn test_quartiles_empty_input() {
        let data: [f64; 0] = [];
        assert_eq!(quartiles(&data), None);
    }

    #[test]
    fn test_quartiles_nan_in_input() {
        let data = [1.0, 2.0, f64::NAN, 4.0];
        assert_eq!(quartiles(&data), None);
    }

    #[test]
    fn test_quartiles_all_same_values() {
        let data = [42.0, 42.0, 42.0, 42.0, 42.0];
        let (q1, q2, q3) = quartiles(&data).unwrap();
        assert_eq!(q1, 42.0);
        assert_eq!(q2, 42.0);
        assert_eq!(q3, 42.0);
    }

    #[test]
    fn test_quartiles_unsorted_input() {
        let data = [9.0, 3.0, 1.0, 10.0, 5.0, 6.0, 2.0, 4.0, 8.0, 7.0];
        let (q1, q2, q3) = quartiles(&data).unwrap();

        assert_close(q1, 3.25, 1e-10);
        assert_close(q2, 5.5, 1e-10);
        assert_close(q3, 7.75, 1e-10);
    }

    #[test]
    fn test_iqr_empty() {
        let data: [f64; 0] = [];
        assert_eq!(interquartile_range(&data), None);
    }

    #[test]
    fn test_iqr_single_element() {
        let data = [42.0];
        assert_eq!(interquartile_range(&data), Some(0.0));
    }

    #[test]
    fn test_iqr_even_length() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let result = interquartile_range(&data);
        assert_eq!(result, Some(1.5));
    }

    #[test]
    fn test_iqr_odd_length() {
        let data = [
            7.0, 15.0, 36.0, 39.0, 40.0, 41.0, 42.0, 43.0, 47.0, 49.0, 50.0,
        ];
        let result = interquartile_range(&data);
        assert_eq!(result, Some(7.5));
    }

    #[test]
    fn test_iqr_unsorted() {
        let data = [5.0, 1.0, 3.0, 2.0, 4.0];
        let result = interquartile_range(&data);
        assert_eq!(result, Some(2.0));
    }

    #[test]
    fn test_iqr_integers() {
        let data = [10, 20, 30, 40, 50];
        let result = interquartile_range(&data);
        assert_eq!(result, Some(20));
    }

    #[test]
    fn test_z_score_f64_inputs() {
        let z = z_score(10.0f64, 5.0f64, 2.0f64);
        assert_eq!(z, Some(2.5));
    }

    #[test]
    fn test_z_score_f32_inputs() {
        let z = z_score(10.0f32, 5.0f32, 2.0f32);
        assert_eq!(z, Some(2.5));
    }

    #[test]
    fn test_z_score_integer_datapoint() {
        let z = z_score(10i32, 5.0f64, 2.0f64);
        assert_eq!(z, Some(2.5));
    }

    #[test]
    fn test_z_score_zero_sigma() {
        let z = z_score(10.0f64, 5.0f64, 0.0f64);
        assert!(z.unwrap().is_infinite());
    }

    #[test]
    fn test_z_scores_precise() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        let stdev = 2.5f64.sqrt();
        let expected = [
            (1.0 - 3.0) / stdev,
            (2.0 - 3.0) / stdev,
            (3.0 - 3.0) / stdev,
            (4.0 - 3.0) / stdev,
            (5.0 - 3.0) / stdev,
        ];

        let result = z_scores(&data).unwrap();

        for (actual, expected) in result.into_iter().zip(expected.into_iter()) {
            assert_close(actual, expected, 1e-10);
        }
    }

    #[test]
    fn test_z_scores_symmetric_centered() {
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0];

        let stdev = 2.5f64.sqrt();
        let expected = [-2.0 / stdev, -1.0 / stdev, 0.0, 1.0 / stdev, 2.0 / stdev];

        let result = z_scores(&data).unwrap();

        for (actual, expected) in result.into_iter().zip(expected.into_iter()) {
            assert_close(actual, expected, 1e-10);
        }
    }
}
