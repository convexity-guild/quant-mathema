use crate::{Error, stats};
use num_traits::{Num, NumCast, ToPrimitive};

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
///
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

/// Compute the Exponential Moving Average (EMA) of a data window.
///
/// Optimized for *streaming / online* workflows where you only need the
/// most‑recent window’s average. For full‑series (batch) smoothing, see
/// [`crate::smoothing::ema_series`].
///
/// # Examples
/// ```
/// use quant_mathema::smoothing::ema;
///
/// let window = [2.0, 4.0, 6.0, 8.0];
/// let ema_value: f64 = ema(&window);
/// assert!((ema(&window) - 5.648).abs() < 1e-6);
/// ```
pub fn ema<T>(data_window: &[T]) -> f64
where
    T: Num + NumCast + Copy,
{
    let len = data_window.len();
    if len == 0 {
        return 0.0;
    }

    let alpha = 2.0 / (len as f64 + 1.0);

    data_window
        .iter()
        .map(|x| NumCast::from(*x).unwrap_or(0.0))
        .skip(1)
        .fold(NumCast::from(data_window[0]).unwrap_or(0.0), |ema, x| {
            alpha * x + (1.0 - alpha) * ema
        })
}

/// Computes the Exponential Moving Average (EMA) for
/// every sliding window for a given window length.
///
/// `ema_series` is ideal for *batch* analysis and plotting.
/// For real‑time pipelines, prefer [`crate::smoothing::ema`].
///
/// # Examples
/// ```
/// use quant_mathema::smoothing::ema_series;
///
/// let data = [1.0, 2.0, 3.0, 4.0];
/// let result = ema_series(&data, 2);
/// let expected = vec![1.6666667, 2.6666667, 3.6666667];
///
/// for (a, b) in result.iter().zip(expected.iter()) {
///     assert!((a - b).abs() < 1e-6);
/// }
/// ```
pub fn ema_series<T>(data: &[T], window_len: usize) -> Vec<f64>
where
    T: Num + NumCast + Copy,
{
    if window_len == 0 || data.len() < window_len {
        return Vec::new();
    }

    data.windows(window_len).map(ema).collect()
}

/// Compute the Volume Weighted Moving Average (VWMA) of a data window.
///
/// Optimized for *streaming / online* workflows where you only need the
/// most‑recent window’s volume weighted average. For full‑series (batch)
/// smoothing, see [`crate::smoothing::ema_series`].
///
/// NOTE: The length of the volume window must match the length
/// of the data window being smoothed, else returns zero.
///
/// # Examples
/// ```
/// use quant_mathema::smoothing::vwma;
///
/// let price = [100.0, 101.0, 102.0];
/// let volume = [10, 20, 30];
///
/// let vwma_value = vwma(&price, &volume);
/// ```
pub fn vwma<T, V>(data_window: &[T], volume_window: &[V]) -> f64
where
    T: ToPrimitive,
    V: ToPrimitive,
{
    if data_window.len() != volume_window.len() || data_window.is_empty() {
        return 0.0;
    }

    let volume_sum = volume_window
        .iter()
        .fold(f64::default(), |sum, x| sum + (x.to_f64().unwrap_or(0.0)));

    if volume_sum > 0.0 {
        data_window
            .iter()
            .zip(volume_window.iter())
            .map(|(d, v)| d.to_f64().unwrap_or(0.0) * v.to_f64().unwrap_or(0.0))
            .sum::<f64>()
            / volume_sum
    } else {
        data_window
            .last()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .unwrap_or(0.0)
    }
}

/// Computes the Volume Weighted Moving Average (VWMA) for
/// every sliding window for a given window length.
///
/// `vwma_series` is ideal for *batch* analysis and plotting.
/// For real‑time pipelines, prefer [`crate::smoothing::vwma`].
///
/// NOTE: The length of the volume data must match the length
/// of the data being smoothed, else returns empty `Vec`.
///
/// # Examples
/// ```
/// use quant_mathema::smoothing::vwma_series;
///
/// let price = [100.0, 101.0, 102.0, 103.0];
/// let volume = [10, 20, 30, 40];
///
/// let vwma_values = vwma_series(&price, &volume, 3);
/// ```
pub fn vwma_series<T, V>(data: &[T], volume_data: &[V], window_len: usize) -> Vec<f64>
where
    T: ToPrimitive + Copy,
    V: ToPrimitive + Copy,
{
    if data.len() != volume_data.len() || data.is_empty() {
        return Vec::new();
    }

    data.windows(window_len)
        .zip(volume_data.windows(window_len))
        .map(|(data_window, volume_window)| vwma(data_window, volume_window))
        .collect()
}

// TODO: The bulk of this calculation should be seperated into a
// smaller regression sums like function, as it will need to
// implemented down the road for other things as well.
//
/// Compute the Least Squares Moving Average (LSMA) of a data window.
///
/// Optimized for *streaming / online* workflows where you only need the
/// most‑recent window’s volume weighted average. For full‑series (batch)
/// smoothing, see [`crate::smoothing::lsma_series`].
///
/// # Example
/// ```
/// use quant_mathema::smoothing::lsma;
///
/// let data = vec![
///     111.50, 111.53, 111.55, 111.55, 111.56, 111.58, 111.58,
///     111.58, 111.58, 111.59, 111.59, 111.51, 111.64, 111.70,
/// ];
/// let result = lsma(&data).unwrap();
///
/// assert!((result - 111.62).abs() < 0.1);
/// ```
pub fn lsma<T>(data_window: &[T]) -> Result<f64, Error>
where
    T: Num + NumCast + Copy,
{
    if data_window.len() == 1 {
        return NumCast::from(data_window[0]).ok_or(Error::InvalidCast);
    }

    let n: f64 = NumCast::from(data_window.len()).ok_or(Error::InvalidCast)?;
    let x_sum = data_window.iter().copied().try_fold(0.0, |sum, x| {
        let x: f64 = NumCast::from(x).ok_or(Error::InvalidCast)?;
        Ok(sum + x)
    })?;
    let t_sum = (0..data_window.len()).try_fold(0.0, |sum, t| {
        let t: f64 = NumCast::from(t).ok_or(Error::InvalidCast)?;

        Ok(sum + t)
    })?;
    let t_squared_sum = (0..data_window.len()).try_fold(0.0, |sum, t| {
        let t: f64 = NumCast::from(t).ok_or(Error::InvalidCast)?;

        Ok(sum + (t * t))
    })?;
    let x_t_sum = data_window
        .iter()
        .copied()
        .enumerate()
        .try_fold(0.0, |sum, (t, x)| {
            let t: f64 = NumCast::from(t).ok_or(Error::InvalidCast)?;
            let x: f64 = NumCast::from(x).ok_or(Error::InvalidCast)?;

            Ok(sum + (t * x))
        })?;

    let slope_numerator = n * x_t_sum - t_sum * x_sum;
    let slope_denominator = n * t_squared_sum - t_sum * t_sum;
    let slope = slope_numerator / slope_denominator;

    let intercept = (x_sum - slope * t_sum) / n;

    let output = slope * (n - 1.0) + intercept;
    Ok(output)
}

/// Compute the Least Squares Moving Average (LSMA) of a data window.
///
/// `lsma_series` is ideal for *batch* analysis and plotting.
/// For real‑time pipelines, prefer [`crate::smoothing::lsma`].
///
/// # Example
/// ```
/// use quant_mathema::smoothing::lsma_series;
///
/// let data = vec![
///     111.50, 111.53, 111.55, 111.55, 111.56, 111.58, 111.58,
///     111.58, 111.58, 111.59, 111.59, 111.51, 111.64, 111.70,
/// ];
/// let result = lsma_series(&data, 5).unwrap();
/// ```
pub fn lsma_series<T>(data: &[T], window_len: usize) -> Result<Vec<f64>, Error>
where
    T: Num + NumCast + Copy,
{
    if data.is_empty() || window_len == 0 {
        return Ok(Vec::new());
    }

    data.windows(window_len)
        .try_fold(Vec::new(), |mut series, data_window| {
            let lsma_value = lsma(data_window)?;
            series.push(lsma_value);
            Ok(series)
        })
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

    #[test]
    fn test_ema_basic_float() {
        let window = [2.0, 4.0, 6.0, 8.0];
        let result = ema(&window);
        let expected = 5.648;
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_ema_single_element() {
        let window = [42.0];
        let result: f64 = ema(&window);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_ema_empty() {
        let window: [f64; 0] = [];
        let result: f64 = ema(&window);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_ema_two_elements() {
        let window = [10.0, 20.0];
        let alpha = 2.0 / (2.0 + 1.0);
        let expected = alpha * 20.0 + (1.0 - alpha) * 10.0;
        let result = ema(&window);
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_ema_i32_integer_input() {
        let window = [2, 4, 6, 8];
        let result: f64 = ema(&window);
        assert_eq!(result, 5.648);
    }

    #[test]
    fn test_ema_longer_window() {
        let window = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = ema(&window);

        let expected = 3.964836;
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_ema_series_basic_float() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema_series(&data, 3);
        let expected = [2.25, 3.25, 4.25];

        assert_eq!(result.len(), expected.len());
        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_close(*res, *exp, 1e-6);
        }
    }

    #[test]
    fn test_ema_series_int() {
        let data = [10, 20, 30, 40, 50, 60];
        let result = ema_series(&data, 3);
        let expected = [22.5, 32.5, 42.5, 52.5];

        assert_eq!(result.len(), expected.len());
        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_close(*res, *exp, 1e-6);
        }
    }

    #[test]
    fn test_ema_series_window_equals_data_len() {
        let data = [5.0, 10.0, 15.0];
        let result = ema_series(&data, data.len());
        let expected = 11.25;

        assert_eq!(result.len(), 1);
        assert_close(result[0], expected, 1e-6);
    }

    #[test]
    fn test_ema_series_window_one() {
        let data = [2.0, 4.0, 6.0];
        let result = ema_series(&data, 1);
        let expexted = [2.0, 4.0, 6.0];

        assert_eq!(result, expexted);
    }

    #[test]
    fn test_ema_series_empty_data() {
        let data: [f64; 0] = [];
        let result = ema_series(&data, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_ema_series_window_too_large() {
        let data = [1.0, 2.0];
        let result = ema_series(&data, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_ema_series_window_zero() {
        let data = [1.0, 2.0, 3.0];
        let result = ema_series(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_vwma_basic_integer_volume() {
        let price = [100.0, 101.0, 102.0];
        let volume = [10, 20, 30];

        let result = vwma(&price, &volume);
        let expected = 101.333333;
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_vwma_with_floating_volume() {
        let price = [1.0, 2.0, 3.0];
        let volume = [0.1, 0.2, 0.7];

        let result = vwma(&price, &volume);
        let expected = 2.6;
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_vwma_equal_weights_equals_sma() {
        let price = [10.0, 20.0, 30.0];
        let volume = [1, 1, 1];

        let result = vwma(&price, &volume);
        let expected = 20.0;
        assert_close(result, expected, 1e-6);
    }

    #[test]
    fn test_vwma_zero_volume() {
        let price = [100.0, 101.0, 102.0];
        let volume = [0, 0, 0];

        let result = vwma(&price, &volume);
        assert_eq!(result, 102.0);
    }

    #[test]
    fn test_vwma_mismatched_lengths() {
        let price = [1.0, 2.0];
        let volume = [10];

        let result = vwma(&price, &volume);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_vwma_empty_input() {
        let price: [f64; 0] = [];
        let volume: [i32; 0] = [];

        let result = vwma(&price, &volume);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_vwma_series_basic_integer() {
        let price = [100.0, 101.0, 102.0, 103.0];
        let volume = [10, 20, 30, 40];

        let result = vwma_series(&price, &volume, 3);
        let expected = [101.333333, 102.222222];

        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_close(*res, *exp, 1e-6);
        }
    }

    #[test]
    fn test_vwma_series_floating_volume() {
        let price = [1.0, 2.0, 3.0, 4.0];
        let volume = [0.1, 0.2, 0.7, 1.0];

        let result = vwma_series(&price, &volume, 3);
        let expected = [2.6, 3.4210526];

        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_close(*res, *exp, 1e-6);
        }
    }

    #[test]
    fn test_vwma_series_equal_volume_is_sma() {
        let price = [1.0, 2.0, 3.0, 4.0];
        let volume = [1.0, 1.0, 1.0, 1.0];

        let result = vwma_series(&price, &volume, 3);
        let expected = [2.0, 3.0];

        for (res, exp) in result.iter().zip(expected.iter()) {
            assert_close(*res, *exp, 1e-6);
        }
    }

    #[test]
    fn test_vwma_series_mismatched_lengths() {
        let price = [1.0, 2.0, 3.0];
        let volume = [1.0, 2.0];

        let result = vwma_series(&price, &volume, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_vwma_series_window_too_large() {
        let price = [1.0, 2.0];
        let volume = [1.0, 2.0];

        let result = vwma_series(&price, &volume, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_vwma_series_empty_input() {
        let price: [f64; 0] = [];
        let volume: [f64; 0] = [];

        let result = vwma_series(&price, &volume, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_lsma_perfect_linear_data() {
        let data = vec![3, 5, 7, 9, 11];
        let result = lsma(&data).unwrap();

        assert_close(result, 11.0, 1e-6);
    }

    #[test]
    fn test_lsma_constant_data() {
        let data = vec![5, 5, 5, 5, 5];
        let result = lsma(&data).unwrap();

        assert_close(result, 5.0, 1e-6);
    }

    #[test]
    fn test_lsma_single_point() {
        let data = vec![42];
        let result = lsma(&data).unwrap();

        assert_close(result, 42.0, 1e-6);
    }

    #[test]
    fn test_lsma_floating_point_data() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let result = lsma(&data).unwrap();

        assert_close(result, 9.0, 1e-6);
    }

    #[test]
    fn test_lsma_small_random_data() {
        let data = vec![1, 2, 4, 7];
        let result = lsma(&data).unwrap();

        assert_close(result, 6.5, 1e-6);
    }

    #[test]
    fn test_lsma_random_data() {
        let data = vec![
            111.50, 111.53, 111.55, 111.55, 111.56, 111.58, 111.58, 111.58, 111.58, 111.59, 111.59,
            111.51, 111.64, 111.70,
        ];

        let result = lsma(&data).unwrap();
        assert_close(result, 111.62, 0.1);
    }

    #[test]
    fn test_lsma_series_empty_data() {
        let data: Vec<f64> = vec![];
        let result = lsma_series(&data, 5).unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_lsma_series_zero_window_len() {
        let data = vec![1.0, 2.0, 3.0];
        let result = lsma_series(&data, 0).unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_lsma_series_window_larger_than_data() {
        let data = vec![1.0, 2.0, 3.0];
        let result = lsma_series(&data, 5).unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_lsma_series_exact_data_len_window() {
        let data = vec![1.0, 2.0, 3.0];
        let result = lsma_series(&data, 3).unwrap();

        assert_eq!(result.len(), 1);
        assert_close(result[0], 3.0, 1e-6);
    }

    #[test]
    fn test_lsma_series_sliding_windows() {
        let data = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let result = lsma_series(&data, 3).unwrap();

        assert_eq!(result.len(), 3);

        assert_close(result[0], 3.83, 1e-2);
        assert_close(result[1], 6.83, 1e-2);
        assert_close(result[2], 10.83, 1e-2);
    }

    #[test]
    fn test_lsma_series_random_data() {
        let data = vec![
            111.50, 111.53, 111.55, 111.55, 111.56, 111.58, 111.58, 111.58, 111.58, 111.59, 111.59,
            111.51, 111.64, 111.70,
        ];

        let window_len = 5;
        let result = lsma_series(&data, window_len).unwrap();

        assert_eq!(result.len(), data.len() - window_len + 1);

        let expected = vec![
            111.57, 111.58, 111.58, 111.59, 111.59, 111.59, 111.59, 111.54, 111.58, 111.66,
        ];

        assert_eq!(result.len(), expected.len());

        for (ith_result, ith_expected) in result.iter().copied().zip(expected) {
            assert_close(ith_result, ith_expected, 1e-2);
        }
    }
}
