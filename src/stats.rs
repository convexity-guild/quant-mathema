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
}
