use num_traits::{Num, NumCast};

pub(crate) fn bin_series<T>(data: &[T], n_bins: usize) -> Option<Vec<usize>>
where
    T: Num + NumCast + Copy + PartialOrd,
{
    let fdata: Vec<f64> = data.iter().filter_map(|&x| x.to_f64()).collect();
    if fdata.is_empty() {
        return None;
    }

    let min = fdata.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = fdata.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (max - min).max(f64::EPSILON) / n_bins as f64;

    Some(
        fdata
            .iter()
            .map(|&x| {
                let mut bin = ((x - min) / bin_width).floor() as usize;
                if bin >= n_bins {
                    bin = n_bins - 1;
                }
                bin
            })
            .collect(),
    )
}
