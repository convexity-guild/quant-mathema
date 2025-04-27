use std::error::Error as StdErrorTrait;

#[derive(Debug)]
/// The different types of `quant_mathema` errors.
pub enum Error {
    InvalidCast,
}
impl StdErrorTrait for Error {}
/// Implement display trait for `Error`
impl std::fmt::Display for Error {
    /// The error message display format
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::InvalidCast => write!(f, "Invalid cast during computation"),
        }
    }
}
