use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Errors returned by the fallible tensor operations (`try_*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// A shape with no dimension was provided.
    EmptyShape,

    /// A shape contains a dimension of size zero.
    ZeroDimension { axis: usize },

    /// The data buffer does not match the number of elements described by the shape.
    DataLengthMismatch { actual: usize, expected: usize },

    /// The number of indices does not match the rank of the tensor.
    RankMismatch { actual: usize, expected: usize },

    /// An index is outside of the bounds of its axis.
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        size: usize,
    },

    /// The requested shape does not hold the same number of elements.
    ReshapeSizeMismatch { actual: usize, requested: usize },

    /// Two operands of an element-wise operation have different shapes.
    ShapeMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    /// An operation was applied to a tensor of the wrong rank.
    ExpectedRank {
        op: &'static str,
        expected: usize,
        actual: usize,
    },

    /// The inner dimensions of a matrix multiplication do not line up.
    MatmulInnerMismatch { lhs_cols: usize, rhs_rows: usize },

    /// A backend was requested but is not compiled in, or no device was found.
    BackendUnavailable { requested: &'static str },
}

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::EmptyShape => write!(f, "shape must have at least one dimension"),
            Self::ZeroDimension { axis } => {
                write!(f, "dimension {axis} has size 0, which is not supported")
            }
            Self::DataLengthMismatch { actual, expected } => write!(
                f,
                "data length {actual} does not match the {expected} elements described by the shape"
            ),
            Self::RankMismatch { actual, expected } => write!(
                f,
                "expected {expected} indices for a {expected}-dimensional tensor, got {actual}"
            ),
            Self::IndexOutOfBounds { axis, index, size } => write!(
                f,
                "index {index} is out of bounds for dimension {axis} of size {size}"
            ),
            Self::ReshapeSizeMismatch { actual, requested } => write!(
                f,
                "cannot reshape a tensor of {actual} elements into a shape of {requested} elements"
            ),
            Self::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "tensor shapes must match for {op}: {lhs:?} vs {rhs:?}")
            }
            Self::ExpectedRank {
                op,
                expected,
                actual,
            } => write!(
                f,
                "{op} expects a {expected}-dimensional tensor, got {actual} dimensions"
            ),
            Self::MatmulInnerMismatch { lhs_cols, rhs_rows } => write!(
                f,
                "inner dimensions must match for matrix multiplication: {lhs_cols} vs {rhs_rows}"
            ),
            Self::BackendUnavailable { requested } => {
                write!(f, "the {requested} backend is not available on this build")
            }
        }
    }
}

impl Error for TensorError {}

pub type TensorResult<T> = Result<T, TensorError>;
