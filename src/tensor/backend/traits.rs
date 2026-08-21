use std::fmt::Debug;

/// The single extension point for compute backends.
///
/// Implementors are the only place where the actual arithmetic happens; the
/// tensor layer owns shapes, strides and validation. A backend may therefore
/// assume its inputs are already valid, and must uphold the following contract:
///
/// - buffers are contiguous and laid out in row-major order;
/// - element-wise operations receive slices of equal length and return a buffer
///   of that same length;
/// - `matmul` receives two rank-2 shapes with matching inner dimensions and
///   returns `a_shape[0] * b_shape[1]` elements;
/// - `allocate`, `zeros` and `ones` return `shape.iter().product()` elements.
///
/// Implementations keep their own assertions as a safety net for direct use of
/// the backend, outside of [`Tensor`](crate::tensor::Tensor).
pub trait Backend<T: Clone + Debug> {
    /// Allocate memory for a tensor of the given shape.
    fn allocate(&self, shape: &[usize]) -> Vec<T>;

    /// Initialize the tensor with zeros.
    fn zeros(&self, shape: &[usize]) -> Vec<T>;

    /// Initialize the tensor with ones
    fn ones(&self, shape: &[usize]) -> Vec<T>;

    /// Add two tensors element-wise
    fn add(&self, a: &[T], b: &[T]) -> Vec<T>;

    /// Subtract two tensors element-wise
    fn sub(&self, a: &[T], b: &[T]) -> Vec<T>;

    /// Multiply two tensors element-wise
    fn mul(&self, a: &[T], b: &[T]) -> Vec<T>;

    /// Divide two tensors element-wise
    fn div(&self, a: &[T], b: &[T]) -> Vec<T>;

    /// Matrix multiplication of two tensors
    fn matmul(&self, a: &[T], a_shape: &[usize], b: &[T], b_shape: &[usize]) -> Vec<T>;
}

pub trait Numeric:
    Clone
    + Copy
    + Default
    + From<u8>
    + std::fmt::Debug
    + Send
    + Sync
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
}

impl<T> Numeric for T where
    T: Clone
        + Copy
        + Default
        + From<u8>
        + std::fmt::Debug
        + Send
        + Sync
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
{
}
