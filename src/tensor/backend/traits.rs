use std::fmt::Debug;

pub use crate::tensor::numeric::{Float, Numeric};

/// The single extension point for compute backends.
///
/// Implementors are the only place where the actual arithmetic happens; the
/// tensor layer owns shapes, strides and validation. A backend may therefore
/// assume its inputs are already valid, and must uphold the following contract:
///
/// - buffers are contiguous and laid out in row-major order;
/// - element-wise operations receive buffers of equal length and return a
///   buffer of that same length;
/// - `matmul` receives two rank-2 shapes with matching inner dimensions and
///   returns `a_shape[0] * b_shape[1]` elements;
/// - `allocate`, `zeros` and `ones` return `shape.iter().product()` elements.
///
/// Implementations keep their own assertions as a safety net for direct use of
/// the backend, outside of [`Tensor`](crate::tensor::Tensor).
pub trait Backend<T: Clone + Debug> {
    /// Where a tensor's elements actually live for this backend.
    ///
    /// A CPU backend stores them in host memory (`Vec<T>`); an accelerator
    /// stores them in device memory and never hands out a host slice. Tensors
    /// are therefore opaque over their storage, and reaching the elements from
    /// the host always goes through [`Backend::to_host`].
    type Storage: Clone + Debug;

    /// Upload host data to the backend.
    fn to_device(&self, data: &[T]) -> Self::Storage;

    /// Download backend data back to host memory.
    fn to_host(&self, storage: &Self::Storage) -> Vec<T>;

    /// Allocate uninitialized storage for a tensor of the given shape.
    fn allocate(&self, shape: &[usize]) -> Self::Storage;

    /// Allocate storage filled with zeros.
    fn zeros(&self, shape: &[usize]) -> Self::Storage;

    /// Allocate storage filled with ones.
    fn ones(&self, shape: &[usize]) -> Self::Storage;

    /// Add two buffers element-wise.
    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Subtract two buffers element-wise.
    fn sub(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Multiply two buffers element-wise.
    fn mul(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Divide two buffers element-wise.
    fn div(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Multiply two rank-2 buffers.
    fn matmul(
        &self,
        a: &Self::Storage,
        a_shape: &[usize],
        b: &Self::Storage,
        b_shape: &[usize],
    ) -> Self::Storage;
}
