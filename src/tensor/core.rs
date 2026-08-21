use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::ops::{Index, IndexMut};

use super::backend::traits::{Backend, Numeric};
use super::error::{TensorError, TensorResult};

/// A generic n-dimensional tensor, laid out contiguously in row-major order.
///
/// The elements live wherever the backend puts them, which is not necessarily
/// host memory: reaching them from the host always goes through
/// [`Tensor::to_host`]. Backends that do store their elements in host memory
/// additionally expose the borrowing accessors ([`Tensor::data`],
/// [`Tensor::get`], indexing), which cannot exist for a device buffer.
///
/// Every fallible operation comes in two flavours: a `try_*` variant returning
/// a [`TensorError`], and a panicking wrapper for ergonomic use.
pub struct Tensor<T: Numeric, B: Backend<T> + Clone> {
    storage: B::Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    backend: B,
}

impl<T: Numeric, B: Backend<T> + Clone> Tensor<T, B> {
    /// Create a tensor filled with the default value of `T`.
    pub fn new(shape: &[usize], backend: B) -> Self {
        Self::try_new(shape, backend).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_new(shape: &[usize], backend: B) -> TensorResult<Self> {
        validate_shape(shape)?;
        let storage = backend.allocate(shape);
        Ok(Self::from_parts(storage, shape, backend))
    }

    pub fn zeros(shape: &[usize], backend: B) -> Self {
        Self::try_zeros(shape, backend).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_zeros(shape: &[usize], backend: B) -> TensorResult<Self> {
        validate_shape(shape)?;
        let storage = backend.zeros(shape);
        Ok(Self::from_parts(storage, shape, backend))
    }

    pub fn ones(shape: &[usize], backend: B) -> Self {
        Self::try_ones(shape, backend).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_ones(shape: &[usize], backend: B) -> TensorResult<Self> {
        validate_shape(shape)?;
        let storage = backend.ones(shape);
        Ok(Self::from_parts(storage, shape, backend))
    }

    /// Upload host data to the backend and wrap it as a tensor of `shape`.
    pub fn from_data(data: Vec<T>, shape: &[usize], backend: B) -> Self {
        Self::try_from_data(data, shape, backend).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_data(data: Vec<T>, shape: &[usize], backend: B) -> TensorResult<Self> {
        validate_shape(shape)?;
        let expected = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                actual: data.len(),
                expected,
            });
        }
        let storage = backend.to_device(&data);
        Ok(Self::from_parts(storage, shape, backend))
    }

    pub(super) fn from_parts(storage: B::Storage, shape: &[usize], backend: B) -> Self {
        Self {
            storage,
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            backend,
        }
    }

    /// Download the elements into host memory.
    ///
    /// This is a copy, and a device transfer for backends that do not compute
    /// on the host.
    pub fn to_host(&self) -> Vec<T> {
        self.backend.to_host(&self.storage)
    }

    pub(super) fn storage(&self) -> &B::Storage {
        &self.storage
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Total number of elements, derived from the shape rather than from the
    /// storage, so that it never triggers a device transfer.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Return a copy of the tensor laid out with `new_shape`.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        self.try_reshape(new_shape)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_reshape(&self, new_shape: &[usize]) -> TensorResult<Self> {
        validate_shape(new_shape)?;
        let requested = new_shape.iter().product();
        if self.size() != requested {
            return Err(TensorError::ReshapeSizeMismatch {
                actual: self.size(),
                requested,
            });
        }
        Ok(Self::from_parts(
            self.storage.clone(),
            new_shape,
            self.backend.clone(),
        ))
    }

    /// Translate n-dimensional indices into an offset in the flat buffer.
    pub(crate) fn try_flat_index(&self, indices: &[usize]) -> TensorResult<usize> {
        if indices.len() != self.ndim() {
            return Err(TensorError::RankMismatch {
                actual: indices.len(),
                expected: self.ndim(),
            });
        }

        let mut flat = 0;
        for (axis, (&index, (&size, &stride))) in indices
            .iter()
            .zip(self.shape.iter().zip(self.strides.iter()))
            .enumerate()
        {
            if index >= size {
                return Err(TensorError::IndexOutOfBounds { axis, index, size });
            }
            flat += index * stride;
        }

        Ok(flat)
    }
}

/// Accessors that borrow the elements in place.
///
/// They only exist for backends whose storage *is* host memory: a device buffer
/// cannot hand out a `&[T]` without a transfer, so those backends expose
/// [`Tensor::to_host`] instead.
impl<T: Numeric, B: Backend<T, Storage = Vec<T>> + Clone> Tensor<T, B> {
    pub fn data(&self) -> &[T] {
        &self.storage
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.storage
    }

    pub fn get(&self, indices: &[usize]) -> T {
        self.try_get(indices).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_get(&self, indices: &[usize]) -> TensorResult<T> {
        Ok(self.storage[self.try_flat_index(indices)?])
    }

    pub fn set(&mut self, indices: &[usize], value: T) {
        self.try_set(indices, value)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_set(&mut self, indices: &[usize], value: T) -> TensorResult<()> {
        let flat = self.try_flat_index(indices)?;
        self.storage[flat] = value;
        Ok(())
    }
}

fn validate_shape(shape: &[usize]) -> TensorResult<()> {
    if shape.is_empty() {
        return Err(TensorError::EmptyShape);
    }
    match shape.iter().position(|&dim| dim == 0) {
        Some(axis) => Err(TensorError::ZeroDimension { axis }),
        None => Ok(()),
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl<T: Numeric, B: Backend<T, Storage = Vec<T>> + Clone> Index<&[usize]> for Tensor<T, B> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        let flat = self
            .try_flat_index(indices)
            .unwrap_or_else(|e| panic!("{e}"));
        &self.storage[flat]
    }
}

impl<T: Numeric, B: Backend<T, Storage = Vec<T>> + Clone> IndexMut<&[usize]> for Tensor<T, B> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let flat = self
            .try_flat_index(indices)
            .unwrap_or_else(|e| panic!("{e}"));
        &mut self.storage[flat]
    }
}

impl<T: Numeric, B: Backend<T> + Clone> Clone for Tensor<T, B> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            backend: self.backend.clone(),
        }
    }
}

impl<T: Numeric + PartialEq, B: Backend<T, Storage = Vec<T>> + Clone> PartialEq for Tensor<T, B> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.storage == other.storage
    }
}

impl<T: Numeric, B: Backend<T> + Clone> Debug for Tensor<T, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("storage", &self.storage)
            .finish()
    }
}

impl<T: Numeric + Display, B: Backend<T> + Clone> Display for Tensor<T, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        for (i, value) in self.to_host().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{value}")?;
        }
        write!(f, "])")
    }
}
