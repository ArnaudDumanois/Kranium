use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::marker::PhantomData;

use super::backend::traits::Backend;

/// A generic n-dimensional tensor structure
pub struct Tensor<T, B: Backend<T> + Clone>
where
    T: Clone + Debug + Copy
{
    /// The underlying data of the tensor
    data: Vec<T>,

    /// The shape of the tensor (dimensions)
    shape: Vec<usize>,

    /// The strides of the tensor for indexing
    strides: Vec<usize>,

    /// The backend used for tensor operations
    backend: B,

    /// Phantom data for type parameter T
    _marker: PhantomData<T>,
}

impl<T, B: Backend<T> + Clone> Tensor<T, B>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> +
    Default + AddAssign + From<u8> + Copy + Debug
{
    /// Create a new tensor with the given shape and backend
    pub fn new(shape: &[usize], backend: B) -> Self {
        let data = backend.allocate(shape);
        let strides = Self::compute_strides(shape);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
            backend,
            _marker: PhantomData,
        }
    }

    /// Create a new tensor filled with zeros
    pub fn zeros(shape: &[usize], backend: B) -> Self {
        let data = backend.zeros(shape);
        let strides = Self::compute_strides(shape);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
            backend,
            _marker: PhantomData,
        }
    }

    /// Create a new tensor filled with ones
    pub fn ones(shape: &[usize], backend: B) -> Self {
        let data = backend.ones(shape);
        let strides = Self::compute_strides(shape);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
            backend,
            _marker: PhantomData,
        }
    }

    /// Create a tensor from existing data
    pub fn from_data(data: Vec<T>, shape: &[usize], backend: B) -> Self {
        let expected_size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_size,
            "Data length {} doesn't match expected size {} from shape {:?}",
            data.len(), expected_size, shape
        );

        let strides = Self::compute_strides(shape);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
            backend,
            _marker: PhantomData,
        }
    }

    /// Compute strides for the given shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len()-1).rev() {
            strides[i] = strides[i+1] * shape[i+1];
        }
        strides
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the tensor
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of elements in the tensor
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get the number of dimensions of the tensor
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Reshape the tensor to a new shape
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            self.size(),
            new_size,
            "Cannot reshape tensor of size {} to shape {:?} with size {}",
            self.size(), new_shape, new_size
        );

        Self::from_data(self.data.clone(), new_shape, self.backend.clone())
    }

    /// Get the flat index from n-dimensional indices
    fn get_flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.ndim(),
            "Number of indices {} must match tensor dimensions {}",
            indices.len(), self.ndim()
        );

        // Check bounds
        for (i, &idx) in indices.iter().enumerate() {
            assert!(
                idx < self.shape[i],
                "Index {} out of bounds for dimension {} with size {}",
                idx, i, self.shape[i]
            );
        }

        // Calculate flat index using strides
        let mut flat_idx = 0;
        for i in 0..indices.len() {
            flat_idx += indices[i] * self.strides[i];
        }

        flat_idx
    }

    /// Element-wise addition of two tensors
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Tensor shapes must match for addition: {:?} vs {:?}",
            self.shape, other.shape
        );

        let result_data = self.backend.add(&self.data, &other.data);

        Self::from_data(result_data, &self.shape, self.backend.clone())
    }

    /// Element-wise subtraction of two tensors
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Tensor shapes must match for subtraction: {:?} vs {:?}",
            self.shape, other.shape
        );

        let result_data = self.backend.sub(&self.data, &other.data);

        Self::from_data(result_data, &self.shape, self.backend.clone())
    }

    /// Element-wise multiplication of two tensors
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Tensor shapes must match for multiplication: {:?} vs {:?}",
            self.shape, other.shape
        );

        let result_data = self.backend.mul(&self.data, &other.data);

        Self::from_data(result_data, &self.shape, self.backend.clone())
    }

    /// Element-wise division of two tensors
    pub fn div(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Tensor shapes must match for division: {:?} vs {:?}",
            self.shape, other.shape
        );

        let result_data = self.backend.div(&self.data, &other.data);

        Self::from_data(result_data, &self.shape, self.backend.clone())
    }

    /// Matrix multiplication of two tensors
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.ndim(), 2,
            "First tensor must be 2D for matrix multiplication, got {:?}",
            self.shape
        );
        assert_eq!(
            other.ndim(), 2,
            "Second tensor must be 2D for matrix multiplication, got {:?}",
            other.shape
        );
        assert_eq!(
            self.shape[1], other.shape[0],
            "Inner dimensions must match for matrix multiplication: {} vs {}",
            self.shape[1], other.shape[0]
        );

        let result_shape = vec![self.shape[0], other.shape[1]];
        let result_data = self.backend.matmul(
            &self.data,
            &self.shape,
            &other.data,
            &other.shape
        );

        Self::from_data(result_data, &result_shape, self.backend.clone())
    }

    /// Get a value at the specified indices
    pub fn get(&self, indices: &[usize]) -> T {
        let idx = self.get_flat_index(indices);
        self.data[idx]
    }

    /// Set a value at the specified indices
    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = self.get_flat_index(indices);
        self.data[idx] = value;
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self) -> Self {
        assert_eq!(
            self.ndim(), 2,
            "Transpose is only implemented for 2D tensors, got shape {:?}",
            self.shape
        );

        let new_shape = vec![self.shape[1], self.shape[0]];
        let mut result_data = Vec::with_capacity(self.data.len());

        // Remplir directement avec les données transposées
        for j in 0..self.shape[1] {
            for i in 0..self.shape[0] {
                result_data.push(self.data[i * self.shape[1] + j]);
            }
        }

        Self::from_data(result_data, &new_shape, self.backend.clone())
    }
}

// Implement operator overloading for Tensor
impl<T, B: Backend<T> + Clone> Add for &Tensor<T, B>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> +
    Default + AddAssign + From<u8> + Copy + Debug
{
    type Output = Tensor<T, B>;

    fn add(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl<T, B: Backend<T> + Clone> Sub for &Tensor<T, B>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> +
    Default + AddAssign + From<u8> + Copy + Debug
{
    type Output = Tensor<T, B>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(other)
    }
}

impl<T, B: Backend<T> + Clone> Mul for &Tensor<T, B>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> +
    Default + AddAssign + From<u8> + Copy + Debug
{
    type Output = Tensor<T, B>;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(other)
    }
}

impl<T, B: Backend<T> + Clone> Div for &Tensor<T, B>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> +
    Default + AddAssign + From<u8> + Copy + Debug
{
    type Output = Tensor<T, B>;

    fn div(self, other: Self) -> Self::Output {
        self.div(other)
    }
}

// Implement Clone for Tensor if Backend is Clone
impl<T, B: Backend<T> + Clone> Clone for Tensor<T, B>
where
    T: Clone + Debug + Copy
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            backend: self.backend.clone(),
            _marker: PhantomData,
        }
    }
}
