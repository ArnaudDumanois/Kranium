use std::ops::{Add, Div, Mul, Sub};

use super::backend::traits::{Backend, Numeric};
use super::core::Tensor;

impl<T: Numeric, B: Backend<T> + Clone> Tensor<T, B> {
    pub fn add(&self, other: &Self) -> Self {
        self.zip_with(other, "addition", |backend, a, b| backend.add(a, b))
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.zip_with(other, "subtraction", |backend, a, b| backend.sub(a, b))
    }

    pub fn mul(&self, other: &Self) -> Self {
        self.zip_with(other, "multiplication", |backend, a, b| backend.mul(a, b))
    }

    pub fn div(&self, other: &Self) -> Self {
        self.zip_with(other, "division", |backend, a, b| backend.div(a, b))
    }

    /// Apply a backend element-wise operation after checking shape compatibility.
    fn zip_with<F>(&self, other: &Self, op: &str, f: F) -> Self
    where
        F: Fn(&B, &[T], &[T]) -> Vec<T>,
    {
        assert_eq!(
            self.shape(),
            other.shape(),
            "tensor shapes must match for {op}: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );

        let data = f(self.backend(), self.data(), other.data());
        Self::from_data(data, self.shape(), self.backend().clone())
    }

    /// Matrix multiplication of two 2-D tensors.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.ndim(),
            2,
            "first tensor must be 2D for matrix multiplication, got {:?}",
            self.shape()
        );
        assert_eq!(
            other.ndim(),
            2,
            "second tensor must be 2D for matrix multiplication, got {:?}",
            other.shape()
        );
        assert_eq!(
            self.shape()[1],
            other.shape()[0],
            "inner dimensions must match for matrix multiplication: {} vs {}",
            self.shape()[1],
            other.shape()[0]
        );

        let result_shape = [self.shape()[0], other.shape()[1]];
        let data = self
            .backend()
            .matmul(self.data(), self.shape(), other.data(), other.shape());

        Self::from_data(data, &result_shape, self.backend().clone())
    }

    /// Transpose a 2-D tensor.
    pub fn transpose(&self) -> Self {
        assert_eq!(
            self.ndim(),
            2,
            "transpose is only implemented for 2D tensors, got shape {:?}",
            self.shape()
        );

        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let mut data = Vec::with_capacity(self.size());
        for j in 0..cols {
            for i in 0..rows {
                data.push(self.data()[i * cols + j]);
            }
        }

        Self::from_data(data, &[cols, rows], self.backend().clone())
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident) => {
        impl<T: Numeric, B: Backend<T> + Clone> $trait for &Tensor<T, B> {
            type Output = Tensor<T, B>;

            fn $method(self, other: Self) -> Self::Output {
                Tensor::$method(self, other)
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);
