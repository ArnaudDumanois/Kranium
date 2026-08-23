use std::ops::{Add, Div, Mul, Sub};

use super::backend::traits::{Backend, Numeric};
use super::core::Tensor;
use super::error::{TensorError, TensorResult};

impl<T: Numeric, B: Backend<T> + Clone> Tensor<T, B> {
    pub fn add(&self, other: &Self) -> Self {
        self.try_add(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_add(&self, other: &Self) -> TensorResult<Self> {
        self.zip_with(other, "addition", |backend, a, b| backend.add(a, b))
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.try_sub(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_sub(&self, other: &Self) -> TensorResult<Self> {
        self.zip_with(other, "subtraction", |backend, a, b| backend.sub(a, b))
    }

    pub fn mul(&self, other: &Self) -> Self {
        self.try_mul(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_mul(&self, other: &Self) -> TensorResult<Self> {
        self.zip_with(other, "multiplication", |backend, a, b| backend.mul(a, b))
    }

    /// Element-wise division. Dividing integer tensors by zero panics, as it
    /// does for the underlying primitive type.
    pub fn div(&self, other: &Self) -> Self {
        self.try_div(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_div(&self, other: &Self) -> TensorResult<Self> {
        self.zip_with(other, "division", |backend, a, b| backend.div(a, b))
    }

    /// Apply a backend element-wise operation once both shapes are known to match.
    fn zip_with<F>(&self, other: &Self, op: &'static str, f: F) -> TensorResult<Self>
    where
        F: Fn(&B, &B::Storage, &B::Storage) -> B::Storage,
    {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                op,
                lhs: self.shape().to_vec(),
                rhs: other.shape().to_vec(),
            });
        }

        let storage = f(self.backend(), self.storage(), other.storage());
        Ok(Self::from_parts(
            storage,
            self.shape(),
            self.backend().clone(),
        ))
    }

    /// Matrix multiplication of two 2-D tensors.
    pub fn matmul(&self, other: &Self) -> Self {
        self.try_matmul(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_matmul(&self, other: &Self) -> TensorResult<Self> {
        self.expect_rank_2("matrix multiplication")?;
        other.expect_rank_2("matrix multiplication")?;

        let (lhs_cols, rhs_rows) = (self.shape()[1], other.shape()[0]);
        if lhs_cols != rhs_rows {
            return Err(TensorError::MatmulInnerMismatch { lhs_cols, rhs_rows });
        }

        let shape = [self.shape()[0], other.shape()[1]];
        let storage =
            self.backend()
                .matmul(self.storage(), self.shape(), other.storage(), other.shape());

        Ok(Self::from_parts(storage, &shape, self.backend().clone()))
    }

    /// Transpose a 2-D tensor.
    pub fn transpose(&self) -> Self {
        self.try_transpose().unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_transpose(&self) -> TensorResult<Self> {
        self.expect_rank_2("transpose")?;

        // La transposition n'a pas encore de primitive dans le backend : on
        // rapatrie, on permute, puis on renvoie sur le device. C'est correct
        // pour tout backend, et ce sera le premier candidat a une primitive
        // dediee quand un vrai kernel GPU arrivera.
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let host = self.to_host();
        let mut data = Vec::with_capacity(self.size());
        for j in 0..cols {
            for i in 0..rows {
                data.push(host[i * cols + j]);
            }
        }

        Ok(Self::from_parts(
            self.backend().to_device(&data),
            &[cols, rows],
            self.backend().clone(),
        ))
    }

    fn expect_rank_2(&self, op: &'static str) -> TensorResult<()> {
        if self.ndim() != 2 {
            return Err(TensorError::ExpectedRank {
                op,
                expected: 2,
                actual: self.ndim(),
            });
        }
        Ok(())
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
