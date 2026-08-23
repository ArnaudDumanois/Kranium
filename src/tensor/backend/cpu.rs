use super::traits::{Backend, Numeric};

#[cfg(feature = "cpu")]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuBackend;

impl<T: Numeric> Backend<T> for CpuBackend {
    /// The CPU computes directly on host memory, so uploads and downloads are
    /// plain copies.
    type Storage = Vec<T>;

    fn to_device(&self, data: &[T]) -> Self::Storage {
        data.to_vec()
    }

    fn to_host(&self, storage: &Self::Storage) -> Vec<T> {
        storage.clone()
    }

    fn allocate(&self, shape: &[usize]) -> Self::Storage {
        vec![T::default(); shape.iter().product()]
    }

    fn zeros(&self, shape: &[usize]) -> Self::Storage {
        vec![T::default(); shape.iter().product()]
    }

    fn ones(&self, shape: &[usize]) -> Self::Storage {
        vec![T::from(1u8); shape.iter().product()]
    }

    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        assert_eq!(
            a.len(),
            b.len(),
            "tensors must have the same length for addition"
        );
        a.iter().zip(b.iter()).map(|(&a, &b)| a + b).collect()
    }

    fn sub(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        assert_eq!(
            a.len(),
            b.len(),
            "tensors must have the same length for subtraction"
        );
        a.iter().zip(b.iter()).map(|(&a, &b)| a - b).collect()
    }

    fn mul(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        assert_eq!(
            a.len(),
            b.len(),
            "tensors must have the same length for multiplication"
        );
        a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect()
    }

    fn div(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        assert_eq!(
            a.len(),
            b.len(),
            "tensors must have the same length for division"
        );
        a.iter().zip(b.iter()).map(|(&a, &b)| a / b).collect()
    }

    fn matmul(
        &self,
        a: &Self::Storage,
        a_shape: &[usize],
        b: &Self::Storage,
        b_shape: &[usize],
    ) -> Self::Storage {
        assert_eq!(
            a_shape.len(),
            2,
            "first tensor must be 2D for matrix multiplication"
        );
        assert_eq!(
            b_shape.len(),
            2,
            "second tensor must be 2D for matrix multiplication"
        );
        assert_eq!(
            a_shape[1], b_shape[0],
            "inner dimensions must match for matrix multiplication"
        );

        let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
        let cell = |index: usize| {
            let (i, j) = (index / n, index % n);
            let mut sum = T::default();
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            sum
        };

        #[cfg(feature = "cpu")]
        {
            (0..m * n).into_par_iter().map(cell).collect()
        }
        #[cfg(not(feature = "cpu"))]
        {
            (0..m * n).map(cell).collect()
        }
    }
}
