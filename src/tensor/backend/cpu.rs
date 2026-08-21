use super::traits::{Backend, Numeric};
use rayon::prelude::*;
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuBackend;

impl<T: Numeric> Backend<T> for CpuBackend {
    fn allocate(&self, shape: &[usize]) -> Vec<T> {
        let size = shape.iter().product();
        vec![T::default(); size]
    }

    fn zeros(&self, shape: &[usize]) -> Vec<T> {
        let size = shape.iter().product();
        vec![T::default(); size]
    }

    fn ones(&self, shape: &[usize]) -> Vec<T> {
        let size = shape.iter().product();
        vec![T::from(1u8); size]
    }

    fn add(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(
            a.len(),
            b.len(),
            "tensors must have the same length for addition"
        );
        a.iter().zip(b.iter()).map(|(&a, &b)| a + b).collect()
    }

    fn sub(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(
            a.len(),
            b.len(),
            "tensors must have the same length for subtraction"
        );
        a.iter().zip(b.iter()).map(|(&a, &b)| a - b).collect()
    }

    fn mul(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(
            a.len(),
            b.len(),
            "tensors must have the same length for multiplication"
        );
        a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect()
    }

    fn div(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(
            a.len(),
            b.len(),
            "tensors must have the same length for division"
        );
        a.iter().zip(b.iter()).map(|(&a, &b)| a / b).collect()
    }

    fn matmul(&self, a: &[T], a_shape: &[usize], b: &[T], b_shape: &[usize]) -> Vec<T> {
        assert_eq!(
            a_shape.len(),
            2,
            "First tensor must be 2D for matrix multiplication"
        );
        assert_eq!(
            b_shape.len(),
            2,
            "Second tensor must be 2D for matrix multiplication"
        );
        assert_eq!(
            a_shape[1], b_shape[0],
            "Inner dimensions must match for matrix multiplication"
        );

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let result: Vec<T> = (0..m * n)
            .into_par_iter()
            .map(|index| {
                let i = index / n;
                let j = index % n;
                let mut sum = T::default();
                for l in 0..k {
                    let a_idx = i * k + l;
                    let b_idx = l * n + j;
                    sum += a[a_idx] * b[b_idx];
                }
                sum
            })
            .collect();

        result
    }
}
