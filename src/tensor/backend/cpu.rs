use super::traits::{Backend, Numeric};
use rayon::prelude::*;
pub struct CpuBackend;

impl<T: Numeric> Backend<T> for CpuBackend
{
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
        assert_eq!(a.len(), b.len(), "Tensors must have the same length for addition");
        let mut result = vec![T::default(); a.len()];
        result.iter_mut()
            .zip(a.iter().zip(b.iter()))
            .for_each(|(res, (&a_val, &b_val))| *res = a_val + b_val);
        result
    }

    fn sub(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(a.len(), b.len(), "Tensors must have the same length for subtraction");
        let mut result = vec![T::default(); a.len()];
        result.iter_mut()
            .zip(a.iter().zip(b.iter()))
            .for_each(|(res, (&a_val, &b_val))| *res = a_val - b_val);
        result
    }

    fn mul(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(a.len(), b.len(), "Tensors must have the same length for multiplication");
        let mut result = vec![T::default(); a.len()];
        result.iter_mut()
            .zip(a.iter().zip(b.iter()))
            .for_each(|(res, (&a_val, &b_val))| *res = a_val * b_val);
        result
    }

    fn div(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(a.len(), b.len(), "Tensors must have the same length for division");
        let mut result = vec![T::default(); a.len()];
        result.iter_mut()
            .zip(a.iter().zip(b.iter()))
            .for_each(|(res, (&a_val, &b_val))| *res = a_val / b_val);
        result
    }

    fn matmul(&self, a: &[T], a_shape: &[usize], b: &[T], b_shape: &[usize]) -> Vec<T> {
        assert_eq!(a_shape.len(), 2, "First tensor must be 2D for matrix multiplication");
        assert_eq!(b_shape.len(), 2, "Second tensor must be 2D for matrix multiplication");
        assert_eq!(a_shape[1], b_shape[0], "Inner dimensions must match for matrix multiplication");

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let result: Vec<T> = (0..m * n)
            .into_par_iter() // Utilisation de rayon pour paralléliser
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