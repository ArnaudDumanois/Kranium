use super::traits::Backend;

pub struct CpuBackend;

impl<T> Backend<T> for CpuBackend
where
    T: Clone + std::ops::Add<Output = T> + std::ops::Sub<Output = T> +
    std::ops::Mul<Output = T> + std::ops::Div<Output = T> +
    std::default::Default + std::ops::AddAssign + From<u8> + Copy + std::fmt::Debug
{
    fn allocate(&self, shape: &[usize]) -> Vec<T> {
        let size = shape.iter().product();
        Vec::with_capacity(size)
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
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| a_val + b_val)
            .collect()
    }

    fn sub(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(a.len(), b.len(), "Tensors must have the same length for subtraction");
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| a_val - b_val)
            .collect()
    }

    fn mul(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(a.len(), b.len(), "Tensors must have the same length for multiplication");
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| a_val * b_val)
            .collect()
    }

    fn div(&self, a: &[T], b: &[T]) -> Vec<T> {
        assert_eq!(a.len(), b.len(), "Tensors must have the same length for division");
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| a_val / b_val)
            .collect()
    }

    fn matmul(&self, a: &[T], a_shape: &[usize], b: &[T], b_shape: &[usize]) -> Vec<T> {
        assert_eq!(a_shape.len(), 2, "First tensor must be 2D for matrix multiplication");
        assert_eq!(b_shape.len(), 2, "Second tensor must be 2D for matrix multiplication");
        assert_eq!(a_shape[1], b_shape[0], "Inner dimensions must match for matrix multiplication");

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let mut result = vec![T::default(); m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    // a[i][l] * b[l][j]
                    let a_idx = i * k + l;
                    let b_idx = l * n + j;
                    sum += a[a_idx] * b[b_idx];
                }
                result[i * n + j] = sum;
            }
        }

        result
    }
}