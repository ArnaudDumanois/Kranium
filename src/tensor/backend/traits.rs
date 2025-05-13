use std::fmt::Debug;

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