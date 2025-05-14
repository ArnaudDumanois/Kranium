use kranium::tensor::backend::cpu::CpuBackend;
use kranium::tensor::backend::traits::Backend;

#[test]
fn test_allocate() {
    let backend = CpuBackend;
    let shape = [2, 3];
    let buffer: Vec<f32> = backend.allocate(&shape);
    assert_eq!(buffer.len(), 6);
}

#[test]
fn test_zeros() {
    let backend = CpuBackend;
    let shape = [2, 2];
    let zeros: Vec<f32> = backend.zeros(&shape);
    assert_eq!(zeros, vec![0.0; 4]);
}

#[test]
fn test_ones() {
    let backend = CpuBackend;
    let shape = [3];
    let ones: Vec<f32> = backend.ones(&shape);
    assert_eq!(ones, vec![1.0; 3]);
}

#[test]
fn test_add() {
    let backend = CpuBackend;
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = backend.add(&a, &b);
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sub() {
    let backend = CpuBackend;
    let a = vec![5.0, 6.0, 7.0];
    let b = vec![2.0, 1.0, 3.0];
    let result = backend.sub(&a, &b);
    assert_eq!(result, vec![3.0, 5.0, 4.0]);
}

#[test]
fn test_mul() {
    let backend = CpuBackend;
    let a = vec![2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0];
    let result = backend.mul(&a, &b);
    assert_eq!(result, vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_div() {
    let backend = CpuBackend;
    let a = vec![8.0, 9.0, 10.0];
    let b = vec![2.0, 3.0, 5.0];
    let result = backend.div(&a, &b);
    assert_eq!(result, vec![4.0, 3.0, 2.0]);
}

#[test]
fn test_matmul() {
    let backend = CpuBackend;
    // Matrix A: 2x3
    let a = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    ];
    // Matrix B: 3x2
    let b = vec![
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    ];
    let result = backend.matmul(&a, &[2, 3], &b, &[3, 2]);
    let expected = vec![
        58.0, 64.0,
        139.0, 154.0
    ];
    assert_eq!(result, expected);
}