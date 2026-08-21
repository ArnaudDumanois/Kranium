use kranium::tensor::backend::cpu::CpuBackend;
use kranium::tensor::{Tensor, TensorError};

type T2 = Tensor<f32, CpuBackend>;

fn t(data: Vec<f32>, shape: &[usize]) -> T2 {
    Tensor::from_data(data, shape, CpuBackend)
}

// --- Element-wise: cas nominaux ---------------------------------------------

#[test]
fn add_elementwise_1d() {
    let r = t(vec![1.0, 2.0, 3.0], &[3]).add(&t(vec![4.0, 5.0, 6.0], &[3]));
    assert_eq!(r.data(), &[5.0, 7.0, 9.0]);
    assert_eq!(r.shape(), &[3]);
}

#[test]
fn sub_elementwise_2d() {
    let r = t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).sub(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    assert_eq!(r.data(), &[4.0, 4.0, 4.0, 4.0]);
    assert_eq!(r.shape(), &[2, 2]);
}

#[test]
fn mul_elementwise_3d() {
    let a = t((1..=8).map(|v| v as f32).collect(), &[2, 2, 2]);
    let r = a.mul(&t(vec![2.0; 8], &[2, 2, 2]));
    assert_eq!(r.data(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    assert_eq!(r.shape(), &[2, 2, 2]);
}

#[test]
fn div_elementwise() {
    let r = t(vec![8.0, 9.0, 10.0], &[3]).div(&t(vec![2.0, 3.0, 5.0], &[3]));
    assert_eq!(r.data(), &[4.0, 3.0, 2.0]);
}

#[test]
fn operands_are_left_untouched() {
    let a = t(vec![1.0, 2.0], &[2]);
    let b = t(vec![3.0, 4.0], &[2]);
    let _ = a.add(&b);
    assert_eq!(a.data(), &[1.0, 2.0]);
    assert_eq!(b.data(), &[3.0, 4.0]);
}

// --- Element-wise: erreurs de shape -----------------------------------------

#[test]
fn try_add_reports_shape_mismatch() {
    let err = t(vec![1.0, 2.0], &[2])
        .try_add(&t(vec![1.0, 2.0, 3.0], &[3]))
        .unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { op: "addition", .. }),
        "got {err:?}"
    );
}

#[test]
fn try_ops_reject_same_size_different_shape() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    assert!(a.try_sub(&b).is_err());
    assert!(a.try_mul(&b).is_err());
    assert!(a.try_div(&b).is_err());
}

#[test]
#[should_panic(expected = "shapes must match for multiplication")]
fn mul_panics_on_shape_mismatch() {
    let _ = t(vec![1.0, 2.0], &[2]).mul(&t(vec![1.0], &[1]));
}

// --- Opérateurs --------------------------------------------------------------

#[test]
fn operators_match_methods() {
    let a = t(vec![6.0, 8.0], &[2]);
    let b = t(vec![3.0, 2.0], &[2]);
    assert_eq!((&a + &b).data(), a.add(&b).data());
    assert_eq!((&a - &b).data(), a.sub(&b).data());
    assert_eq!((&a * &b).data(), a.mul(&b).data());
    assert_eq!((&a / &b).data(), a.div(&b).data());
}

#[test]
fn operations_can_be_chained() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let r = (&a + &b).transpose();
    assert_eq!(r.data(), &[2.0, 4.0, 3.0, 5.0]);
    assert_eq!(r.shape(), &[2, 2]);
}

// --- Transpose ---------------------------------------------------------------

#[test]
fn transpose_rectangular() {
    let r = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).transpose();
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(r.strides(), &[2, 1]);
}

#[test]
fn transpose_is_an_involution() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(a.transpose().transpose(), a);
}

#[test]
fn try_transpose_rejects_non_2d() {
    let err = t(vec![1.0, 2.0, 3.0], &[3]).try_transpose().unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::ExpectedRank {
                op: "transpose",
                expected: 2,
                actual: 1
            }
        ),
        "got {err:?}"
    );
}

#[test]
#[should_panic(expected = "transpose expects a 2-dimensional tensor")]
fn transpose_panics_on_3d() {
    let _ = t(vec![0.0; 8], &[2, 2, 2]).transpose();
}

// --- Matmul ------------------------------------------------------------------

#[test]
fn matmul_rectangular() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let r = a.matmul(&b);
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(r.data(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_with_identity_is_neutral() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let identity = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert_eq!(a.matmul(&identity), a);
}

#[test]
fn try_matmul_reports_inner_mismatch() {
    let err = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
        .try_matmul(&t(vec![1.0, 2.0, 3.0], &[3, 1]))
        .unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::MatmulInnerMismatch {
                lhs_cols: 2,
                rhs_rows: 3
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn try_matmul_rejects_non_2d_operands() {
    let a = t(vec![1.0, 2.0], &[2]);
    let b = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(a.try_matmul(&b).is_err());
    assert!(b.try_matmul(&a).is_err());
}

// --- Égalité -----------------------------------------------------------------

#[test]
fn equality_compares_shape_and_data() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(a, t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    assert_ne!(a, t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    assert_ne!(a, t(vec![1.0, 2.0, 3.0, 9.0], &[2, 2]));
}

// --- Types entiers -----------------------------------------------------------

#[test]
fn works_with_integer_tensors() {
    let a: Tensor<i32, CpuBackend> = Tensor::from_data(vec![1, 2, 3], &[3], CpuBackend);
    let b: Tensor<i32, CpuBackend> = Tensor::from_data(vec![10, 20, 30], &[3], CpuBackend);
    assert_eq!(a.add(&b).data(), &[11, 22, 33]);
}
