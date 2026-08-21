use kranium::tensor::backend::cpu::CpuBackend;
use kranium::tensor::{Tensor, TensorError};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32, CpuBackend> {
    Tensor::from_data(data, shape, CpuBackend)
}

// --- Construction -----------------------------------------------------------

#[test]
fn new_allocates_default_values() {
    let t: Tensor<f32, CpuBackend> = Tensor::new(&[2, 3], CpuBackend);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.size(), 6);
    assert_eq!(t.data(), &[0.0; 6]);
}

#[test]
fn zeros_and_ones() {
    let z: Tensor<f32, CpuBackend> = Tensor::zeros(&[2, 2], CpuBackend);
    let o: Tensor<f32, CpuBackend> = Tensor::ones(&[2, 2], CpuBackend);
    assert_eq!(z.data(), &[0.0; 4]);
    assert_eq!(o.data(), &[1.0; 4]);
}

#[test]
fn from_data_keeps_data_and_shape() {
    let t = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn try_from_data_rejects_length_mismatch() {
    let err = Tensor::try_from_data(vec![1.0, 2.0, 3.0], &[2, 2], CpuBackend).unwrap_err();
    assert!(matches!(err, TensorError::DataLengthMismatch { .. }));
}

#[test]
fn try_new_rejects_empty_shape() {
    let err = Tensor::<f32, CpuBackend>::try_new(&[], CpuBackend).unwrap_err();
    assert!(matches!(err, TensorError::EmptyShape));
}

#[test]
fn try_new_rejects_zero_dimension() {
    let err = Tensor::<f32, CpuBackend>::try_new(&[2, 0], CpuBackend).unwrap_err();
    assert!(matches!(err, TensorError::ZeroDimension { axis: 1 }));
}

#[test]
#[should_panic(expected = "data length")]
fn from_data_panics_on_length_mismatch() {
    tensor(vec![1.0, 2.0, 3.0], &[2, 2]);
}

// --- Strides ----------------------------------------------------------------

#[test]
fn strides_are_row_major() {
    let t = tensor(vec![0.0; 24], &[2, 3, 4]);
    assert_eq!(t.strides(), &[12, 4, 1]);
}

#[test]
fn strides_of_1d_tensor() {
    let t = tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert_eq!(t.strides(), &[1]);
}

// --- Indexing ---------------------------------------------------------------

#[test]
fn get_and_set_by_indices() {
    let mut t = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[1, 2]), 6.0);
    t.set(&[1, 0], 42.0);
    assert_eq!(t.get(&[1, 0]), 42.0);
}

#[test]
fn try_get_reports_rank_mismatch() {
    let t = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(matches!(
        t.try_get(&[0]).unwrap_err(),
        TensorError::RankMismatch { .. }
    ));
}

#[test]
fn try_get_reports_out_of_bounds() {
    let t = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(matches!(
        t.try_get(&[2, 0]).unwrap_err(),
        TensorError::IndexOutOfBounds { axis: 0, .. }
    ));
}

#[test]
#[should_panic(expected = "out of bounds")]
fn get_panics_out_of_bounds() {
    let t = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    t.get(&[0, 5]);
}

#[test]
fn index_operator_reads_elements() {
    let mut t = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(t[&[1, 1][..]], 4.0);
    t[&[0, 1][..]] = 9.0;
    assert_eq!(t.data(), &[1.0, 9.0, 3.0, 4.0]);
}

// --- Reshape ----------------------------------------------------------------

#[test]
fn reshape_preserves_data() {
    let t = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = t.reshape(&[3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.strides(), &[2, 1]);
    assert_eq!(r.data(), t.data());
    // l'original reste intact (immutabilité)
    assert_eq!(t.shape(), &[2, 3]);
}

#[test]
fn try_reshape_rejects_incompatible_size() {
    let t = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(matches!(
        t.try_reshape(&[3, 3]).unwrap_err(),
        TensorError::ReshapeSizeMismatch { .. }
    ));
}

// --- Clone / Display --------------------------------------------------------

#[test]
fn clone_is_independent() {
    let t = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mut c = t.clone();
    c.set(&[0, 0], 99.0);
    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(c.get(&[0, 0]), 99.0);
}

#[test]
fn display_shows_shape_and_data() {
    let t = tensor(vec![1.0, 2.0], &[2]);
    let s = format!("{t}");
    assert!(s.contains("shape=[2]"), "got: {s}");
    assert!(s.contains('1'), "got: {s}");
}

#[test]
fn debug_shows_shape_and_strides() {
    let t = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let s = format!("{t:?}");
    assert!(s.contains("shape"), "got: {s}");
    assert!(s.contains("strides"), "got: {s}");
}
