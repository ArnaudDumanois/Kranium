use kranium::tensor::TensorError;

fn message(error: TensorError) -> String {
    error.to_string()
}

#[test]
fn every_variant_renders_a_readable_message() {
    assert!(message(TensorError::EmptyShape).contains("at least one dimension"));

    assert!(message(TensorError::ZeroDimension { axis: 1 }).contains("dimension 1"));

    let m = message(TensorError::DataLengthMismatch {
        actual: 3,
        expected: 4,
    });
    assert!(m.contains('3') && m.contains('4'), "{m}");

    let m = message(TensorError::RankMismatch {
        actual: 1,
        expected: 2,
    });
    assert!(m.contains("2 indices"), "{m}");

    let m = message(TensorError::IndexOutOfBounds {
        axis: 0,
        index: 5,
        size: 2,
    });
    assert!(m.contains("out of bounds"), "{m}");

    let m = message(TensorError::ReshapeSizeMismatch {
        actual: 4,
        requested: 9,
    });
    assert!(m.contains("reshape"), "{m}");

    let m = message(TensorError::ShapeMismatch {
        op: "addition",
        lhs: vec![2],
        rhs: vec![3],
    });
    assert!(m.contains("addition") && m.contains("[2]"), "{m}");

    let m = message(TensorError::ExpectedRank {
        op: "transpose",
        expected: 2,
        actual: 3,
    });
    assert!(m.contains("transpose"), "{m}");

    let m = message(TensorError::MatmulInnerMismatch {
        lhs_cols: 2,
        rhs_rows: 3,
    });
    assert!(m.contains("inner dimensions"), "{m}");

    let m = message(TensorError::BackendUnavailable { requested: "gpu" });
    assert!(m.contains("gpu"), "{m}");
}

#[test]
fn errors_implement_the_standard_error_trait() {
    fn assert_is_error<E: std::error::Error>(_: &E) {}
    assert_is_error(&TensorError::EmptyShape);
}

#[test]
fn errors_can_be_compared_and_cloned() {
    let error = TensorError::ZeroDimension { axis: 2 };
    assert_eq!(error.clone(), error);
    assert_ne!(error, TensorError::EmptyShape);
}
