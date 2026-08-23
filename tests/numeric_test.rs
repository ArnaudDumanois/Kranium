use kranium::tensor::{Float, Numeric};

// --- Constructeurs numériques ------------------------------------------------

#[test]
fn zero_and_one_are_available_for_every_numeric_type() {
    assert_eq!(<f32 as Numeric>::zero(), 0.0);
    assert_eq!(<f32 as Numeric>::one(), 1.0);
    assert_eq!(<i32 as Numeric>::zero(), 0);
    assert_eq!(<i32 as Numeric>::one(), 1);
    assert_eq!(<u8 as Numeric>::one(), 1);
    assert_eq!(<f64 as Numeric>::one(), 1.0);
}

#[test]
fn from_usize_converts_small_values() {
    assert_eq!(f32::from_usize(3), 3.0);
    assert_eq!(i32::from_usize(3), 3);
}

/// Tous les types primitifs doivent être couverts, pas seulement ceux utilisés
/// par les tests de tenseurs.
#[test]
fn every_primitive_type_implements_numeric() {
    macro_rules! check {
        ($($ty:ty),*) => {
            $(
                assert_eq!(<$ty as Numeric>::zero(), 0 as $ty);
                assert_eq!(<$ty as Numeric>::one(), 1 as $ty);
                assert_eq!(<$ty>::from_usize(7), 7 as $ty);
            )*
        };
    }
    check!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
    );
}

/// La raison d'être de `from_usize` : `From<u8>` plafonnait à 255, ce qui
/// rendait impossible une moyenne sur un tenseur de plus de 255 éléments.
#[test]
fn from_usize_goes_beyond_the_u8_ceiling() {
    assert_eq!(f32::from_usize(1000), 1000.0);
    assert_eq!(f64::from_usize(100_000), 100_000.0);
    assert_eq!(i64::from_usize(70_000), 70_000);
}

#[test]
fn a_mean_over_a_thousand_elements_is_now_expressible() {
    let values: Vec<f32> = (0..1000).map(|v| v as f32).collect();
    let sum = values.iter().fold(f32::zero(), |acc, &v| acc + v);
    let mean = sum / f32::from_usize(values.len());
    assert_eq!(mean, 499.5);
}

// --- Trait Float -------------------------------------------------------------

#[test]
fn float_exposes_the_operations_needed_by_activations() {
    assert_eq!(2.0f32.abs(), 2.0);
    assert_eq!((-2.0f32).abs(), 2.0);
    assert_eq!(0.0f32.exp(), 1.0);
    assert_eq!(1.0f32.ln(), 0.0);
    assert_eq!(4.0f32.sqrt(), 2.0);
    assert_eq!(0.0f32.tanh(), 0.0);

    assert_eq!((-2.0f64).abs(), 2.0);
    assert_eq!(0.0f64.exp(), 1.0);
    assert_eq!(1.0f64.ln(), 0.0);
    assert_eq!(9.0f64.sqrt(), 3.0);
    assert_eq!(0.0f64.tanh(), 0.0);
    assert!(1.0f64.is_finite_value());
}

#[test]
fn float_exposes_comparison_helpers_needed_by_reductions() {
    assert_eq!(Float::max_of(2.0f32, 5.0), 5.0);
    assert_eq!(Float::min_of(2.0f32, 5.0), 2.0);
    assert_eq!(Float::max_of(2.0f64, 5.0), 5.0);
    assert_eq!(Float::min_of(2.0f64, 5.0), 2.0);
}

/// L'ordre doit être atteignable à travers la borne `Float`, pas seulement sur
/// le type concret.
#[test]
fn float_is_ordered_through_the_trait_bound() {
    fn largest<T: Float>(values: &[T]) -> T {
        values.iter().fold(values[0], |acc, &v| acc.max_of(v))
    }
    assert_eq!(largest(&[1.0f32, 7.0, 3.0]), 7.0);
    assert_eq!(largest(&[1.0f64, -7.0, 3.0]), 3.0);
}

#[test]
fn float_exposes_an_epsilon_for_pivot_detection() {
    assert!(f32::epsilon() > 0.0);
    assert!(f32::epsilon() < 1e-4);
    assert!(f64::epsilon() < f32::epsilon() as f64);
}

#[test]
fn floats_are_negatable() {
    assert_eq!(-(2.0f32), -2.0);
    assert_eq!(-(-2.0f64), 2.0);
}

#[test]
fn float_detects_non_finite_values() {
    assert!(1.0f32.is_finite_value());
    assert!(!(f32::INFINITY).is_finite_value());
    assert!(!(f32::NAN).is_finite_value());
}

// --- Utilisation générique ---------------------------------------------------

/// Une fonction générique bornée par `Float` doit compiler pour f32 comme f64.
fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

#[test]
fn a_generic_float_function_works_for_every_float_type() {
    assert_eq!(sigmoid(0.0f32), 0.5);
    assert_eq!(sigmoid(0.0f64), 0.5);
    assert!(sigmoid(10.0f32) > 0.99);
    assert!(sigmoid(-10.0f32) < 0.01);
}

/// À l'inverse, les entiers restent utilisables avec `Numeric` seul.
fn double<T: Numeric>(x: T) -> T {
    x + x
}

#[test]
fn integers_still_work_with_numeric_alone() {
    assert_eq!(double(21i32), 42);
    assert_eq!(double(1.5f32), 3.0);
}
