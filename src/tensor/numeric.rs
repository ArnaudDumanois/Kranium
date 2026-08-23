use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// The element types a tensor can hold.
///
/// Implemented for every primitive integer and floating point type. Operations
/// that only need arithmetic — the four element-wise operations, `matmul`,
/// `transpose` — are bounded by this trait, so integer tensors keep working.
pub trait Numeric:
    Clone
    + Copy
    + Default
    + Debug
    + PartialEq
    + Send
    + Sync
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;

    fn one() -> Self;

    /// Build a value from a count.
    ///
    /// This is what makes `mean` expressible: the previous `From<u8>` bound
    /// capped conversions at 255 elements. Integer types saturate at their own
    /// maximum, as any `as` cast would.
    fn from_usize(n: usize) -> Self;
}

/// Floating point elements.
///
/// Adds ordering, negation and the transcendental functions that activations
/// and pivoting need. Integer tensors deliberately do not get these.
pub trait Float: Numeric + PartialOrd + Neg<Output = Self> {
    fn abs(self) -> Self;

    fn exp(self) -> Self;

    fn ln(self) -> Self;

    fn sqrt(self) -> Self;

    fn tanh(self) -> Self;

    /// Smallest difference the type can represent around 1, used to decide
    /// whether a pivot is numerically zero.
    fn epsilon() -> Self;

    fn is_finite_value(self) -> bool;

    /// Named `max_of` rather than `max` so it never shadows the inherent
    /// `f32::max` at the call site.
    fn max_of(self, other: Self) -> Self {
        if self > other { self } else { other }
    }

    fn min_of(self, other: Self) -> Self {
        if self < other { self } else { other }
    }
}

macro_rules! impl_numeric {
    ($($ty:ty),*) => {
        $(
            impl Numeric for $ty {
                fn zero() -> Self {
                    0 as $ty
                }

                fn one() -> Self {
                    1 as $ty
                }

                fn from_usize(n: usize) -> Self {
                    n as $ty
                }
            }
        )*
    };
}

impl_numeric!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
);

macro_rules! impl_float {
    ($($ty:ty),*) => {
        $(
            impl Float for $ty {
                fn abs(self) -> Self {
                    <$ty>::abs(self)
                }

                fn exp(self) -> Self {
                    <$ty>::exp(self)
                }

                fn ln(self) -> Self {
                    <$ty>::ln(self)
                }

                fn sqrt(self) -> Self {
                    <$ty>::sqrt(self)
                }

                fn tanh(self) -> Self {
                    <$ty>::tanh(self)
                }

                fn epsilon() -> Self {
                    <$ty>::EPSILON
                }

                fn is_finite_value(self) -> bool {
                    <$ty>::is_finite(self)
                }
            }
        )*
    };
}

impl_float!(f32, f64);
