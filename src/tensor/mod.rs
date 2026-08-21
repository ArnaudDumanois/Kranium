pub mod backend;
mod core;
mod error;
mod ops;

pub use backend::traits::{Backend, Numeric};
pub use core::Tensor;
pub use error::{TensorError, TensorResult};
