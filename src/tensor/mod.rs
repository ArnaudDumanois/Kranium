pub mod backend;
mod core;
pub(crate) mod error;
mod ops;

pub use backend::device::{Device, DeviceKind};
pub use backend::traits::{Backend, Numeric};
pub use core::Tensor;
pub use error::{TensorError, TensorResult};
