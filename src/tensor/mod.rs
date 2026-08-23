pub mod backend;
mod core;
pub(crate) mod error;
pub mod numeric;
mod ops;

pub use backend::device::{Device, DeviceKind};
pub use backend::traits::Backend;
pub use core::Tensor;
pub use error::{TensorError, TensorResult};
pub use numeric::{Float, Numeric};
