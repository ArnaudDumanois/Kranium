use super::cpu::CpuBackend;
use super::traits::{Backend, Numeric};
use crate::tensor::error::{TensorError, TensorResult};

#[cfg(any(feature = "wgpu", feature = "cuda"))]
use super::gpu::GpuBackend;

/// The family a device belongs to, independent of the backend actually compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

/// Runtime backend selection.
///
/// `Device` implements [`Backend`] by dispatching on its variant, so a
/// `Tensor<T, Device>` picks its compute backend at runtime while every other
/// backend stays usable as a plain generic parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    Cpu,

    #[cfg(any(feature = "wgpu", feature = "cuda"))]
    Gpu(GpuBackend),
}

impl Device {
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// The best device available on this build, falling back to the CPU.
    pub fn preferred() -> Self {
        Self::try_new(DeviceKind::Gpu).unwrap_or(Self::Cpu)
    }

    /// Request a specific device family.
    ///
    /// Returns [`TensorError::BackendUnavailable`] when the backend was not
    /// compiled in or when no device was found.
    pub fn try_new(kind: DeviceKind) -> TensorResult<Self> {
        match kind {
            DeviceKind::Cpu => Ok(Self::Cpu),
            DeviceKind::Gpu => {
                #[cfg(any(feature = "wgpu", feature = "cuda"))]
                if GpuBackend::is_available() {
                    return Ok(Self::Gpu(GpuBackend::new()));
                }

                Err(TensorError::BackendUnavailable { requested: "gpu" })
            }
        }
    }

    pub fn is_available(kind: DeviceKind) -> bool {
        Self::try_new(kind).is_ok()
    }

    pub fn kind(&self) -> DeviceKind {
        match self {
            Self::Cpu => DeviceKind::Cpu,
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            Self::Gpu(_) => DeviceKind::Gpu,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => {
                if cfg!(feature = "cpu") {
                    "cpu (parallel)"
                } else {
                    "cpu (sequential)"
                }
            }
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            Self::Gpu(_) => GpuBackend::name(),
        }
    }
}

/// Dispatch every backend operation on the selected device.
macro_rules! dispatch {
    ($self:ident, $method:ident $(, $arg:ident)*) => {
        match $self {
            Device::Cpu => CpuBackend.$method($($arg),*),
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            Device::Gpu(backend) => backend.$method($($arg),*),
        }
    };
}

impl<T: Numeric> Backend<T> for Device {
    /// Every backend reachable through `Device` currently stores its elements in
    /// host memory. Once a GPU backend owns device buffers, this becomes an enum
    /// over the storages of the dispatched backends.
    type Storage = Vec<T>;

    fn to_device(&self, data: &[T]) -> Self::Storage {
        dispatch!(self, to_device, data)
    }

    fn to_host(&self, storage: &Self::Storage) -> Vec<T> {
        dispatch!(self, to_host, storage)
    }

    fn allocate(&self, shape: &[usize]) -> Self::Storage {
        dispatch!(self, allocate, shape)
    }

    fn zeros(&self, shape: &[usize]) -> Self::Storage {
        dispatch!(self, zeros, shape)
    }

    fn ones(&self, shape: &[usize]) -> Self::Storage {
        dispatch!(self, ones, shape)
    }

    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        dispatch!(self, add, a, b)
    }

    fn sub(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        dispatch!(self, sub, a, b)
    }

    fn mul(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        dispatch!(self, mul, a, b)
    }

    fn div(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        dispatch!(self, div, a, b)
    }

    fn matmul(
        &self,
        a: &Self::Storage,
        a_shape: &[usize],
        b: &Self::Storage,
        b_shape: &[usize],
    ) -> Self::Storage {
        dispatch!(self, matmul, a, a_shape, b, b_shape)
    }
}
