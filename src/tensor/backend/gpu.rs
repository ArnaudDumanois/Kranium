use super::cpu::CpuBackend;
use super::traits::{Backend, Numeric};

/// Skeleton GPU backend.
///
/// It exists so that the extension point is exercised by the compiler and so
/// that device selection has something to dispatch to. No kernel is implemented
/// yet: every operation delegates to [`CpuBackend`], and [`GpuBackend::is_available`]
/// reports `false` so that [`Device::preferred`](crate::tensor::Device::preferred)
/// keeps returning the CPU.
///
/// The [`Backend`] trait now carries a `Storage` associated type, so a real
/// implementation can keep its buffers on the device instead of copying host to
/// device and back on every operation — see `docs/adr/002-device-storage.md`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuBackend;

impl GpuBackend {
    pub fn new() -> Self {
        Self
    }

    /// Whether a usable GPU device was found. Always `false` until real kernels land.
    pub fn is_available() -> bool {
        false
    }

    pub fn name() -> &'static str {
        if cfg!(feature = "cuda") {
            "gpu (cuda, skeleton)"
        } else {
            "gpu (wgpu, skeleton)"
        }
    }
}

impl<T: Numeric> Backend<T> for GpuBackend {
    /// Host memory for now: without kernels there is nothing to keep on a
    /// device. A real implementation replaces this with a device buffer, which
    /// is precisely what the associated type exists for.
    type Storage = Vec<T>;

    fn to_device(&self, data: &[T]) -> Self::Storage {
        CpuBackend.to_device(data)
    }

    fn to_host(&self, storage: &Self::Storage) -> Vec<T> {
        CpuBackend.to_host(storage)
    }

    fn allocate(&self, shape: &[usize]) -> Self::Storage {
        CpuBackend.allocate(shape)
    }

    fn zeros(&self, shape: &[usize]) -> Self::Storage {
        CpuBackend.zeros(shape)
    }

    fn ones(&self, shape: &[usize]) -> Self::Storage {
        CpuBackend.ones(shape)
    }

    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        CpuBackend.add(a, b)
    }

    fn sub(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        CpuBackend.sub(a, b)
    }

    fn mul(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        CpuBackend.mul(a, b)
    }

    fn div(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        CpuBackend.div(a, b)
    }

    fn matmul(
        &self,
        a: &Self::Storage,
        a_shape: &[usize],
        b: &Self::Storage,
        b_shape: &[usize],
    ) -> Self::Storage {
        CpuBackend.matmul(a, a_shape, b, b_shape)
    }
}
