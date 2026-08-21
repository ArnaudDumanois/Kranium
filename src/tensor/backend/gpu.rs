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
/// Note that the [`Backend`] trait hands out `Vec<T>` living in host memory, so
/// a real implementation would have to copy host to device and back on every
/// single operation. Making the GPU worthwhile requires device-resident storage
/// first — see `docs/adr/001-backend-dispatch.md`.
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
    fn allocate(&self, shape: &[usize]) -> Vec<T> {
        CpuBackend.allocate(shape)
    }

    fn zeros(&self, shape: &[usize]) -> Vec<T> {
        CpuBackend.zeros(shape)
    }

    fn ones(&self, shape: &[usize]) -> Vec<T> {
        CpuBackend.ones(shape)
    }

    fn add(&self, a: &[T], b: &[T]) -> Vec<T> {
        CpuBackend.add(a, b)
    }

    fn sub(&self, a: &[T], b: &[T]) -> Vec<T> {
        CpuBackend.sub(a, b)
    }

    fn mul(&self, a: &[T], b: &[T]) -> Vec<T> {
        CpuBackend.mul(a, b)
    }

    fn div(&self, a: &[T], b: &[T]) -> Vec<T> {
        CpuBackend.div(a, b)
    }

    fn matmul(&self, a: &[T], a_shape: &[usize], b: &[T], b_shape: &[usize]) -> Vec<T> {
        CpuBackend.matmul(a, a_shape, b, b_shape)
    }
}
