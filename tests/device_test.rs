use kranium::tensor::backend::cpu::CpuBackend;
use kranium::tensor::{Device, DeviceKind, Tensor, TensorError};

fn cpu_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32, CpuBackend> {
    Tensor::from_data(data, shape, CpuBackend)
}

fn device_tensor(data: Vec<f32>, shape: &[usize], device: Device) -> Tensor<f32, Device> {
    Tensor::from_data(data, shape, device)
}

// --- Sélection du device -----------------------------------------------------

#[test]
fn cpu_is_always_available() {
    assert!(Device::is_available(DeviceKind::Cpu));
    assert_eq!(Device::cpu().kind(), DeviceKind::Cpu);
}

#[test]
fn preferred_never_panics_and_is_usable() {
    let device = Device::preferred();
    let t: Tensor<f32, Device> = Tensor::ones(&[2, 2], device);
    assert_eq!(t.data(), &[1.0; 4]);
}

#[test]
fn preferred_falls_back_to_cpu_without_a_gpu() {
    if !Device::is_available(DeviceKind::Gpu) {
        assert_eq!(Device::preferred().kind(), DeviceKind::Cpu);
    }
}

#[test]
fn try_new_reports_an_unavailable_backend() {
    match Device::try_new(DeviceKind::Gpu) {
        Ok(device) => assert_eq!(device.kind(), DeviceKind::Gpu),
        Err(e) => assert!(matches!(e, TensorError::BackendUnavailable { .. }), "{e:?}"),
    }
}

#[test]
fn try_new_cpu_always_succeeds() {
    assert_eq!(
        Device::try_new(DeviceKind::Cpu).unwrap().kind(),
        DeviceKind::Cpu
    );
}

#[test]
fn device_has_a_readable_name() {
    assert!(!Device::preferred().name().is_empty());
}

// --- Dispatch : mêmes résultats que le backend CPU direct --------------------

#[test]
fn dispatch_matches_cpu_backend_on_elementwise_ops() {
    let (a, b) = (vec![6.0, 8.0, 10.0, 12.0], vec![3.0, 2.0, 5.0, 4.0]);
    let shape = [2, 2];

    let (ca, cb) = (cpu_tensor(a.clone(), &shape), cpu_tensor(b.clone(), &shape));
    let device = Device::cpu();
    let (da, db) = (
        device_tensor(a, &shape, device),
        device_tensor(b, &shape, device),
    );

    assert_eq!(da.add(&db).data(), ca.add(&cb).data());
    assert_eq!(da.sub(&db).data(), ca.sub(&cb).data());
    assert_eq!(da.mul(&db).data(), ca.mul(&cb).data());
    assert_eq!(da.div(&db).data(), ca.div(&cb).data());
}

#[test]
fn dispatch_matches_cpu_backend_on_matmul_and_transpose() {
    let (a, b) = (
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    );

    let (ca, cb) = (
        cpu_tensor(a.clone(), &[2, 3]),
        cpu_tensor(b.clone(), &[3, 2]),
    );
    let device = Device::preferred();
    let (da, db) = (
        device_tensor(a, &[2, 3], device),
        device_tensor(b, &[3, 2], device),
    );

    assert_eq!(da.matmul(&db).data(), ca.matmul(&cb).data());
    assert_eq!(da.transpose().data(), ca.transpose().data());
}

#[test]
fn dispatch_matches_cpu_backend_on_constructors() {
    let device = Device::preferred();
    let zeros: Tensor<f32, Device> = Tensor::zeros(&[2, 3], device);
    let ones: Tensor<f32, Device> = Tensor::ones(&[2, 3], device);
    let new: Tensor<f32, Device> = Tensor::new(&[2, 3], device);

    assert_eq!(zeros.data(), &[0.0; 6]);
    assert_eq!(ones.data(), &[1.0; 6]);
    assert_eq!(new.data(), &[0.0; 6]);
}

#[test]
fn errors_are_reported_through_the_device_backend() {
    let device = Device::preferred();
    let a = device_tensor(vec![1.0, 2.0], &[2], device);
    let b = device_tensor(vec![1.0, 2.0, 3.0], &[3], device);
    assert!(matches!(
        a.try_add(&b).unwrap_err(),
        TensorError::ShapeMismatch { .. }
    ));
}

#[test]
fn a_device_tensor_exposes_its_backend() {
    let t: Tensor<f32, Device> = Tensor::zeros(&[2], Device::cpu());
    assert_eq!(t.backend().kind(), DeviceKind::Cpu);
}

// --- Couverture des trous relevés par tarpaulin ------------------------------

#[test]
fn data_mut_allows_in_place_edition() {
    let mut t = cpu_tensor(vec![1.0, 2.0, 3.0], &[3]);
    t.data_mut()[1] = 9.0;
    assert_eq!(t.data(), &[1.0, 9.0, 3.0]);
}

// --- Chemin GPU, uniquement quand un backend GPU est compilé -----------------

#[cfg(any(feature = "wgpu", feature = "cuda"))]
mod gpu_skeleton {
    use kranium::tensor::backend::gpu::GpuBackend;
    use kranium::tensor::{Device, DeviceKind, Tensor};

    #[test]
    fn gpu_is_not_available_until_kernels_land() {
        assert!(!GpuBackend::is_available());
        assert!(!Device::is_available(DeviceKind::Gpu));
        assert_eq!(Device::preferred().kind(), DeviceKind::Cpu);
    }

    #[test]
    fn the_gpu_variant_dispatches_to_the_skeleton() {
        let device = Device::Gpu(GpuBackend::new());
        assert_eq!(device.kind(), DeviceKind::Gpu);
        assert!(device.name().contains("skeleton"));

        let a: Tensor<f32, Device> = Tensor::from_data(vec![1.0, 2.0], &[2], device);
        let b: Tensor<f32, Device> = Tensor::from_data(vec![3.0, 4.0], &[2], device);
        assert_eq!(a.add(&b).data(), &[4.0, 6.0]);
        assert_eq!(
            a.reshape(&[2, 1]).matmul(&b.reshape(&[1, 2])).shape(),
            &[2, 2]
        );
    }
}
