//! Prouve que `Tensor` fonctionne avec un backend dont le stockage n'est PAS
//! de la mémoire hôte. Sans ce backend, tous les backends réels ayant
//! `Storage = Vec<T>`, rien ne garantirait que l'accès hôte n'a pas été
//! re-supposé quelque part.

use kranium::tensor::{Backend, Numeric, Tensor};

/// Un buffer volontairement opaque : ni `Deref`, ni `as_slice`, ni indexation.
/// Le seul moyen d'en lire le contenu est de passer par le backend.
#[derive(Debug, Clone, PartialEq)]
pub struct OpaqueBuffer<T> {
    inner: Vec<T>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpaqueBackend;

impl<T: Numeric> Backend<T> for OpaqueBackend {
    type Storage = OpaqueBuffer<T>;

    fn to_device(&self, data: &[T]) -> Self::Storage {
        OpaqueBuffer {
            inner: data.to_vec(),
        }
    }

    fn to_host(&self, storage: &Self::Storage) -> Vec<T> {
        storage.inner.clone()
    }

    fn allocate(&self, shape: &[usize]) -> Self::Storage {
        OpaqueBuffer {
            inner: vec![T::default(); shape.iter().product()],
        }
    }

    fn zeros(&self, shape: &[usize]) -> Self::Storage {
        self.allocate(shape)
    }

    fn ones(&self, shape: &[usize]) -> Self::Storage {
        OpaqueBuffer {
            inner: vec![T::from(1u8); shape.iter().product()],
        }
    }

    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        self.zip(a, b, |x, y| x + y)
    }

    fn sub(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        self.zip(a, b, |x, y| x - y)
    }

    fn mul(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        self.zip(a, b, |x, y| x * y)
    }

    fn div(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        self.zip(a, b, |x, y| x / y)
    }

    fn matmul(
        &self,
        a: &Self::Storage,
        a_shape: &[usize],
        b: &Self::Storage,
        b_shape: &[usize],
    ) -> Self::Storage {
        let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
        let mut inner = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    sum += a.inner[i * k + l] * b.inner[l * n + j];
                }
                inner.push(sum);
            }
        }
        OpaqueBuffer { inner }
    }
}

impl OpaqueBackend {
    fn zip<T: Numeric, F: Fn(T, T) -> T>(
        &self,
        a: &OpaqueBuffer<T>,
        b: &OpaqueBuffer<T>,
        f: F,
    ) -> OpaqueBuffer<T> {
        OpaqueBuffer {
            inner: a
                .inner
                .iter()
                .zip(b.inner.iter())
                .map(|(&x, &y)| f(x, y))
                .collect(),
        }
    }
}

type OpaqueTensor = Tensor<f32, OpaqueBackend>;

fn t(data: Vec<f32>, shape: &[usize]) -> OpaqueTensor {
    Tensor::from_data(data, shape, OpaqueBackend)
}

// --- Aller-retour hôte / device ----------------------------------------------

#[test]
fn from_data_then_to_host_round_trips() {
    let tensor = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor.to_host(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn constructors_do_not_need_host_memory() {
    let zeros: OpaqueTensor = Tensor::zeros(&[2, 3], OpaqueBackend);
    let ones: OpaqueTensor = Tensor::ones(&[2, 3], OpaqueBackend);
    let new: OpaqueTensor = Tensor::new(&[2, 3], OpaqueBackend);

    assert_eq!(zeros.to_host(), vec![0.0; 6]);
    assert_eq!(ones.to_host(), vec![1.0; 6]);
    assert_eq!(new.to_host(), vec![0.0; 6]);
}

// --- Métadonnées : disponibles sans toucher au stockage ----------------------

#[test]
fn metadata_is_available_without_reading_the_storage() {
    let tensor = t(vec![0.0; 24], &[2, 3, 4]);
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.strides(), &[12, 4, 1]);
    assert_eq!(tensor.size(), 24);
    assert_eq!(tensor.ndim(), 3);
}

// --- Les opérations restent entièrement côté device --------------------------

#[test]
fn elementwise_ops_work_on_opaque_storage() {
    let a = t(vec![6.0, 8.0, 10.0, 12.0], &[2, 2]);
    let b = t(vec![3.0, 2.0, 5.0, 4.0], &[2, 2]);

    assert_eq!(a.add(&b).to_host(), vec![9.0, 10.0, 15.0, 16.0]);
    assert_eq!(a.sub(&b).to_host(), vec![3.0, 6.0, 5.0, 8.0]);
    assert_eq!(a.mul(&b).to_host(), vec![18.0, 16.0, 50.0, 48.0]);
    assert_eq!(a.div(&b).to_host(), vec![2.0, 4.0, 2.0, 3.0]);
}

#[test]
fn operators_work_on_opaque_storage() {
    let a = t(vec![1.0, 2.0], &[2]);
    let b = t(vec![3.0, 4.0], &[2]);
    assert_eq!((&a + &b).to_host(), vec![4.0, 6.0]);
}

#[test]
fn matmul_and_transpose_work_on_opaque_storage() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

    let product = a.matmul(&b);
    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(product.to_host(), vec![58.0, 64.0, 139.0, 154.0]);

    let transposed = a.transpose();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.to_host(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn reshape_works_on_opaque_storage() {
    let reshaped = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).reshape(&[3, 2]);
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.to_host(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn errors_are_reported_the_same_way() {
    let a = t(vec![1.0, 2.0], &[2]);
    let b = t(vec![1.0, 2.0, 3.0], &[3]);
    assert!(a.try_add(&b).is_err());
    assert!(a.try_transpose().is_err());
}

#[test]
fn clone_keeps_the_storages_independent() {
    let a = t(vec![1.0, 2.0], &[2]);
    let cloned = a.clone();
    assert_eq!(cloned.to_host(), a.to_host());
}
