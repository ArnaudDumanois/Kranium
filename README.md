# Kranium

Bibliothèque de calcul tensoriel en Rust, écrite from scratch.

```rust
use kranium::{Device, Tensor};

let device = Device::preferred();
let a: Tensor<f32, Device> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device);
let b: Tensor<f32, Device> = Tensor::ones(&[2, 2], device);

let product = a.matmul(&b);
```

## Documentation

Tout se trouve dans le **[wiki du projet](https://github.com/ArnaudDumanois/Kranium/wiki)** :
[architecture](https://github.com/ArnaudDumanois/Kranium/wiki/Architecture),
[guide pour écrire un backend](https://github.com/ArnaudDumanois/Kranium/wiki/Backend-Guide),
[décisions d'architecture](https://github.com/ArnaudDumanois/Kranium/wiki/ADR-001-Backend-Dispatch)
et [roadmap](https://github.com/ArnaudDumanois/Kranium/wiki/Roadmap).

## Développement

```bash
cargo test                        # suite complète
cargo test --all-features         # squelettes GPU compris
cargo test --no-default-features  # repli CPU séquentiel
cargo clippy --all-targets --all-features -- -D warnings
cargo tarpaulin --fail-under 80   # couverture
```
