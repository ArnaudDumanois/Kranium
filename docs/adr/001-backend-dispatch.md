# 001 — Sélection de backend à l'exécution

- **Statut** : accepté
- **Date** : 2026-08-22
- **Contexte** : issue #7

## Contexte

`Tensor<T, B: Backend<T>>` choisit son backend à la compilation. L'issue #7 demande
une sélection à l'exécution avec repli sur le CPU, et un squelette de backend GPU.

## Décision

Un `enum Device` implémente `Backend<T>` et dispatche sur sa variante. Un
`Tensor<T, Device>` choisit donc son backend à l'exécution, sans que `Tensor`
ait besoin d'être modifié.

L'alternative — `Arc<dyn Backend<T>>` — a été écartée : elle impose une vtable sur
chaque opération, elle supprime le paramètre `B` de toute l'API publique existante,
et surtout elle obligerait le trait à rester object-safe pour toujours, ce qui
interdirait le type associé `Storage` décrit ci-dessous.

Contrepartie assumée : l'ensemble des backends adressables par `Device` est fermé
au crate. `Backend` reste implémentable librement à l'extérieur, `Device` n'est
qu'un aiguilleur pour les backends fournis.

## Limite connue : les données vivent en mémoire hôte

Le trait manipule des `&[T]` et retourne des `Vec<T>` :

```rust
fn add(&self, a: &[T], b: &[T]) -> Vec<T>;
```

Un backend GPU réel devrait donc copier hôte → device, calculer, puis recopier
device → hôte **à chaque opération**. Le transfert dominerait le calcul et un tel
backend serait plus lent que le CPU sur tout sauf des matmul très grandes.

Rendre le GPU rentable suppose que le tenseur possède un buffer résident sur le
device, donc un type associé :

```rust
trait Backend<T> {
    type Storage;
    fn to_device(&self, data: &[T]) -> Self::Storage;
    fn to_host(&self, storage: &Self::Storage) -> Vec<T>;
    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
}
```

Ce changement touche `Tensor` en entier (`data: Vec<T>` → `storage: B::Storage`),
et donc `data()`, `get`/`set`, `Index`, `Display`, `PartialEq`.

Il a été délibérément reporté hors de #7 pour garder la PR reviewable, mais son
coût croît avec chaque module bâti sur l'API actuelle. Il doit être traité **avant**
l'autograd (#9).

## Conséquences

- `GpuBackend` est un squelette : il compile, il dispatche, et délègue au CPU.
  `GpuBackend::is_available()` retourne `false`, donc `Device::preferred()` rend le CPU.
- Aucune dépendance GPU n'est ajoutée tant qu'aucun kernel n'existe : `wgpu` et
  `cuda` sont des features vides, pour ne pas alourdir la compilation en CI.
- `CpuBackend` n'est jamais compilé hors du binaire : la feature `cpu` ne contrôle
  que l'accélération rayon, pas l'existence du backend, puisque tout repli en dépend.
