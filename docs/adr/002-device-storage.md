# 002 — Stockage résident sur le device

- **Statut** : accepté
- **Date** : 2026-08-22
- **Contexte** : issue #13, lève la limite consignée dans [001](./001-backend-dispatch.md)

## Contexte

Le trait `Backend<T>` manipulait des `&[T]` et retournait des `Vec<T>` : les
données vivaient donc toujours en mémoire hôte. Un backend GPU réel aurait dû
copier hôte → device, calculer, puis recopier device → hôte **à chaque
opération**, le transfert dominant le calcul.

## Décision

`Backend<T>` porte un type associé `Storage`, et `Tensor` stocke un
`B::Storage` au lieu d'un `Vec<T>`.

```rust
trait Backend<T> {
    type Storage: Clone + Debug;
    fn to_device(&self, data: &[T]) -> Self::Storage;
    fn to_host(&self, storage: &Self::Storage) -> Vec<T>;
    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
}
```

### Comment l'accès hôte survit

`data() -> &[T]` ne peut pas exister pour un buffer device. Plutôt que de le
supprimer partout, l'API est scindée par une contrainte sur le type associé :

```rust
// Toujours disponible
impl<T: Numeric, B: Backend<T>> Tensor<T, B> {
    pub fn to_host(&self) -> Vec<T> { ... }
}

// Uniquement quand le stockage EST de la mémoire hôte
impl<T: Numeric, B: Backend<T, Storage = Vec<T>>> Tensor<T, B> {
    pub fn data(&self) -> &[T] { ... }
    // data_mut, get, set, Index, IndexMut, PartialEq
}
```

Le compilateur retire donc les accès empruntés pour les backends à stockage
opaque, sans rien coûter aux backends hôte. Aucun test existant passant par
`Tensor` n'a eu besoin d'être modifié.

### Ce qui prouve l'abstraction

Tous les backends livrés ont `Storage = Vec<T>`. Rien ne garantirait donc que
l'accès hôte n'a pas été re-supposé quelque part. `tests/opaque_backend_test.rs`
définit un backend dont le stockage n'expose ni `Deref`, ni `as_slice`, ni
indexation : il ne compile que si `Tensor` est réellement opaque sur son
stockage.

## Conséquences

- `size()` se calcule depuis la shape et non depuis le stockage : plus aucun
  transfert pour lire une métadonnée.
- `transpose` rapatrie, permute, puis renvoie sur le device, faute de primitive
  dédiée dans le trait. C'est correct pour tout backend et ce sera le premier
  candidat à une primitive quand un vrai kernel GPU arrivera.
- `Display` passe par `to_host()`, donc un affichage de tenseur device implique
  un transfert. Acceptable : afficher est déjà une opération de debug.
- `Device::Storage` reste `Vec<T>` tant que `GpuBackend` délègue au CPU. Il
  deviendra une enum sur les stockages des backends aiguillés.
