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

## Limite connue : levée depuis

Ce trait manipulait des `&[T]` et retournait des `Vec<T>`, ce qui rendait tout
backend GPU réel contre-productif : un transfert hôte ↔ device par opération.

Cette limite a été levée par l'issue #13 : `Backend` porte désormais un type
associé `Storage`. Voir [002 — Stockage résident sur le device](./002-device-storage.md).

## Conséquences

- `GpuBackend` est un squelette : il compile, il dispatche, et délègue au CPU.
  `GpuBackend::is_available()` retourne `false`, donc `Device::preferred()` rend le CPU.
- Aucune dépendance GPU n'est ajoutée tant qu'aucun kernel n'existe : `wgpu` et
  `cuda` sont des features vides, pour ne pas alourdir la compilation en CI.
- `CpuBackend` n'est jamais compilé hors du binaire : la feature `cpu` ne contrôle
  que l'accélération rayon, pas l'existence du backend, puisque tout repli en dépend.
- Le choix de l'enum plutôt que de `dyn Backend` s'est confirmé : un trait object
  aurait dû rester object-safe, ce qui aurait interdit le type associé `Storage`.
