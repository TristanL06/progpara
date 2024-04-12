Flou Gaussien

Implémentation :
- Pour faire RGB il faudrait 3 threads par pixel, on aurait un thread par canal de couleur. Pour une image on aurait donc 3 fois plus de threads que de pixels.
- Il y aura un accès concurrent en lecture, pas en écriture. CREW. En EW sur une nouvelle image il n'y aura pas de problème lié à la synchronisation.

- Il y aura donc besoin d'autant de threads que de pixels sur l'image.
- Il n'y aura pas de data race ou de problème de synchronisation.

