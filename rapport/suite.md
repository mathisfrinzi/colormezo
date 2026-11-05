[Sommaire](../README.md#le-rapport) · [← Fausses pistes](fausses_pistes.md) · [Reproduire →](reproduire.md)

# Suite

Par ordre d'utilité :

1. **Un arrêt anticipé réglé sur le jeu de réglage** pour ColorMeZO : sa précision culmine vers 1 500 lectures puis recule
   (partie 5.8). Mesurer s'il rejoint alors l'oracle, au lieu de la borne optimiste lue sur le test (0,814 sous dérive
   contre 0,835).
2. **Un pas réel qui ne grandit pas en fin de course** : le pas η/ℓ de ColorMeZO augmente d'un facteur 10 quand la perte
   baisse. Une décroissance de η programmée sur le nombre de pas, qui n'utilise aucune lecture du capteur, ne réintroduit
   pas le gain.
3. **Plus de graines sur le LLM** (10) et **un balayage de σ_g** (0,01 ; 0,02 ; 0,05 ; 0,1), pour retrouver, ou non, le
   croisement à ~0,02 du jouet.
4. **La phase sous dérive lente** : un décalage de phase en marche aléatoire, sur le jouet, à 1 lecture par pas
   (partie 6.4), puis à 100 et 1 000 paramètres.
5. **Chercher un capteur réel** : extremum seeking sous dérive de gain, surveillance de puissance sur puce en photonique
   intégrée. Si une voie de référence y règle déjà la dérive, le scénario de ce rapport n'a pas d'application.
6. **Un budget plus long** (10 000 à 20 000 lectures), pour se rapprocher du régime de MeZO.
7. **Régler η automatiquement pour ColorMeZO** : η × ℓ₀ (la perte lue au départ, à la décennie près) devrait donner le
   bon ordre de grandeur. Le pilote l'a utilisé ; la campagne ne l'a pas testé.
8. **Adam sur la pente ColorMeZO** sur le LLM : la meilleure méthode à gradient du jouet sous dérive (0,45), au prix de deux
   tampons de la taille des poids.

---

[Sommaire](../README.md#le-rapport) · [← Fausses pistes](fausses_pistes.md) · [Reproduire →](reproduire.md)
