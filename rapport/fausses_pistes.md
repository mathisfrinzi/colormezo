[Sommaire](../README.md#le-rapport) · [← Partie 7](7_synthese.md) · [Suite →](suite.md)

# Fausses pistes et corrections

Ce qui a été essayé, n'a pas marché ou a dû être corrigé avant interprétation, dans l'ordre chronologique.

| Où | Ce qui s'est passé | Ce qui a été fait |
|---|---|---|
| Partie 1 | Budget initial de 600 lectures : même l'oracle échouait 0/20. Le problème était trop dur pour le budget, pas pour les méthodes. | Budget porté à 6 000 lectures avant la campagne. Contrôle à gain stable (C1) ajouté après la première lecture des résultats, avant toute interprétation. |
| Partie 2 | ES non antithétique à 8 lectures par génération : plancher à 0,10 pour toutes les variantes, rien de discriminable. | Remplacé par 4 paires antithétiques ; seul ce banc est interprété. |
| Partie 3 | Piste « teinte = courbure » (brique 5.2 du projet couleur) pour réduire le facteur 2. | Abandonnée sur analyse : elle exige une échelle absolue (la cible m₀). Remplacée par le re-étalonnage geo. |
| Partie 4 | Broyden et BFGS restaient au point de départ même à gain stable : leur pas de différences finies (10⁻⁴, ~1,5·10⁻⁸ par défaut dans scipy) était plus petit que le pas de quantification du capteur (1/2¹² ≈ 2,4·10⁻⁴ en u). Le gradient FD était du bruit de quantification. | ε = 10⁻² calibré sur le contrôle stable, jacobienne FD explicite fournie à scipy, variante BFGS-log ajoutée. Consigné dans le docstring du script. |
| Partie 5, pilote | À η égal, ColorMeZO descendait 3 fois moins que MeZO. | Artefact d'échelle prévu (pas réel η/ℓ) ; corrigé dans le pilote par η × ℓ₀. Dans la campagne, η est réglé par méthode. |
| Partie 5, pilote | La meilleure descente du pilote à 1 graine (3,48 contre 3,59 pour MeZO sous dérive) venait d'une graine favorable : sur 5 graines, ColorMeZO mis à l'échelle fait jeu égal avec MeZO sous dérive (−0,115 contre −0,118 en médiane), et ne le bat que sur cette graine-là. | Pilote rétrogradé au rang d'essai de fonctionnement ; la dérive ne mord pas à 30 lectures, d'où la campagne à 1 000 à 2 000 lectures. |
| Partie 5, campagne | `load_dataset("glue", "sst2")` refusé par les versions récentes de `datasets` (identifiant `namespace/name` exigé). | Remplacé par `nyu-mll/glue`. |
| Partie 5, campagne 1 | La borne de norme du jouet (0,1) écrêtait 80 % des pas au réglage et 94 à 99 % en finale : seul le signe de la pente comptait, MeZO et ColorMeZO faisaient la même chose, aucune méthode n'apprenait (précision 0,516). | Borne retirée pour la campagne 2. Résultats conservés : [`campagne1_ecretage.json`](../campagne-llm/results/campagne1_ecretage.json). |
| Partie 5, campagne 2 | η* = 3·10⁻⁶ au bord haut de la grille pour les cinq méthodes. | Grille prolongée à 10⁻⁵, 3·10⁻⁵ et 10⁻⁴ ([`campagne3_grille_etendue.json`](../campagne-llm/results/campagne3_grille_etendue.json)). |
| Partie 5, grille étendue | Le critère de réglage écrit avant la campagne (perte après 1 000 lectures) retient 3·10⁻⁵ ou 10⁻⁵ : des η qui atteignent vite un plateau de perte (≈ 0,8) où le modèle répond au hasard (précision 0,47 à 0,51 en finale). | Classement refait sur la précision de réglage, mesurée aux mêmes runs ; les deux lectures sont données en partie 5.7. Le carnet propose désormais `CRITERE_REGLAGE = "precision"`. |
| Partie 5, finale | Sur le jeu de réglage (2 graines, 1 000 lectures), ColorMeZO sous dérive égalait l'oracle (0,748 contre 0,749). | Finale sur le jeu de test à 5 graines : 0,762 contre 0,833. Le constat a été retiré ; seule la robustesse à la dérive est retenue (partie 5.8). |
| Partie 6, run 1 | Sur 20 graines, la phase battait ColorMeZO à coût égal (8/20 contre 6/20), mais η* était au bord bas de la grille (0,003) pour les deux, et ε = 0,1 échouait partout. | Grille étendue à 10⁻⁴, 50 graines, ε = 10⁻² seul : 15/50 contre 12/50 (p = 0,65), pas d'avantage. Le run 1 reste reproductible (`--run1`). |
| Partie 6, prédictions | Deux prédictions écrites avant le run 1 sont fausses : ε = 0,1 devait égaler ou dépasser ColorMeZO (0/20), et un bruit de phase comparable à ε·pente devait détruire la méthode (5/20 à 10/20 selon la comptabilité). | Consignées telles quelles en partie 6.4. Le bruit testé est indépendant d'une lecture à l'autre ; une dérive lente de la phase reste à tester. |

---

[Sommaire](../README.md#le-rapport) · [← Partie 7](7_synthese.md) · [Suite →](suite.md)
