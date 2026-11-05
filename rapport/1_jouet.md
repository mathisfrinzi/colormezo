[Sommaire](../README.md#le-rapport) · [← Introduction](0_introduction.md) · [Partie 2 →](2_derive.md)

# Partie 1. Le jouet : la dérive casse MeZO, pas ColorMeZO

> **Ce qu'il faut retenir**
>
> - Sous gain dérivant, MeZO tombe à **1 succès sur 20** ; ColorMeZO en garde **6 sur 20**, presque son niveau à gain
>   stable (7 sur 20).
> - À gain stable, MeZO capteur égale l'oracle (11 sur 20) : la quantification sur 12 bits ne gêne pas, c'est bien la dérive
>   qui casse MeZO.
> - L'oracle reste devant ColorMeZO, avec deux fois moins de lectures : la lecture couleur ne gagne que là où l'échelle
>   manque.

## 1.1 La tâche

Professeur-élève : un réseau 2 → 3 neurones tanh → 1 sortie linéaire (m = 12 paramètres), 100 points, cibles = professeur +
bruit 0,1 (le minimum n'est pas nul). **Succès** : perte vraie ≤ 1,05 × perte du professeur. Budget de 6 000 lectures
(3 000 pas), ε = 10⁻², pas borné à 0,5 en norme pour toutes les méthodes. η choisi dans {0,003 ; 0,01 ; 0,03 ; 0,1 ; 0,3}
sur 5 graines de réglage, puis 20 graines de test. Capteur de l'[introduction](0_introduction.md#le-capteur), σ_g = 0,05
par lecture.

## 1.2 Résultats

`src/bench_colormezo.py`, ~30 s CPU :

| Méthode | η | Succès | Ratio médian ℓ/ℓ* | Lectures médianes (succès) |
|:---|:---:|:---:|:---:|:---:|
| MeZO oracle (perte exacte) | 0,1 | **11/20** | 0,98 | 1 594 |
| MeZO, capteur stable | 0,1 | **11/20** | 0,98 | 1 734 |
| MeZO, capteur dérivant | 0,01 | 1/20 | 2,78 | 1 662 (n = 1) |
| ColorMeZO, capteur stable | 0,003 | 7/20 | 1,12 | 3 270 |
| **ColorMeZO, capteur dérivant** | 0,003 | **6/20** | 1,16 | 3 293 |

- **Sous dérive, MeZO s'effondre** : 1 succès, perte médiane à 2,8 fois le minimum. Son réglage se replie sur un η dix fois
  plus petit qu'à gain stable, sans suffire.
- **ColorMeZO tient** : il ne perd qu'un succès par rapport à son propre régime stable. Le gain, qui brouille les différences
  de valeurs, s'élimine des rapports.
- **À gain stable, la valeur reste meilleure** : 11 succès en ~1 600 lectures pour l'oracle, 7 en ~3 300 pour ColorMeZO. Le
  gradient de `ln ℓ` est un gradient relatif (divisé par ℓ), prudent loin du minimum, et coûte ici environ deux fois plus de
  lectures.

Résultats bruts : [`results/bench_quick.json`](../results/bench_quick.json).

## 1.3 Bilan de la partie 1

**Ce qui marche.** Sous gain dérivant par lecture, ColorMeZO garde 6 succès sur 20 quand MeZO n'en a plus qu'un.

**Les limites.**
- 20 graines, une tâche, une taille : 6/20 contre 1/20 est net, 6/20 contre 7/20 ne dit rien.
- Le capteur donne la décennie, ce qui avantage MeZO.
- Un seul σ_g : la partie 2 balaie la dérive.

---

[Sommaire](../README.md#le-rapport) · [← Introduction](0_introduction.md) · [Partie 2 →](2_derive.md)
