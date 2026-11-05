[Sommaire](../README.md#le-rapport) · [← Suite](suite.md)

# Reproduire

## Le jouet (parties 1 à 4), sur CPU avec NumPy

```bash
pip install -r requirements.txt
python3 src/bench_colormezo.py   # partie 1 → results/bench_quick.json        (~30 s)
python3 src/bench_sweep.py       # partie 2 → results/bench_sweep.json
python3 src/bench_etude3.py      # partie 3 → results/bench_etude3.json
python3 src/bench_etude4.py      # partie 4 → results/bench_etude4.json
python3 src/bench_phase.py       # partie 6 → results/bench_phase_50graines.json (~3 min 20)
python3 src/bench_phase.py --run1   # partie 6.4 → results/bench_phase.json (~6 min)
```

## Le pilote OPT-125M (partie 5.1), sur CPU en NumPy pur

Poids `pytorch_model.bin`, `vocab.json` et `merges.txt` de `facebook/opt-125m` à placer dans `pilot-llm/` (250 Mo, non
commités), puis :

```bash
cd pilot-llm
python3 opt_forward.py                    # contrôle : "The capital of France is" → token suivant
python3 opt_mezo_pilot.py lin_stable      # puis lin_drift, log_stable, log_drift (~175 s chacun)
```

Pilote à 5 graines : `python3 opt_mezo_seeds.py` (25 runs, ~165 s chacun ; figure : `python3 fig_seeds.py`).

## La campagne SST-2 (parties 5.2 à 5.8), sur GPU

[`campagne-llm/colormezo_llm_sst2.ipynb`](../campagne-llm/colormezo_llm_sst2.ipynb), dans Colab avec un GPU :

1. *Exécution → Modifier le type d'exécution → GPU*, puis *Tout exécuter*. La configuration par défaut est celle qui a
   tourné (`PRESET = "full"`, `STEP_NORM_MAX = None`, η de 10⁻⁸ à 10⁻⁴, réglage sur la perte). `CRITERE_REGLAGE =
   "precision"` règle sur la précision, comme l'analyse de la partie 5.7.
2. Après chaque méthode (réglage) et chaque graine (campagne finale), une archive horodatée des résultats est téléchargée
   par le navigateur (`TELECHARGER = True`), sans passer par Drive.
3. Après une déconnexion : déposer le dernier `colormezo_llm_results.json` dans le panneau *Fichiers* et relancer ; les runs
   déjà faits sont sautés.

Les presets : `"full"` (~2 h mesurées avec la grille à 9 valeurs), `"quick"` (1 graine de réglage, 1 000 lectures), `"smoke"` (petit OPT aléatoire et
données synthétiques, CPU, quelques secondes : vérifie le code, aucune valeur scientifique).

**La finale à η réglé (partie 5.8)** : `MODE_FINALE = True` saute le réglage, impose les η de `ETA_FIXE` (10⁻⁶ pour
l'oracle et MeZO, 3·10⁻⁶ pour ColorMeZO) et lance les 5 graines de test `SEEDS_TEST` (0 à 4), soit 25 runs de 2 000
lectures, 52 min mesurées sur GPU.

La campagne 1 se reproduit avec `STEP_NORM_MAX = 0.1` et `eta_grid = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5]`.

**Variante CPU, non exécutée.** [`campaign/campaign_sst2.py`](../campaign/campaign_sst2.py) et
[`campaign/run_campaign.ps1`](../campaign/run_campaign.ps1) décrivent une autre campagne SST-2, en NumPy pur sans PyTorch
(32 exemples few-shot, gabarit « Review: … Sentiment: » avec les verbaliseurs positive / negative, 250 phrases de dev,
1 000 lectures par run, ~1 h par run sur 2 cœurs CPU). Elle n'a pas été lancée ; les résultats de la partie 5 viennent tous
du carnet Colab ci-dessus.

## Les figures

```bash
python3 src/figures_rapport.py   # schémas, figures LLM et figure de la phase, depuis les JSON enregistrés (quelques secondes)
```

Le script reconstitue la trajectoire du gain `ln g(t)` depuis la graine du capteur (`np.random.default_rng(seed + 777)`) et
vérifie qu'elle redonne le `ln_g_max` enregistré pour chaque run. Les figures `e2_*`, `e3_*` et `e4_*` des parties 2 à 4
viennent des bancs du jouet.

---

[Sommaire](../README.md#le-rapport) · [← Suite](suite.md)
