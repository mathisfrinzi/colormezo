"""
Étude 3. ColorMeZO re-etalonne : combler l'ecart a l'oracle sans perdre
la robustesse a la derive du gain.

Motivation (mesure E1) : ColorMeZO estime le gradient de ln l, scale par 1/l
-> ~2x les lectures de l'oracle. Mais si l'on multiplie l'estimation par la
perte reconstruite depuis la MEME paire de lectures :

    l_geo = sqrt(10^u+ * 10^u-)
    gamma_geo = (delta ln l / (2 eps)) * l_geo
              ~= (l+ - l-) / (2 eps)      [moyenne geometrique : exact au 2e ordre]

on recupere la difference lineaire de MeZO, construite seulement a partir de
lectures couleur. Le gain g_t du capteur entre dans l_geo : au gain stable,
ColorMeZO-geo ~= MeZO stable (= oracle, 0,55) ; au gain derivant, l'echelle du
pas se met a errer exactement comme en E2.2 (P2) ; la question est ou passe
la frontiere.

P4 (pré-enregistrée) : a derive nulle ou tres lente, ColorMeZO-geo recupere
le niveau de l'oracle (~1 600 lectures) ; a derive par lecture >= ~0,02, il
redescend sous ColorMeZO (wandering d'echelle). Ni l'un ni l'autre ne domine :
frontier, pas de vainqueur unique.

Protocole : identique E1/E2 (tache m=12, critere 1.05*l*, budget 6 000
lectures, capteur 12 bits, 5 graines de reglage, 40 de test). Grille eta
etendue a {0,003 ; 0,01 ; 0,03 ; 0,1 ; 0,3 ; 1 ; 3} car l'echelle du gradient
change. Conditions : derive par lecture sg in {0, 0.005, 0.02, 0.05} et
derive par pas sg = 0.05. Methodes : colormezo (reference), colormezo_geo,
oracle.
"""

import json
import numpy as np
from pathlib import Path

from bench_colormezo import (make_task, true_loss, teacher_loss,
                              EPS_SPSA, BUDGET, TOL, BITS, TUNE_SEEDS)
from bench_sweep import Sensor2, unwrap_delta, summarize

TEST_SEEDS_40 = list(range(1000, 1040))
ETA_GRID_EXT = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def run_spsa3(task, eta, mode, seed, drift_mode, sigma_g, budget=BUDGET):
    X, y, teacher, v0 = task
    l_star = teacher_loss(X, y, teacher)
    rng = np.random.default_rng(seed + 555)
    sensor = Sensor2(np.random.default_rng(seed + 777), drift_mode, sigma_g)
    v = v0.copy()
    m = len(v)
    evals = 0
    best = true_loss(v, X, y)
    success_at = 0 if best <= TOL * l_star else None
    while evals < budget:
        sensor._drift() if sensor.mode == "step" else None
        z = rng.normal(0, 1, m)
        if mode == "exact":
            lp = true_loss(v + EPS_SPSA * z, X, y)
            lm = true_loss(v - EPS_SPSA * z, X, y)
            delta = lp - lm
            scale = 1.0
        elif mode == "color":
            up = sensor.read(true_loss(v + EPS_SPSA * z, X, y))
            um = sensor.read(true_loss(v - EPS_SPSA * z, X, y))
            delta = np.log(10.0) * unwrap_delta(up, um)
            scale = 1.0
        elif mode == "color_geo":
            up = sensor.read(true_loss(v + EPS_SPSA * z, X, y))
            um = sensor.read(true_loss(v - EPS_SPSA * z, X, y))
            delta = np.log(10.0) * unwrap_delta(up, um)
            # l_geo : moyenne geometrique des deux reconstructions (decade
            # connue, gain g_t inclus) -> re-etalonne le gradient de ln l
            scale = np.sqrt((10.0 ** up) * (10.0 ** um))
        else:
            raise ValueError(mode)
        evals += 2
        g = (delta / (2 * EPS_SPSA)) * scale * z
        step = -eta * g
        nrm = np.linalg.norm(step)
        if nrm > 0.5:            # garde-fou identique partout
            step *= 0.5 / nrm
        v = v + step
        lt = true_loss(v, X, y)
        best = min(best, lt)
        if success_at is None and lt <= TOL * l_star:
            success_at = evals
    return {"success": success_at is not None,
            "evals_to_success": success_at,
            "final_ratio": best / l_star}


def tune(mode, drift_mode, sigma_g):
    best_eta, best_med = None, np.inf
    for eta in ETA_GRID_EXT:
        ratios = [run_spsa3(make_task(s), eta, mode, s, drift_mode, sigma_g)
                  ["final_ratio"] for s in TUNE_SEEDS]
        med = float(np.median(ratios))
        if med < best_med:
            best_med, best_eta = med, eta
    return best_eta


def main():
    out = {"eta_grid": ETA_GRID_EXT, "test_seeds": 40}
    conditions = [("read", 0.0), ("read", 0.005), ("read", 0.02),
                  ("read", 0.05), ("step", 0.05)]
    rows = []
    for drift_mode, sg in conditions:
        row = {"drift_mode": drift_mode, "sigma_g": sg}
        for name, mode in [("colormezo", "color"),
                           ("colormezo_geo", "color_geo")]:
            eta = tune(mode, drift_mode, sg)
            runs = [run_spsa3(make_task(s), eta, mode, s, drift_mode, sg)
                    for s in TEST_SEEDS_40]
            row[name] = {"eta": eta, **summarize(runs)}
        rows.append(row)
        print("cond", drift_mode, sg,
              {k: round(v["success_rate"], 2) for k, v in row.items()
               if isinstance(v, dict)})
    out["conditions"] = rows

    eta = tune("exact", "none", 0.0)
    runs = [run_spsa3(make_task(s), eta, "exact", s, "none", 0.0)
            for s in TEST_SEEDS_40]
    out["oracle"] = {"eta": eta, **summarize(runs)}
    print("oracle", out["oracle"])

    dest = Path(__file__).resolve().parents[1] / "results" / "bench_etude3.json"
    dest.write_text(json.dumps(out, indent=2))
    print("saved:", dest)


if __name__ == "__main__":
    main()
