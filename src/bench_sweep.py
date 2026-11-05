"""
Étude 2. ColorMeZO : où l'avantage disparaît-il ? (suite pré-enregistrée)

É2.1 Balayage de la dérive, par lecture : sigma_g dans
     {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}, MeZO-capteur vs ColorMeZO,
     40 graines, eta re-regle a chaque point.
     Predicition preenregistree P1 : MeZO se degrade avec sigma_g,
     ColorMeZO reste plat jusqu'a la quantification 12 bits.

E2.2 Derive PAR PAS (gain constant au sein de la paire +/-, derive entre les
     pas), sigma_g = 0.05 et 0.1.
     Prediction preenregistree P2 : la difference lineaire n'est plus biaisee
     (le gain s'elimine en direction) mais l'echelle du pas derive
     aleatoirement (gain promene sur 3000 pas : ln g ~ +/-0.05*sqrt(3000)) ;
     MeZO doit se degrader par bruit de pas, ColorMeZO y est exactement
     invariant (ln(g*l+) - ln(g*l-) = ln l+ - ln l-).

E2.3 Point ColorES : ES population antithetique (4 paires, 8 lectures/pas),
     poids par valeur standardisee
     (sensible a la derive par lecture) vs poids par rangs (invariants par
     toute transformation monotone positive du gain).
     Prediction preenregistree P3 : l'ES a rangs est insensible a la derive
     -> la niche "couleur" serait vide pour les methodes a rangs.

Protocole identique a bench_colormezo : professeur-eleve m=12, criteres
1.05*l*, budget 6000 lectures, capteur 12 bits, 5 graines de reglage,
40 (E2.1/E2.2) ou 20 (E2.3) graines de test.
"""

import json
import numpy as np
from pathlib import Path

from bench_colormezo import (make_task, true_loss, teacher_loss,
                             EPS_SPSA, BUDGET, TOL, BITS, ETA_GRID,
                             TUNE_SEEDS, TEST_SEEDS)

TEST_SEEDS_40 = list(range(1000, 1040))


class Sensor2:
    """u = log10(g*l) quantifie BITS bits. derive_mode :
    'read' (a chaque lecture) / 'step' (une fois par pas, constante pendant
    la paire) / 'none'."""

    def __init__(self, rng, drift_mode="read", sigma_g=0.05):
        self.rng = rng
        self.mode = drift_mode
        self.sg = sigma_g
        self.ln_g = 0.0

    def _drift(self):
        if self.mode != "none":
            self.ln_g += self.rng.normal(0, self.sg)

    def read(self, loss):
        if loss <= 0:
            loss = 1e-300
        if self.mode == "read":
            self._drift()
        u = np.log10(np.exp(self.ln_g) * loss)
        return np.round(u * (2 ** BITS)) / (2 ** BITS)

    def linear_value(self, loss):
        return 10.0 ** self.read(loss)


def unwrap_delta(up, um):
    d = up - um
    if d > 0.5:
        d -= 1.0
    if d < -0.5:
        d += 1.0
    return d


def run_spsa(task, eta, mode, seed, drift_mode, sigma_g, budget=BUDGET):
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
        elif mode == "sensor":
            lp = sensor.linear_value(true_loss(v + EPS_SPSA * z, X, y))
            lm = sensor.linear_value(true_loss(v - EPS_SPSA * z, X, y))
            delta = lp - lm
        elif mode == "color":
            up = sensor.read(true_loss(v + EPS_SPSA * z, X, y))
            um = sensor.read(true_loss(v - EPS_SPSA * z, X, y))
            delta = np.log(10.0) * unwrap_delta(up, um)
        evals += 2
        g = (delta / (2 * EPS_SPSA)) * z
        step = -eta * g
        nrm = np.linalg.norm(step)
        if nrm > 0.5:
            step *= 0.5 / nrm
        v = v + step
        lt = true_loss(v, X, y)
        best = min(best, lt)
        if success_at is None and lt <= TOL * l_star:
            success_at = evals
    return {"success": success_at is not None,
            "evals_to_success": success_at,
            "final_ratio": best / l_star}


def run_es(task, eta, sigma_es, weights, seed, drift_mode, sigma_g,
           budget=BUDGET, k=8):
    """OpenAI-ES : theta += (eta/k) * sum(u_j * z_j), antithetique non utilise
    (k directions simples, comme MeZO a population). weights : 'value'
    (standardise) ou 'rank'."""
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
        Z = rng.normal(0, 1, (k, m))
        vals = np.empty(k)
        for j in range(k):
            vals[j] = sensor.linear_value(true_loss(v + sigma_es * Z[j], X, y))
        evals += k
        if weights == "value":
            sd = vals.std()
            u = (vals.mean() - vals) / (sd if sd > 0 else 1.0)
        else:  # rank : la plus petite perte doit avoir le poids le plus positif
            r = np.empty(k)
            order = np.argsort(vals)
            r[order] = np.arange(k - 1, -1, -1)   # meilleur = k-1
            u = r - (k - 1) / 2.0
        step = (eta / (k * sigma_es)) * (u @ Z)
        nrm = np.linalg.norm(step)
        if nrm > 0.5:
            step *= 0.5 / nrm
        v = v + step
        lt = true_loss(v, X, y)
        best = min(best, lt)
        if success_at is None and lt <= TOL * l_star:
            success_at = evals
    return {"success": success_at is not None,
            "evals_to_success": success_at,
            "final_ratio": best / l_star}


def run_es_antithetic(task, eta, sigma_es, weights, seed, drift_mode, sigma_g,
                    budget=BUDGET, k=4):
    """ES antithetique : k paires (+/- z_j), 2k lectures par pas. C'est la
    version qui tranche E2.3 (la version simple k=8 reste trop faible)."""
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
        Z = rng.normal(0, 1, (k, m))
        vals = np.empty(2 * k)
        dirs = np.empty((2 * k, m))
        for j in range(k):
            vals[j] = sensor.linear_value(true_loss(v + sigma_es * Z[j], X, y))
            dirs[j] = Z[j]
            vals[k + j] = sensor.linear_value(true_loss(v - sigma_es * Z[j], X, y))
            dirs[k + j] = -Z[j]
        evals += 2 * k
        if weights == "value":
            sd = vals.std()
            u = (vals.mean() - vals) / (sd if sd > 0 else 1.0)
        else:  # rank : la plus petite perte = poids le plus positif
            r = np.empty(2 * k)
            order = np.argsort(vals)
            r[order] = np.arange(2 * k - 1, -1, -1)
            u = r - (2 * k - 1) / 2.0
        step = (eta / (2 * k * sigma_es)) * (u @ dirs)
        nrm = np.linalg.norm(step)
        if nrm > 0.5:
            step *= 0.5 / nrm
        v = v + step
        lt = true_loss(v, X, y)
        best = min(best, lt)
        if success_at is None and lt <= TOL * l_star:
            success_at = evals
    return {"success": success_at is not None,
            "evals_to_success": success_at,
            "final_ratio": best / l_star}


def summarize(runs):
    succ = [r for r in runs if r["success"]]
    return {"success_rate": len(succ) / len(runs),
            "median_final_ratio": float(np.median([r["final_ratio"] for r in runs])),
            "median_evals": (float(np.median([r["evals_to_success"] for r in succ]))
                             if succ else None)}


def tune_spsa(mode, drift_mode, sigma_g):
    best_eta, best_med = None, np.inf
    for eta in ETA_GRID:
        ratios = [run_spsa(make_task(s), eta, mode, s, drift_mode, sigma_g)
                  ["final_ratio"] for s in TUNE_SEEDS]
        med = float(np.median(ratios))
        if med < best_med:
            best_med, best_eta = med, eta
    return best_eta


def main():
    out = {}

    # ---- E2.1 : balayage sigma_g, derive par lecture, 40 graines
    sweep = []
    for sg in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        row = {"sigma_g": sg, "drift_mode": "read"}
        for name, mode in [("mezo", "sensor"), ("colormezo", "color")]:
            eta = tune_spsa(mode, "read", sg)
            runs = [run_spsa(make_task(s), eta, mode, s, "read", sg)
                    for s in TEST_SEEDS_40]
            row[name] = {"eta": eta, **summarize(runs)}
        sweep.append(row)
        print("E2.1 sg=", sg, {k: round(v["success_rate"], 2) for k, v in row.items() if isinstance(v, dict)})
    out["sweep_read"] = sweep

    # ---- E2.2 : derive PAR PAS
    steps = []
    for sg in [0.05, 0.1]:
        row = {"sigma_g": sg, "drift_mode": "step"}
        for name, mode in [("mezo", "sensor"), ("colormezo", "color")]:
            eta = tune_spsa(mode, "step", sg)
            runs = [run_spsa(make_task(s), eta, mode, s, "step", sg)
                    for s in TEST_SEEDS_40]
            row[name] = {"eta": eta, **summarize(runs)}
        steps.append(row)
        print("E2.2 sg=", sg, {k: round(v["success_rate"], 2) for k, v in row.items() if isinstance(v, dict)})
    # oracle de reference sur les memes 40 graines
    eta = tune_spsa("exact", "none", 0.0)
    runs = [run_spsa(make_task(s), eta, "exact", s, "none", 0.0) for s in TEST_SEEDS_40]
    out["oracle_40seeds"] = {"eta": eta, **summarize(runs)}
    out["sweep_step"] = steps

    # ---- E2.3 : ES antithetique valeur vs rangs, derive par lecture sg=0.05,
    # 20 graines (la version simple k=8, run_es, reste au plancher 0.10 :
    # pas discriminante, conservee dans le JSON sous "es" pour trace)
    es = {}
    for name, weights, drift in [
            ("esanti_value_drift", "value", "read"),
            ("esanti_rank_drift", "rank", "read"),
            ("esanti_value_stable", "value", "none")]:
        best = None
        for eta in [0.003, 0.01, 0.03, 0.1]:
            for ses in [0.03, 0.1, 0.3]:
                ratios = [run_es_antithetic(make_task(s), eta, ses, weights, s,
                                            drift, 0.05)["final_ratio"]
                          for s in TUNE_SEEDS]
                med = float(np.median(ratios))
                if best is None or med < best[0]:
                    best = (med, eta, ses)
        _, eta, ses = best
        runs = [run_es_antithetic(make_task(s), eta, ses, weights, s, drift, 0.05)
                for s in TEST_SEEDS]
        es[name] = {"eta": eta, "sigma_es": ses, "k_pairs": 4, **summarize(runs)}
        print("E2.3", name, es[name])
    out["es_antithetic"] = es

    dest = Path(__file__).resolve().parents[1] / "results" / "bench_sweep.json"
    dest.write_text(json.dumps(out, indent=2))
    print("saved:", dest)


if __name__ == "__main__":
    main()
