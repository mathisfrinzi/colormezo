"""
Étude 4. À armes égales sur le capteur dérivant : Broyden classique et les
autres méthodes lisent le même capteur que ColorMeZO.

Question Q2 (pré-enregistrée) : quand TOUTES les méthodes (classiques ou
couleur) lisent le même capteur à gain dérivant (sigma_g = 0,05/lecture,
budget 6 000 lectures, critère 1,05*l*), lesquelles survivent, et la lecture
couleur bat-elle les survivantes ?

Prédictions P5 (pré-enregistrées) :
- P5a : les méthodes à différences linéaires et à comparaisons de valeurs
  s'effondrent (MeZO déjà mesuré : 1/20 ; BFGS-FD, Broyden-lin,
  Nelder-Mead attendus au plancher) : jacobienne et simplex mélangent des
  gains différents.
- P5b : les versions « log » des mêmes méthodes (Broyden-log : différences
  finies sur les lectures log, le gain devient un bruit additif ; Adam-log)
  sont partiellement robustes, plus que leurs versions linéaires, moins
  que la sonde par paires (le bruit de gain s'accumule dans la jacobienne
  au lieu de s'annuler par paire).
- P5c : l'ES-rangs (mesuré E2.3 : 0,50) reste la meilleure méthode robuste ;
  ColorMeZO ne bat pas le classement : l'avantage « couleur » se limite aux
  méthodes à différences par paires. Si Broyden-log bat ColorMeZO, c'est
  l'écho direct de la partie 6 du projet couleur (« Broyden classique reste
  moins cher »).

Méthodes (toutes : même capteur, budget 6 000 lectures, 20 graines) :
  mezo_lin   SPSA sur valeurs reconstruites (référence, connu : 1/20)
  colormezo  SPSA sur differences de ln l (dépliées)
  adam_lin   SPSA linéaire + Adam
  adam_log   SPSA log + Adam
  nm         Nelder-Mead (scipy) sur valeurs reconstruites
  bfgs_fd    BFGS (scipy) sur valeurs reconstruites, jacobienne FD explicite
  bfgs_log   idem sur lectures log (gain -> bruit additif)
  broyden_lin quasi-Newton (good Broyden) gradient par différences finies sur valeurs
  broyden_log idem sur lectures log (gain -> bruit additif)
  es_rank    ES antithétique à rangs (référence robuste, à re-mesurer ici)

Correction post-hoc documentée (avant toute interprétation) : la première
passe faisait échouer le contrôle C2 : Broyden et BFGS-FD restaient au point
de départ même à capteur stable. Cause : pas de différences finies (1e-4 pour
Broyden, ~1.5e-8 par défaut pour scipy) très inférieur au pas de
quantification du capteur en u (1/2^12 ~ 2.4e-4) ; le gradient FD était du
bruit de quantification pur. Corrigé : eps=1e-2 calibré sur le contrôle
stable, jacobienne FD explicite fournie à scipy. C'est un bug
d'implémentation, pas un résultat.

Le succès = perte vraie <= 1,05*l* sur un point évalué par la méthode
(suivi gratuit dans le wrapper : chaque lecture du capteur calcule la vraie
perte de toute façon ; aucun crédit de budget pour ce suivi).
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

from bench_colormezo import (make_task, true_loss, teacher_loss,
                              EPS_SPSA, BUDGET, TOL, TUNE_SEEDS, TEST_SEEDS)
from bench_sweep import Sensor2, unwrap_delta, summarize


class Budget(Exception):
    pass


def unwrap_full(u_plus, u_minus):
    """dépliage sur plusieurs décennies pour une vraie valeur log connue."""
    d = u_plus - u_minus
    return d - np.round(d)


# ------------------------------------------------------------- wrapper capteur
def make_wrapped(task, sensor, budget):
    """f(x) lu au capteur (valeurs reconstruites linéaires) + suivi gratuit
    de la meilleure vraie perte."""
    X, y, teacher, v0 = task
    l_star = teacher_loss(X, y, teacher)
    state = {"reads": 0, "best": np.inf, "l_star": l_star, "success_at": None}

    def f(x):
        if state["reads"] >= budget:
            raise Budget()
        v = np.asarray(x).ravel()
        lt = true_loss(v, X, y)
        state["reads"] += 1
        if lt < state["best"]:
            state["best"] = lt
        if state["success_at"] is None and lt <= TOL * l_star:
            state["success_at"] = state["reads"]
        return sensor.linear_value(lt)

    def f_log(x):
        return np.log10(f(x))     # valeur log reconstruite (décennie connue)

    return f, f_log, state


def result_of(state):
    return {"success": state["success_at"] is not None,
            "evals_to_success": state["success_at"],
            "final_ratio": state["best"] / state["l_star"]}


# ------------------------------------------------------------- methodes
def run_spsa_family(task, eta, mode, seed, use_adam=False, alpha=0.01, drift_mode="read", sigma_g=0.05):
    X, y, teacher, v0 = task
    l_star = teacher_loss(X, y, teacher)
    rng = np.random.default_rng(seed + 555)
    sensor = Sensor2(np.random.default_rng(seed + 777), drift_mode, sigma_g)
    v = v0.copy()
    m = len(v)
    evals = 0
    best = true_loss(v, X, y)
    success_at = 0 if best <= TOL * l_star else None
    mom = np.zeros(m)
    vel = np.zeros(m)
    t = 0
    while evals < BUDGET:
        z = rng.normal(0, 1, m)
        if mode == "lin":
            lp = sensor.linear_value(true_loss(v + EPS_SPSA * z, X, y))
            lm = sensor.linear_value(true_loss(v - EPS_SPSA * z, X, y))
            delta = lp - lm
        else:  # log
            up = sensor.read(true_loss(v + EPS_SPSA * z, X, y))
            um = sensor.read(true_loss(v - EPS_SPSA * z, X, y))
            delta = np.log(10.0) * unwrap_delta(up, um)
        evals += 2
        g = (delta / (2 * EPS_SPSA)) * z
        if not use_adam:
            step = -eta * g
        else:
            t += 1
            b1, b2 = 0.9, 0.999
            mom = b1 * mom + (1 - b1) * g
            vel = b2 * vel + (1 - b2) * g * g
            mhat = mom / (1 - b1 ** t)
            vhat = vel / (1 - b2 ** t)
            step = -alpha * mhat / (np.sqrt(vhat) + 1e-8)
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


def run_broyden(task, seed, mode, drift_mode="read", sigma_g=0.05):
    """Quasi-Newton (good Broyden sur l'inverse de la hessienne approchee),
    gradient par differences finies avant ; lin : valeurs reconstruites,
    log : lectures log (le gain devient un bruit additif)."""
    X, y, teacher, v0 = task
    l_star = teacher_loss(X, y, teacher)
    rng = np.random.default_rng(seed + 555)
    sensor = Sensor2(np.random.default_rng(seed + 777), drift_mode, sigma_g)
    v = v0.copy()
    m = len(v)
    # eps choisi au-dessus de la quantification du capteur : pas de
    # quantification en u = 1/2^12 ~ 2.4e-4 ; un pas de 1e-4 donnerait un
    # gradient fait uniquement de bruit de quantification (C2 echouait).
    # Calibration sur le controle stable : eps=1e-2 marche (2/3 graines).
    eps = 1e-2
    state = {"reads": 0, "best": np.inf, "l_star": l_star, "success_at": None}

    def read_lin(x):
        if state["reads"] >= BUDGET:
            raise Budget()
        lt = true_loss(x, X, y)
        state["reads"] += 1
        if lt < state["best"]:
            state["best"] = lt
        if state["success_at"] is None and lt <= TOL * l_star:
            state["success_at"] = state["reads"]
        return sensor.linear_value(lt)

    def read_log(x):
        return np.log10(read_lin(x))

    reader = read_lin if mode == "lin" else read_log
    scale = 1.0 / np.log(10.0) if mode == "log" else 1.0

    def grad(x):
        f0 = reader(x)
        g = np.empty(m)
        for i in range(m):
            xp = x.copy()
            xp[i] += eps
            g[i] = (reader(xp) - f0) / eps
        return g * scale if mode == "log" else g

    H = np.eye(m)
    try:
        g = grad(v)
        while state["reads"] < BUDGET:
            d = -H @ g
            nd = np.linalg.norm(d)
            if nd > 0.5:
                d *= 0.5 / nd
            # recherche lineaire minimale : 3 essais geometriques
            f0 = reader(v)
            cand = [0.25, 0.5, 1.0, 2.0]
            best_step, best_f = None, f0
            for s_mult in cand:
                vv = v + s_mult * d
                fv = reader(vv)
                if fv < best_f:
                    best_f, best_step = fv, s_mult
            if best_step is None:
                # pas descendant : reinitialiser H (restart du proxy)
                H = np.eye(m)
                v = v + 0.01 * rng.normal(0, 1, m)
                continue
            v_new = v + best_step * d
            g_new = grad(v_new)
            s = v_new - v
            yv = g_new - g
            sy = s @ yv
            if abs(sy) > 1e-12:
                H = H + np.outer(s - H @ yv, s) / sy   # good Broyden
            v, g = v_new, g_new
    except Budget:
        pass
    return result_of(state)


def run_scipy(task, seed, method, drift_mode="read", sigma_g=0.05):
    """Nelder-Mead ou BFGS (scipy) sur le capteur, budget par exception."""
    X, y, teacher, v0 = task
    sensor = Sensor2(np.random.default_rng(seed + 777), drift_mode, sigma_g)
    f, f_log, state = make_wrapped(task, sensor, BUDGET)
    opts = {"maxfev": BUDGET, "xatol": 1e-10, "fatol": 1e-10} if method == "Nelder-Mead" else {"maxiter": 10 ** 6}
    try:
        minimize(f, v0, method=method, options=opts)
    except Budget:
        pass
    return result_of(state)


def run_bfgs(task, seed, mode, drift_mode="read", sigma_g=0.05):
    """BFGS (scipy) avec jacobienne explicite par differences finies avant.
    Par defaut scipy fait du FD avec un pas ~1.5e-8, sous la resolution du
    capteur quantifie -> gradient nul. On fournit notre propre jacobienne
    avec le meme eps que Broyden (1e-2). mode 'lin' : valeurs reconstruites,
    'log' : valeurs log (gain -> bruit additif)."""
    X, y, teacher, v0 = task
    sensor = Sensor2(np.random.default_rng(seed + 777), drift_mode, sigma_g)
    f, f_log, state = make_wrapped(task, sensor, BUDGET)
    reader = f if mode == "lin" else f_log
    m = len(v0)
    eps = 1e-2

    def jac(x):
        f0 = reader(x)
        g = np.empty(m)
        for i in range(m):
            xp = np.array(x, dtype=float).ravel().copy()
            xp[i] += eps
            g[i] = (reader(xp) - f0) / eps
        return g

    try:
        minimize(f, v0, method="BFGS", jac=jac, options={"maxiter": 10 ** 6})
    except Budget:
        pass
    return result_of(state)


def tune_spsa(mode, use_adam):
    grid = ([0.003, 0.01, 0.03, 0.1] if use_adam else
            [0.003, 0.01, 0.03, 0.1, 0.3])
    key = "alpha" if use_adam else "eta"
    best = None
    for eta in grid:
        ratios = [run_spsa_family(make_task(s), eta, mode, s, use_adam, eta)
                  ["final_ratio"] for s in TUNE_SEEDS]
        med = float(np.median(ratios))
        if best is None or med < best[0]:
            best = (med, eta)
    return best[1]


def main():
    out = {"sigma_g_drift": 0.05, "budget": BUDGET, "test_seeds": len(TEST_SEEDS),
           "conditions": ["read_0.05", "none_0.0 (controle C2)"]}
    conds = [("read", 0.05), ("none", 0.0)]
    res = {}

    for drift_mode, sg in conds:
        tag = "drift" if drift_mode == "read" else "stable"
        for name, mode, use_adam in [("mezo_lin", "lin", False),
                                     ("colormezo", "log", False),
                                     ("adam_lin", "lin", True),
                                     ("adam_log", "log", True)]:
            grid = [0.003, 0.01, 0.03, 0.1] if use_adam else [0.003, 0.01, 0.03, 0.1, 0.3]
            best = None
            for eta in grid:
                ratios = [run_spsa_family(make_task(s), eta, mode, s, use_adam, eta,
                                          drift_mode, sg)["final_ratio"]
                          for s in TUNE_SEEDS]
                med = float(np.median(ratios))
                if best is None or med < best[0]:
                    best = (med, eta)
            _, eta = best
            runs = [run_spsa_family(make_task(s), eta, mode, s, use_adam, eta,
                                    drift_mode, sg) for s in TEST_SEEDS]
            res[f"{name}_{tag}"] = {"eta": eta, **summarize(runs)}
            print(f"{name}_{tag}", res[f"{name}_{tag}"], flush=True)

        for name, mode in [("broyden_lin", "lin"), ("broyden_log", "log")]:
            runs = [run_broyden(make_task(s), s, mode, drift_mode, sg)
                    for s in TEST_SEEDS]
            res[f"{name}_{tag}"] = summarize(runs)
            print(f"{name}_{tag}", res[f"{name}_{tag}"], flush=True)

        for name, method in [("nelder_mead", "Nelder-Mead")]:
            runs = [run_scipy(make_task(s), s, method, drift_mode, sg)
                    for s in TEST_SEEDS]
            res[f"{name}_{tag}"] = summarize(runs)
            print(f"{name}_{tag}", res[f"{name}_{tag}"], flush=True)

        for name, mode in [("bfgs_fd", "lin"), ("bfgs_log", "log")]:
            runs = [run_bfgs(make_task(s), s, mode, drift_mode, sg)
                    for s in TEST_SEEDS]
            res[f"{name}_{tag}"] = summarize(runs)
            print(f"{name}_{tag}", res[f"{name}_{tag}"], flush=True)

        # ES antithetique a rangs, re-mesure dans le meme harnais
        from bench_sweep import run_es_antithetic
        best = None
        for eta in [0.003, 0.01, 0.03, 0.1]:
            for ses in [0.03, 0.1, 0.3]:
                ratios = [run_es_antithetic(make_task(s), eta, ses, "rank", s,
                                            drift_mode, sg)["final_ratio"]
                          for s in TUNE_SEEDS]
                med = float(np.median(ratios))
                if best is None or med < best[0]:
                    best = (med, eta, ses)
        _, eta, ses = best
        runs = [run_es_antithetic(make_task(s), eta, ses, "rank", s, drift_mode, sg)
                for s in TEST_SEEDS]
        res[f"es_rank_{tag}"] = {"eta": eta, "sigma_es": ses, **summarize(runs)}
        print(f"es_rank_{tag}", res[f"es_rank_{tag}"], flush=True)

    out["methods"] = res
    dest = Path(__file__).resolve().parents[1] / "results" / "bench_etude4.json"
    dest.write_text(json.dumps(out, indent=2))
    print("saved:", dest)


if __name__ == "__main__":
    main()
