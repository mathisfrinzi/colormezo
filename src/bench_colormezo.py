"""
Banc rapide : MeZO vs ColorMeZO sur un capteur à gain inconnu et dérivant.

Contexte (idée 24 du registre) : transposer la sonde couleur de
complex-colorimetry-optimization (parties 4-5) aux méthodes d'ordre zéro
de type MeZO (SPSA, Malladi et al. 2023).

Question pré-enregistrée Q1 : quand la perte ne se lit qu'à travers un
capteur logarithmique quantifié dont le gain dérive (échelle inconnue),
ColorMeZO (SPSA sur les différences de ln l, lues comme des rapports,
dépliées sur une décennie) réussit-il mieux que MeZO sur le même capteur
(SPA sur des différences linéaires de valeurs reconstruites) ?

Protocole :
- Réseau professeur-élève 2 -> 3 -> 1 (tanh, sortie linéaire), 12 paramètres.
- 100 points, cibles = professeur + bruit gaussien 0.1 (minimum non nul,
  comme la partie 5 du projet couleur).
- Critère de succès : perte vraie <= 1.05 * perte du professeur (à 5 % près).
- Budget : 6000 lectures du capteur (= 3000 pas SPSA à 2 lectures).
- Capteur identique pour mezo_sensor et colormezo :
    u = log10(g_t * l), quantifié 12 bits ; g_t dérive à CHAQUE lecture
    (ln g <- ln g + N(0, 0.05^2)). Le capteur ne rend jamais l.
    - mezo_sensor reçoit la reconstruction la plus favorable : l_hat = 10^u
      (décade connue), soit g_t * l arrondi -> ses différences linéaires
      mélangent deux gains différents.
    - colormezo n'utilise que la différence dépliée (mod 1 décade) :
      du = u+ - u- (dépliage a +/- 0.5 décennie), qui lit ln(l+/l-) ;
      le gain (constant ou dérivant) s'élimine des rapports.
- mezo_exact : SPSA sur la perte exacte (oracle, borne haute).
- Réglage : eta tiré d'une grille {0.003, 0.01, 0.03, 0.1, 0.3} sur
  5 graines de réglage ; 20 graines de test. eps = 1e-2 fixe.
"""

import json
import numpy as np
from pathlib import Path

H = 3            # neurones caches -> m = 4h = 12 parametres
N_DATA = 100
NOISE = 0.1
EPS_SPSA = 1e-2
BUDGET = 6000
TOL = 1.05
BITS = 12
SIGMA_G = 0.05   # derive du gain par lecture (en log)
ETA_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]
TUNE_SEEDS = [900, 901, 902, 903, 904]
TEST_SEEDS = list(range(1000, 1020))


# ---------------------------------------------------------------- réseau
def init_net(rng):
    return {"W1": rng.normal(0, 1, (H, 2)), "b1": rng.normal(0, 1, H),
            "W2": rng.normal(0, 1, (1, H)), "b2": np.zeros(1)}


def forward(net, X):
    A = np.tanh(net["W1"] @ X.T + net["b1"][:, None])   # (H, N)
    return (net["W2"] @ A + net["b2"]).ravel()          # (N,)


def params_to_vec(net):
    return np.concatenate([net["W1"].ravel(), net["b1"], net["W2"].ravel(), net["b2"]])


def vec_to_net(v):
    i = 0
    out = {}
    out["W1"] = v[i:i + 2 * H].reshape(H, 2); i += 2 * H
    out["b1"] = v[i:i + H]; i += H
    out["W2"] = v[i:i + H].reshape(1, H); i += H
    out["b2"] = v[i:i + 1]
    return out


def make_task(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N_DATA, 2))
    teacher = init_net(rng)
    y = forward(teacher, X) + rng.normal(0, NOISE, N_DATA)
    return X, y, teacher, params_to_vec(init_net(rng))


def true_loss(v, X, y):
    net = vec_to_net(v)
    r = forward(net, X) - y
    return float(np.mean(r * r))


def teacher_loss(X, y, teacher):
    r = forward(teacher, X) - y
    return float(np.mean(r * r))


# ---------------------------------------------------------------- capteur
class Sensor:
    """Lecture lumiere : u = log10(g * l) quantifie BITS bits, g derive a
    chaque lecture. Ne rend jamais l."""

    def __init__(self, rng, drift=True):
        self.rng = rng
        self.drift = drift
        self.ln_g = 0.0

    def read(self, loss):
        if loss <= 0:
            loss = 1e-300
        if self.drift:
            self.ln_g += self.rng.normal(0, SIGMA_G)
        u = np.log10(np.exp(self.ln_g) * loss)
        q = np.round(u * (2 ** BITS)) / (2 ** BITS)   # quantification
        return q

    def linear_value(self, loss):
        """Reconstruction la plus favorable pour MeZO : decade connue."""
        return 10.0 ** self.read(loss)


# ---------------------------------------------------------------- methodes
def unwrap_delta(u_plus, u_minus):
    d = u_plus - u_minus
    if d > 0.5:
        d -= 1.0
    if d < -0.5:
        d += 1.0
    return d


def run_spsa(task, eta, mode, seed, budget=BUDGET, drift=True):
    """mode: 'exact' | 'sensor' | 'color'."""
    X, y, teacher, v0 = task
    l_star = teacher_loss(X, y, teacher)
    rng = np.random.default_rng(seed + 555)
    sensor = Sensor(np.random.default_rng(seed + 777), drift=drift)
    v = v0.copy()
    m = len(v)
    evals = 0
    best = true_loss(v, X, y)
    evals_to_success = None
    if best <= TOL * l_star:
        evals_to_success = 0
    while evals < budget:
        z = rng.normal(0, 1, m)
        if mode == "exact":
            lp = true_loss(v + EPS_SPSA * z, X, y)
            lm = true_loss(v - EPS_SPSA * z, X, y)
            delta = lp - lm                       # difference lineaire exacte
        elif mode == "sensor":
            lp = sensor.linear_value(true_loss(v + EPS_SPSA * z, X, y))
            lm = sensor.linear_value(true_loss(v - EPS_SPSA * z, X, y))
            delta = lp - lm                       # melange deux gains
        elif mode == "color":
            up = sensor.read(true_loss(v + EPS_SPSA * z, X, y))
            um = sensor.read(true_loss(v - EPS_SPSA * z, X, y))
            delta = np.log(10.0) * unwrap_delta(up, um)  # ln(l+/l-)
        else:
            raise ValueError(mode)
        evals += 2
        g = (delta / (2 * EPS_SPSA)) * z
        step = -eta * g
        nrm = np.linalg.norm(step)
        if nrm > 0.5:            # garde-fou anti-explosion (identique partout)
            step *= 0.5 / nrm
        v = v + step
        lt = true_loss(v, X, y)
        best = min(best, lt)
        if evals_to_success is None and lt <= TOL * l_star:
            evals_to_success = evals
    return {"success": evals_to_success is not None,
            "evals_to_success": evals_to_success,
            "final_ratio": best / l_star}


# ---------------------------------------------------------------- campagne
def tune(mode, drift=True):
    best_eta, best_med = None, np.inf
    for eta in ETA_GRID:
        ratios = [run_spsa(make_task(s), eta, mode, s, drift=drift)["final_ratio"]
                  for s in TUNE_SEEDS]
        med = float(np.median(ratios))
        if med < best_med:
            best_med, best_eta = med, eta
    return best_eta, best_med


def main():
    out = {"config": {"h": H, "m": 4 * H, "n_data": N_DATA, "noise": NOISE,
                     "eps": EPS_SPSA, "budget": BUDGET, "tol": TOL,
                     "bits": BITS, "sigma_g": SIGMA_G,
                     "eta_grid": ETA_GRID, "n_tune": len(TUNE_SEEDS),
                     "n_test": len(TEST_SEEDS)}}
    # 5 variantes : oracle ; capteur stable vs dérivant ; pour chaque lecture.
    modes = {"mezo_exact": ("exact", False),
             "mezo_sensor_stable": ("sensor", False),
             "mezo_sensor_drift": ("sensor", True),
             "colormezo_stable": ("color", False),
             "colormezo_drift": ("color", True)}
    results = {}
    for name, (mode, drift) in modes.items():
        eta, med = tune(mode, drift=drift)
        runs = [run_spsa(make_task(s), eta, mode, s, drift=drift) for s in TEST_SEEDS]
        succ = [r for r in runs if r["success"]]
        results[name] = {
            "eta": eta,
            "tune_median_ratio": med,
            "success_rate": len(succ) / len(runs),
            "median_evals_to_success": (float(np.median([r["evals_to_success"] for r in succ]))
                                        if succ else None),
            "median_final_ratio": float(np.median([r["final_ratio"] for r in runs])),
            "runs": runs,
        }
        print(f"{name:12s} eta={eta:<6g} succes={len(succ)}/{len(runs)}  "
              f"ratio median={results[name]['median_final_ratio']:.3f}  "
              f"evals medianes={results[name]['median_evals_to_success']}")
    out["results"] = results
    dest = Path(__file__).resolve().parents[1] / "results" / "bench_quick.json"
    dest.write_text(json.dumps(out, indent=2))
    print("saved:", dest)


if __name__ == "__main__":
    main()
