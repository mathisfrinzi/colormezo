"""
Pilote idée 24 multi-graines : OPT-125M en NumPy pur, MeZO vs ColorMeZO
sous le même capteur couleur dérivant (u = log10(g*l), 12 bits,
ln g dérive de sigma_g = 0.05 par lecture).

Pré-enregistrement (écrit avant le lancement, aucune mesure regardée) :
- Reprise EXACTE du protocole du pilote 1 graine (opt_mezo_pilot.py) :
  phrase fixe "Paris is the capital of France.", 15 pas x 2 lectures,
  eps = 1e-3, eta = 1e-6 NON réglé, pas non borné en norme.
- 5 graines (0-4). La graine change le bruit z (zs = 1000 + 1000*seed + step)
  ET le chemin de dérive du capteur (rng seed+777). Les poids de départ sont
  identiques pour toutes les graines (restauration depuis une copie pristine).
- Conditions : lin_stable, lin_drift, log_stable, log_drift,
  log_drift_scaled (eta = 1e-6 * l0 : correspondance d'échelle a priori du
  pilote ; l0 = perte initiale, identique pour toutes les graines).
- Mesure : perte vraie à chaque pas (passe avant supplémentaire ; c'est la
  mesure, pas l'entraînement).
- Question Q-pilote5 : l'ordre observé sur 1 graine (dérive : log_scaled
  meilleur que lin ; stable : lin meilleur que log) survit-il à 5 graines ?
- Statistique honnête : n = 5 → descriptif uniquement (médiane, étendue,
  comparaison appariée par graine). Pas de test formel.
- Checkpoint : results_seeds.json réécrit après chaque run (reprise sûre).

Usage : python3 opt_mezo_seeds.py [n_seeds]
"""
import json
import time
import numpy as np
import opt_forward as F

SENT = "Paris is the capital of France."
EPS = 1e-3
ETA = 1e-6
SIGMA_G = 0.05
BITS = 12
STEPS = 15
CHUNK = 4_000_000

CONDS = ["lin_stable", "lin_drift", "log_stable", "log_drift", "log_drift_scaled"]
OUT = "results_seeds.json"


def trainable():
    return [(k, v) for k, v in F.SD.items() if "embed_positions" not in k]


def z_blocks(v, step_seed):
    """Générateur reproductif de z par blocs (RAM limitée : pas de z entier)."""
    rng = np.random.default_rng(step_seed)
    flat = v.ravel()
    for i in range(0, flat.size, CHUNK):
        yield i, rng.normal(0, 1, min(CHUNK, flat.size - i))


def add_cz(c, step_seed):
    """Applique c*z à tous les poids (c scalaire)."""
    for k, v in trainable():
        for i, z in z_blocks(v, step_seed):
            v.ravel()[i:i + z.size] += c * z.astype(np.float32)


class Sensor:
    def __init__(self, rng, drift):
        self.rng, self.drift, self.ln_g = rng, drift, 0.0

    def read(self, loss):
        if self.drift:
            self.ln_g += self.rng.normal(0, SIGMA_G)
        u = np.log10(np.exp(self.ln_g) * max(loss, 1e-300))
        return np.round(u * (2 ** BITS)) / (2 ** BITS)


def loss_of(ids):
    lg = F.forward(ids)[0]
    T = lg.shape[0] - 1
    lp = lg[:-1][np.arange(T), ids[0, 1:]]
    return float(-np.mean(lp - np.logaddexp.reduce(lg[:-1], axis=-1)))


def restore():
    """Recharge les poids pristine depuis le disque (la copie en RAM ferait
    exploser la limite cgroup : 655 Mo de poids + 655 Mo de copie > 1 Gio)."""
    import gc
    F.SD = None
    F.init()
    gc.collect()


def run(cond, seed):
    mode = "lin" if cond.startswith("lin") else "log"
    drift = cond.endswith("drift") or cond == "log_drift_scaled"
    ids = np.array([[2] + F.TOK.encode(SENT)])
    sensor = Sensor(np.random.default_rng(seed + 777), drift)
    hist = []
    ln10 = np.log(10.0)
    t0 = time.time()
    l0 = loss_of(ids)
    eta = ETA * l0 if cond.endswith("_scaled") else ETA
    for step in range(STEPS):
        zs = 1000 + 1000 * seed + step
        add_cz(+EPS, zs)
        u_p = sensor.read(loss_of(ids))
        add_cz(-2 * EPS, zs)          # theta+eps*z -> theta-eps*z
        u_m = sensor.read(loss_of(ids))
        if mode == "lin":
            delta = 10.0 ** u_p - 10.0 ** u_m
        else:
            d = u_p - u_m
            delta = ln10 * (d - np.round(d))
        g = delta / (2 * EPS)
        # depuis theta-eps*z : appliquer (eps - eta*g)*z = restauration + pas
        add_cz(EPS - eta * g, zs)
        lt = loss_of(ids)
        hist.append(lt)
    return {"cond": cond, "seed": seed, "mode": mode, "drift": drift,
            "eta": eta, "eps": EPS,
            "sigma_g": SIGMA_G if drift else 0.0, "steps": STEPS,
            "loss_init": round(l0, 4),
            "loss_final": round(hist[-1], 4),
            "delta": round(hist[-1] - l0, 4),
            "loss_hist": [round(h, 4) for h in hist],
            "seconds": round(time.time() - t0, 1)}


def main():
    import sys
    import time as _t
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    # budget temps : arrêt propre AVANT de commencer un run qui dépasserait
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
    t_start = _t.time()
    EXPECT = 170.0  # secondes mesurées par run (rechargement exclus)
    F.init()
    try:
        done = {r["cond"] + "_" + str(r["seed"])
                for r in json.load(open(OUT))}
    except (FileNotFoundError, json.JSONDecodeError):
        done = set()
    stopped = False
    for seed in range(n_seeds):
        if stopped:
            break
        for cond in CONDS:
            if cond + "_" + str(seed) in done:
                continue
            if _t.time() - t_start + EXPECT > budget:
                print("BUDGET ÉCOULÉ : état sauvegardé, relancer pour continuer",
                      flush=True)
                stopped = True
                break
            restore()
            r = run(cond, seed)
            done.add(cond + "_" + str(seed))
            rows = json.load(open(OUT)) if __import__("os").path.exists(OUT) else []
            rows.append(r)
            with open(OUT, "w") as f:
                json.dump(rows, f, indent=1)
            print(f"[done] {cond} seed {seed} : {r['loss_init']} -> "
                  f"{r['loss_final']} (delta {r['delta']}, {r['seconds']}s)",
                  flush=True)
    if not stopped:
        print("TOUS LES RUNS SONT FINIS", flush=True)


if __name__ == "__main__":
    main()
